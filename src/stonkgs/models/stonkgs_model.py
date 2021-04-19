# -*- coding: utf-8 -*-

"""STonKGs model on the fine-tuning classification task, assuming the model embeddings are pre-trained."""

import logging
import os
from typing import List, Optional

import mlflow
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from torch import nn
from transformers import (
    BertConfig,
    BertForPreTraining,
    BertModel,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.models.bert.modeling_bert import BertForPreTrainingOutput, BertLMPredictionHead
from transformers.trainer_utils import get_last_checkpoint

from stonkgs.constants import (
    EMBEDDINGS_PATH,
    MLFLOW_TRACKING_URI,
    NLP_MODEL_TYPE,
    PRETRAINING_PREPROCESSED_DF_PATH,
    STONKGS_PRETRAINING_DIR,
)
from stonkgs.models.kg_baseline_model import _prepare_df

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_pre_training_data(
    pretraining_preprocessed_path: str = PRETRAINING_PREPROCESSED_DF_PATH,
    dataset_format: str = 'torch',
):
    """Create a pytorch dataset based on a preprocessed dataframe for the pretraining dataset."""
    # Load the pickled preprocessed dataframe
    pretraining_preprocessed_df = pd.read_pickle(pretraining_preprocessed_path)
    # TODO (later on): Use device='cuda' for format_kwargs in set_format?
    pretraining_dataset = Dataset.from_pandas(pretraining_preprocessed_df)
    pretraining_dataset.set_format(dataset_format)

    return pretraining_dataset


def get_train_test_splits(
    train_data: pd.DataFrame,
    type_column_name: str = "class",
    random_seed: int = 42,
    n_splits: int = 5,
) -> List:
    """Return train/test indices for n_splits many splits based on the fine-tuning dataset that is passed."""
    # Leave out the label in the dataset
    X = train_data.drop(type_column_name, axis=1)  # noqa: N806
    y = train_data[type_column_name]

    # TODO: think about whether a validation split is necessary
    # For now: implement stratified train/test splits
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=False)

    return [[train_idx, test_idx] for train_idx, test_idx in skf.split(X, y)]


class STonKGsELMPredictionHead(BertLMPredictionHead):
    """Custom masked entity and language modeling (ELM) head used to predict both entities and text tokens."""

    def __init__(self, config):
        """Initialize the ELM head based on the (hyper)parameters in the provided BertConfig."""
        super().__init__(config)

        # There are two different "decoders": The first half of the sequence is projected onto the dimension of
        # the text vocabulary index, the second half is projected onto the dimension of the kg vocabulary index
        # 1. Text decoder
        self.text_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 2. Entity decoder
        self.entity_decoder = nn.Linear(config.hidden_size, config.kg_vocab_size, bias=False)

        # Determine half of the maximum sequence length based on the config
        self.half_length = config.max_position_embeddings // 2

        # Set the biases differently for the decoder layers
        self.text_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.entity_bias = nn.Parameter(torch.zeros(config.kg_vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.text_bias = self.text_bias
        self.decoder.entity_bias = self.entity_bias

    def forward(self, hidden_states):
        """Map hidden states to values for the text vocab (first half) and kg vocab (second half)."""
        # Common transformations (dense layer, layer norm + activation function) performed on both text and entities
        # transform is initialized in the parent BertLMPredictionHead class
        hidden_states = self.transform(hidden_states)

        # The first half is processed with the text decoder, the second with the entity decoder to map to the text
        # vocab size and kg vocab size, respectively
        text_hidden_states_to_vocab = self.text_decoder(hidden_states[:, :self.half_length])
        ent_hidden_states_to_kg_vocab = self.entity_decoder(hidden_states[:, self.half_length:])

        return text_hidden_states_to_vocab, ent_hidden_states_to_kg_vocab


class STonKGsForPreTraining(BertForPreTraining):
    """Create the pre-training part of the STonKGs model based on both text and entity embeddings."""

    def __init__(self, nlp_model_type, kg_embedding_dict):
        """Initialize the model architecture components of STonKGs."""
        # Add the number of KG entities to the default config of a standard BERT model
        config = BertConfig.from_pretrained(nlp_model_type)
        config.update({'kg_vocab_size': len(kg_embedding_dict)})
        # Initialize the underlying BertForPreTraining model that will be used to build the STonKGs Transformer layers
        super().__init__(config)

        # Override the standard MLM head: In the underlying BertForPreTraining model, change the MLM head to the custom
        # STonKGsELMPredictionHead so that it can be used on the concatenated text/entity input
        self.cls.predictions = STonKGsELMPredictionHead(config)

        # LM backbone initialization (pre-trained BERT to get the initial embeddings) based on the specified
        # nlp_model_type (e.g. BioBERT)
        self.lm_backbone = BertModel.from_pretrained(nlp_model_type)
        # Freeze the parameters of the LM backbone so that they're not updated during training
        # (We only want to train the STonKGs Transformer layers)
        for param in self.lm_backbone.parameters():
            param.requires_grad = False
        # Get the separator token id (needed in the forward pass) from a nlp_model_type specific tokenizer
        self.lm_sep_id = BertTokenizer.from_pretrained(nlp_model_type).sep_token_id
        # Get the mask token id (needed in the forward pass) from a nlp_model_type specific tokenizer
        self.lm_mask_id = BertTokenizer.from_pretrained(nlp_model_type).mask_token_id

        # KG backbone initialization
        # TODO: move that to a custom dataset class maybe?
        # Generate numeric indices for the KG node names (iterating .keys() is deterministic)
        self.kg_idx_to_name = {i: key for i, key in enumerate(kg_embedding_dict.keys())}
        # Initialize KG index to embeddings based on the provided kg_embedding_dict
        self.kg_backbone = {i: torch.tensor(kg_embedding_dict[self.kg_idx_to_name[i]])
                            for i in self.kg_idx_to_name.keys()}
        # Add the MASK (LM backbone) embedding vector to the KG backbone for masked entity tokens
        # i = -1 indicates that this entity is masked, therefore it is replaced with the embedding vector of the
        # [MASK] token
        # [0][0][0] is required to get the shape from batch x seq_len x hidden_size to hidden_size
        self.kg_backbone[-1] = self.lm_backbone(torch.tensor([[self.lm_mask_id]]))[0][0][0]

    def forward(
        self,
        # required parameters
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        masked_lm_labels=None,
        ent_masked_lm_labels=None,
        next_sentence_labels=None,
        # determined the return type
        return_dict=None,
        # position_ids will usually stay None here so that default BERT position_ids are used
        position_ids=None,
        # in case certain masks do need to be canceled out
        head_mask=None,
    ):
        """Perform one forward pass for a given sequence of text_input_ids + ent_input_ids."""
        # The code is based on CoLAKE: https://github.com/txsun1997/CoLAKE/blob/master/pretrain/model.py

        # Use the LM backbone to get the pre-trained token embeddings
        # batch x half_length x hidden_size
        # The first element of the returned tuple from the LM backbone forward() pass is the sequence of hidden states
        token_embeddings = self.lm_backbone(input_ids[:, :self.cls.predictions.half_length])[0]

        # Use the KG backbone to obtain the pre-trained entity embeddings
        # batch x half_length x hidden_size
        ent_embeddings = torch.stack([
            # for each numeric index in the random walks sequence: get the embedding vector from the KG backbone
            torch.stack([self.kg_backbone[i.item()] for i in j])
            # for each example in the batch: get the random walks sequence
            for j in input_ids[:, self.cls.predictions.half_length:]],
        )

        # TODO (later on): Just use random walks of length 127 and concatenate with [SEP] instead
        # Replace the middle with the [SEP] embedding vector to distinguish between the first and second random walk
        # sequences and also add the [SEP] embedding vector at the end
        # [0][0][0] is required to get the shape from batch x seq_len x hidden_size to hidden_size
        ent_embeddings[len(ent_embeddings) // 2] = self.lm_backbone(torch.tensor([[self.lm_sep_id]]))[0][0][0]
        ent_embeddings[-1] = self.lm_backbone(torch.tensor([[self.lm_sep_id]]))[0][0][0]

        # Concatenate token and entity embeddings obtained from the LM and KG backbones and cast to float
        # batch x seq_len x hidden_size
        inputs_embeds = torch.cat([token_embeddings, ent_embeddings], dim=1).type(torch.FloatTensor)

        # Get the hidden states from the basic STonKGs Transformer layers
        # batch x half_length x hidden_size
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            return_dict=None,
        )
        # batch x seq_len x hidden_size
        sequence_output, pooled_output = outputs[:2]

        # Generate the prediction scores (mapping to text and entity vocab sizes + NSP) for the training objectives
        # Seq_relationship_score = NSP score
        # prediction_scores = Text MLM and Entity "MLM" scores
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        # The custom SToNKGsELMPredictionHead returns a pair of prediction score sequences for tokens and entities,
        # respectively
        token_prediction_scores, entity_predictions_scores = prediction_scores

        # Calculate the loss
        total_loss = None
        if masked_lm_labels is not None and ent_masked_lm_labels is not None and next_sentence_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 1. Text-based MLM
            masked_lm_loss = loss_fct(
                token_prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            )
            # 2. Entity-based masked "language" (entity) modeling
            ent_masked_lm_loss = loss_fct(
                entity_predictions_scores.view(-1, self.config.kg_vocab_size),
                ent_masked_lm_labels.view(-1),
            )
            # 3. Next "sentence" loss: Whether a text and random walk sequence belong together or not
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2),
                next_sentence_labels.view(-1),
            )
            total_loss = masked_lm_loss + ent_masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def pretrain_stonkgs(
    batch_size: int = 8,
    lr: float = 1e-4,
    logging_dir: Optional[str] = MLFLOW_TRACKING_URI,
    logging_steps: int = 100,
    max_steps: int = 10000,
    overwrite_output_dir: bool = False,
    save_limit: int = 5,
    save_steps: int = 5000,
    training_dir: str = STONKGS_PRETRAINING_DIR,
):
    """Run the pre-training procedure for the STonKGs model based on the transformers Trainer and TrainingArguments."""
    # Part of this code is taken from
    # https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py

    # Logging with mlflow
    # End previous run
    mlflow.end_run()
    # Initialize mlflow run, set tracking URI to use the same experiment for all runs,
    # so that one can compare them
    mlflow.set_tracking_uri(logging_dir)
    mlflow.set_experiment('STonKGs Pre-Training')

    # Initialize the STonKGs model
    kg_embed_dict = _prepare_df(EMBEDDINGS_PATH)
    stonkgs_model = STonKGsForPreTraining(NLP_MODEL_TYPE, kg_embed_dict)

    # Initialize the TrainingArguments
    training_args = TrainingArguments(
        output_dir=training_dir,
        overwrite_output_dir=overwrite_output_dir,
        do_train=True,
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,  # Use max_steps rather than num_training_epochs
        learning_rate=lr,  # Default is to use that lr with a linear scheduler
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_limit,
        report_to=['mlflow'],
    )

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. ",
                "Use --overwrite_output_dir to overcome.",
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change ",
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.",
            )

    # Initialize the dataset
    pretraining_data = _load_pre_training_data()

    # Initialize the Trainer
    trainer = Trainer(
        model=stonkgs_model,
        args=training_args,
        train_dataset=pretraining_data,
    )
    # And train STonKGs to the moon
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    # Log the number of pre-training samples
    metrics["train_samples"] = len(pretraining_data)
    # Log all metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    # TODO: divide this class into three classes: stonkgs_model, stonkgs_pretraining, stonkgs_finetuning
    pretrain_stonkgs(overwrite_output_dir=True)
