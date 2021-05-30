# -*- coding: utf-8 -*-

"""Runs the STonKGs model on the fine-tuning classification task, assuming the model embeddings are pre-trained.

Run with:
python -m src.stonkgs.models.stonkgs_finetuning
"""

import logging
import os
from typing import List

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert import BertModel, BertTokenizer, BertTokenizerFast

from stonkgs.constants import (
    EMBEDDINGS_PATH,
    NLP_MODEL_TYPE,
    ORGAN_DIR,
    # PRETRAINED_STONKGS_DUMMY_PATH,
    RANDOM_WALKS_PATH,
    VOCAB_FILE,
)
from stonkgs.models.kg_baseline_model import _prepare_df
from stonkgs.models.stonkgs_model import STonKGsForPreTraining

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_train_test_splits(
    train_data: pd.DataFrame,
    type_column_name: str = "class",
    random_seed: int = 42,
    n_splits: int = 5,
) -> List:
    """Return train/test indices for n_splits many splits based on the fine-tuning dataset that is passed."""
    # Leave out the label in the dataset
    data = train_data.drop(type_column_name, axis=1)
    labels = train_data[type_column_name]

    # Implement stratified train/test splits
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=False)

    return [[train_idx, test_idx] for train_idx, test_idx in skf.split(data, labels)]


def preprocess_fine_tuning_data(
    train_data_path: str,
    embedding_name_to_vector_path: str = EMBEDDINGS_PATH,
    embedding_name_to_random_walk_path: str = RANDOM_WALKS_PATH,
    nlp_model_type: str = NLP_MODEL_TYPE,
    sep_id: int = 102,
    unk_id: int = 100,
) -> pd.DataFrame:
    """Generate input_ids, attention_mask, token_type_ids etc. based on the source, target, evidence columns."""
    # Load the KG embedding dict to convert the names to numeric indices
    kg_embed_dict = _prepare_df(embedding_name_to_vector_path)
    kg_name_to_idx = {key: i for i, key in enumerate(kg_embed_dict.keys())}
    # Load the random walks for each node
    random_walk_dict = _prepare_df(embedding_name_to_random_walk_path)
    # Convert random walk sequences to list of numeric indices
    random_walk_idx_dict = {k: [kg_name_to_idx[node] for node in v] for k, v in random_walk_dict.items()}

    # Load the raw fine-tuning dataset with source, target and evidence
    unprocessed_df = pd.read_csv(train_data_path, sep='\t', usecols=["source", "target", "evidence", "class"])

    # Check how many nodes in the fine-tuning dataset are not covered by the learned KG embeddings
    number_of_pre_training_nodes = len(set(unprocessed_df["source"]).union(set(unprocessed_df["target"])))
    if number_of_pre_training_nodes > len(kg_embed_dict):
        logger.warning(f'{number_of_pre_training_nodes - len(kg_embed_dict)} out of {number_of_pre_training_nodes}'
                       f'nodes are not covered by the embeddings learned in the pretraining dataset')

    # Get the length of the text or entity embedding sequences (2 random walks + 2 = entity embedding sequence length)
    random_walk_length = len(next(iter(random_walk_idx_dict.values())))
    half_length = random_walk_length * 2 + 2

    # Initialize a FAST tokenizer if it's the default one (BioBERT)
    if nlp_model_type == NLP_MODEL_TYPE:
        # Initialize the fast tokenizer for getting the text token ids
        tokenizer = BertTokenizerFast(vocab_file=VOCAB_FILE)
    else:
        # Initialize a slow tokenizer used for getting the text token ids
        tokenizer = BertTokenizer.from_pretrained(nlp_model_type)

    # Initialize the preprocessed data
    fine_tuning_preprocessed = []

    # Log progress with a progress bar
    for _, row in tqdm(
        unprocessed_df.iterrows(),
        total=unprocessed_df.shape[0],
        desc='Preprocessing the fine-tuning dataset',
    ):
        # 1. "Token type IDs": 0 for text tokens, 1 for entity tokens
        token_type_ids = [0] * half_length + [1] * half_length

        # 2. Tokenization for getting the input ids and attention masks for the text
        # Use encode_plus to also get the attention mask ("padding" mask)
        encoded_text = tokenizer.encode_plus(
            row['evidence'],
            padding='max_length',
            truncation=True,
            max_length=half_length,
        )
        text_token_ids = encoded_text['input_ids']
        text_attention_mask = encoded_text['attention_mask']

        # 3. Get the random walks sequence and the node indices, add the SEP (usually with id=102) in between
        # Use a sequence of UNK tokens if the node is not contained in the dictionary of the nodes from pre-training
        random_w_source = random_walk_idx_dict[
            row['source']
        ] if row['source'] in random_walk_idx_dict.keys() else [unk_id] * random_walk_length
        random_w_target = random_walk_idx_dict[
            row['target']
        ] if row['target'] in random_walk_idx_dict.keys() else [unk_id] * random_walk_length
        random_w_ids = random_w_source + [sep_id] + random_w_target + [sep_id]

        # 4. Total attention mask (attention mask is all 1 for the entity sequence)
        attention_mask = text_attention_mask + [1] * half_length

        # 5. Total input_ids = half text ids + half entity ids
        input_ids = text_token_ids + random_w_ids

        # Add all the features to the preprocessed data
        fine_tuning_preprocessed.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,  # Remove the MLM, ELM and NSP labels since it's not needed anymore
            'label': row['class'],  # Add the annotation/relation label for fine-tuning instead
        })

    # Put the preprocessed data into a dataframe
    fine_tuning_preprocessed_df = pd.DataFrame(fine_tuning_preprocessed)

    return fine_tuning_preprocessed_df


class STonKGsForSequenceClassification(STonKGsForPreTraining):
    """Create the fine-tuning part of the STonKGs model based the pre-trained STonKGs model.

    Note that this class inherits from STonKGsForPreTraining rather than PreTrainedModel, thereby it's deviating from
    the typical huggingface inheritance logic of the fine-tuning classes.
    """

    def __init__(self, config):
        """Initialize the STonKGs sequence classification model based on the pre-trained STonKGs model architecture."""
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Load the pretrained STonKGs Transformer here
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        # Initialize all the pre-trained as well as new weights (i.e. classifier weights) here
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Perform one forward pass for a given sequence of text_input_ids + ent_input_ids."""
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

        # Concatenate token and entity embeddings obtained from the LM and KG backbones and cast to float
        # batch x seq_len x hidden_size
        inputs_embeds = torch.cat(
            [token_embeddings, ent_embeddings.to(token_embeddings.device)],
            dim=1,
        ).type(torch.FloatTensor).to(self.device)

        # Get the hidden states from the pretrained STonKGs Transformer layers
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            return_dict=None,
        )
        # Only use the pooled output (of the [CLS] token)
        pooled_output = outputs[1]

        # Apply dropout and the linear layer
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":
    # Add a test for loading the model
    # unique_tags = 10
    # model = STonKGsForSequenceClassification.from_pretrained(
    #     pretrained_model_name_or_path=PRETRAINED_STONKGS_DUMMY_PATH,
    #     num_labels=unique_tags)
    # TODO: set up fine tuning

    # Add a test for loading and preprocessing the fine-tuning data
    test_df = preprocess_fine_tuning_data(os.path.join(ORGAN_DIR, 'organ_filtered.tsv'))
    logger.info(test_df.head())
