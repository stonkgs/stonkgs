# -*- coding: utf-8 -*-

"""STonKGs model on the fine-tuning classification task, assuming the model embeddings are pre-trained."""

from typing import List

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from transformers import BertConfig, BertForPreTraining, BertModel
from transformers.models.bert.modeling_bert import BertForPreTrainingOutput, BertLMPredictionHead

from stonkgs.models.kg_baseline_model import _prepare_df
from stonkgs.constants import EMBEDDINGS_PATH, NLP_MODEL_TYPE

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
        self.half_length = config.max_position_embeddings / 2

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
        text_hidden_states_to_vocab = self.text_decoder(hidden_states[:self.half_length])
        ent_hidden_states_to_kg_vocab = self.entity_decoder(hidden_states[self.half_length:])

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

        # KG backbone initialization
        # TODO: move that to a custom dataset class maybe?
        # Generate numeric indices for the KG node names (iterating .keys() is deterministic)
        self.kg_idx_to_name = {i: key for i, key in enumerate(kg_embedding_dict.keys())}
        # Initialize KG index to embeddings based on the provided kg_embedding_dict
        self.kg_backbone = {i: kg_embedding_dict[self.kg_idx_to_name[i]]
                            for i in self.kg_idx_to_name.keys()}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        masked_lm_labels=None,
        ent_masked_lm_labels=None,
        next_sentence_labels=None,
        return_dict=None,
    ):
        """Perform one forward pass for a given sequence of text_input_ids + ent_input_ids."""
        # The code is based on CoLAKE: https://github.com/txsun1997/CoLAKE/blob/master/pretrain/model.py

        # Use the LM backbone to get the pre-trained token embeddings
        # batch x half_length x hidden_size
        # The first element of the returned tuple from the LM backbone forward() pass is the sequence of hidden states
        token_embeddings = self.lm_backbone(input_ids[:, :self.cls.predictions.half_length])[0]

        # Use the KG backbone to obtain the pre-trained entity embeddings
        # batch x half_length x hidden_size
        ent_embeddings = torch.tensor(
            [self.kg_backbone(i) for i in input_ids[:, self.cls.predictions.half_length:]],
        )

        # Concatenate token and entity embeddings obtained from the LM and KG backbones
        inputs_embeds = torch.cat([token_embeddings, ent_embeddings], dim=1)  # batch x seq_len x hidden_size

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


if __name__ == "__main__":
    # Just a simple test to see if the model can be initialized without errors
    kg_embed_dict = _prepare_df(EMBEDDINGS_PATH)
    stonkgs_dummy_model = STonKGsForPreTraining(NLP_MODEL_TYPE, kg_embed_dict)
