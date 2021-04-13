# -*- coding: utf-8 -*-

"""STonKGs model on the fine-tuning classification task, assuming the model embeddings are pre-trained."""

from typing import List

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import BertForPreTraining


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


class STonKGsForPreTraining(BertForPreTraining):
    """Create the pre-training part of the STonKGs model based on both text and entity embeddings."""

    def __init__(self, nlp_model_type, kg_embedding_dict):
        """Initialize the model architecture components of STonKGs."""
        # Initialize the BertForMaskedLM model based on the specified model type (e.g., BioBERT)

        # TODO: index to name for kg_embedding_dict (sort keys alphabetically first maybe?? which is probably not good
        #  for such a large KG)
        # kg_embeddings = ... something with kg_embedding_dict

        super().__init__(nlp_model_type)
        # TODO: freeze the underlying BERT model

        # TODO: initialize text embeddings?
        # TODO: initialize KG embeddings? -> use nn.Embeddings.from_pretrained? Or just a normal dictionary/index?

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        masked_lm_labels=None,
        ent_masked_lm_labels=None,
        rel_masked_lm_labels=None,
        n_word_nodes=None,
        n_entity_nodes=None,
        ent_index=None,
    ):
        """Perform one forward pass for a given sequence of text_input_ids + ent_input_ids."""
        # The code is based on CoLAKE: https://github.com/txsun1997/CoLAKE/blob/master/pretrain/model.py
        # Number of tokens (the rest are entity embeddings)
        n_word_nodes = n_word_nodes[0]
        word_embeddings = self.bert.embeddings.word_embeddings(
            input_ids[:, :n_word_nodes])  # batch x n_word_nodes x hidden_size

        ent_embeddings = self.kg_embeddings(
            input_ids[:, n_word_nodes:])

        inputs_embeds = torch.cat([word_embeddings, ent_embeddings],
                                  dim=1)  # batch x seq_len x hidden_size

        # TODO: define the encoder and stuff in init
        outputs = self.encoder(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]  # batch x seq_len x hidden_size

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

        # TODO: integrate NSP loss here (Is a given text evidence matching with an entity sequence?)
        nsp_loss = 0

        # TODO: initialize a LM head
        word_logits = self.lm_head(sequence_output[:, :n_word_nodes, :])
        word_predict = torch.argmax(word_logits, dim=-1)
        masked_lm_loss = loss_fct(word_logits.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        ent_cls_weight = self.ent_embeddings(ent_index[0].view(1, -1)).squeeze()
        ent_logits = self.ent_lm_head(sequence_output[:, n_word_nodes:, :],
                                      ent_cls_weight)
        ent_predict = torch.argmax(ent_logits, dim=-1)
        ent_masked_lm_loss = loss_fct(ent_logits.view(-1, ent_logits.size(-1)), ent_masked_lm_labels.view(-1))

        loss = masked_lm_loss + ent_masked_lm_loss + nsp_loss

        return {
            'loss': loss,
            'word_pred': word_predict,
            'entity_pred': ent_predict,
        }
