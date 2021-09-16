# -*- coding: utf-8 -*-

"""ProtSTonKGs model architecture components."""

from __future__ import annotations

import logging

import torch
from transformers import (
    BertModel,
    BigBirdConfig,
    BigBirdForPreTraining,
    # BigBirdModel,
    BigBirdTokenizer,
)

from stonkgs.constants import EMBEDDINGS_PATH, PROT_NLP_MODEL_TYPE
from stonkgs.models.kg_baseline_model import prepare_df

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ProtSTonKGsForPreTraining(BigBirdForPreTraining):
    """Create the pre-training part of the STonKGs model based on both text and entity embeddings."""

    def __init__(
        self,
        config,  # Required for automated methods such as .from_pretrained in classes that inherit from this one,
        # the config is loaded from scratch later on anyways
        nlp_model_type: str = PROT_NLP_MODEL_TYPE,
        kg_embedding_dict_path: str = EMBEDDINGS_PATH,
    ):
        """Initialize the model architecture components of STonKGs."""
        # Initialize the KG dict from the file here, rather than passing it as a parameter, so that it can
        # be loaded from a checkpoint
        kg_embedding_dict = prepare_df(kg_embedding_dict_path)

        # Add the number of KG entities to the default config of a standard BERT model
        config = BigBirdConfig.from_pretrained(nlp_model_type)
        config.update({"kg_vocab_size": len(kg_embedding_dict)})
        # Initialize the underlying BertForPreTraining model that will be used to build the STonKGs Transformer layers
        super().__init__(config)

        # Override the standard MLM head: In the underlying BertForPreTraining model, change the MLM head to the custom
        # STonKGsELMPredictionHead so that it can be used on the concatenated text/entity input
        # TODO: ProtSTonKGs Prediction head
        # self.cls.predictions = STonKGsELMPredictionHead(config)

        # TODO: adapt the rest of the code to ProtSTonKGs
        # Language Model (LM) backbone initialization (pre-trained BERT to get the initial embeddings)
        # based on the specified nlp_model_type (e.g. BioBERT)
        self.lm_backbone = BertModel.from_pretrained(nlp_model_type)
        # Put the LM backbone on the GPU if possible
        if torch.cuda.is_available():
            self.lm_backbone.to("cuda")
        # Freeze the parameters of the LM backbone so that they're not updated during training
        # (We only want to train the STonKGs Transformer layers)
        for param in self.lm_backbone.parameters():
            param.requires_grad = False

        # Get the separator, mask and unknown token ids from a nlp_model_type specific tokenizer
        self.lm_sep_id = BigBirdTokenizer.from_pretrained(
            nlp_model_type
        ).sep_token_id  # usually 102
        self.lm_mask_id = BigBirdTokenizer.from_pretrained(
            nlp_model_type
        ).mask_token_id  # usually 103
        self.lm_unk_id = BigBirdTokenizer.from_pretrained(
            nlp_model_type
        ).unk_token_id  # usually 100

        # KG backbone initialization
        # Get numeric indices for the KG embedding vectors except for the sep, unk, mask ids which are reserved for the
        # LM [SEP] embedding vectors (see below)
        numeric_indices = list(range(len(kg_embedding_dict) + 3))
        # Keep the numeric indices of the special tokens free, don't put the kg embeds there
        for special_token_id in [self.lm_sep_id, self.lm_mask_id, self.lm_unk_id]:
            numeric_indices.remove(special_token_id)

        # Generate numeric indices for the KG node names (iterating .keys() is deterministic)
        self.kg_idx_to_name = {i: key for i, key in zip(numeric_indices, kg_embedding_dict.keys())}
        # Initialize KG index to embeddings based on the provided kg_embedding_dict
        self.kg_backbone = {
            i: torch.tensor(kg_embedding_dict[self.kg_idx_to_name[i]]).to(self.lm_backbone.device)
            for i in self.kg_idx_to_name.keys()
        }
        # Add the MASK, SEP and UNK (LM backbone) embedding vectors to the KG backbone so that the labels are correctly
        # identified in the loss function later on
        # [0][0][0] is required to get the shape from batch x seq_len x hidden_size to hidden_size
        for special_token_id in [self.lm_sep_id, self.lm_mask_id, self.lm_unk_id]:
            self.kg_backbone[special_token_id] = self.lm_backbone(
                torch.tensor([[special_token_id]]).to(self.lm_backbone.device),
            )[0][0][0]
