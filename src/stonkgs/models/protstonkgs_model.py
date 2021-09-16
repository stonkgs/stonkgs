# -*- coding: utf-8 -*-

"""ProtSTonKGs model architecture components."""

from __future__ import annotations

import logging

import torch
from torch import nn
from transformers import (
    BertModel,
    BertTokenizer,
    BigBirdConfig,
    BigBirdForPreTraining,
    # BigBirdModel,
    BigBirdTokenizer,
)
from transformers.models.big_bird.modeling_big_bird import BigBirdLMPredictionHead

from stonkgs.constants import (
    EMBEDDINGS_PATH,
    NLP_MODEL_TYPE,
    PROTSTONKGS_MODEL_TYPE,
    PROT_SEQ_MODEL_TYPE,
)
from stonkgs.models.kg_baseline_model import prepare_df

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ProtSTonKGsPELMPredictionHead(BigBirdLMPredictionHead):
    """Custom masked protein, entity and language modeling (PELM) head for proteins, entities and text tokens."""

    def __init__(self, config):
        """Initialize the ELM head based on the (hyper)parameters in the provided BertConfig."""
        super().__init__(config)

        # There are three different "decoders":
        # 1. The text part of the sequence that is projected onto the dimension of the text vocabulary index
        # 2. The KG part of the sequence that is projected onto the dimension of the kg vocabulary index
        # 3. The protein sequence part that is projected onto the dimension of the protein vocabulary index
        # 1. Text decoder
        self.text_decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 2. KG/Entity decoder
        self.entity_decoder = nn.Linear(config.hidden_size, config.kg_vocab_size, bias=False)
        # 3. Protein decoder
        self.prot_decoder = nn.Linear(config.hidden_size, config.prot_vocab_size, bias=False)

        # Set the biases differently for the decoder layers
        self.text_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.entity_bias = nn.Parameter(torch.zeros(config.kg_vocab_size))
        self.prot_bias = nn.Parameter(torch.zeros(config.prot_vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.text_bias = self.text_bias
        self.decoder.entity_bias = self.entity_bias
        self.decoder.prot_bias = self.prot_bias

    def forward(self, hidden_states):
        """Map hidden states to values for the text (1st part), kg (2nd part) and protein vocabs (3rd part)."""
        # Common transformations (dense layer, layer norm + activation function) performed on text, KG and protein data
        # transform is initialized in the parent BigBirdLMPredictionHead class
        hidden_states = self.transform(hidden_states)

        # The first part is processed with the text decoder, the second with the entity decoder, and the third with the
        # protein decoder to map to the text, kg, and protein vocab size, respectively
        text_hidden_states_to_vocab = self.text_decoder(hidden_states[:, : self.kg_start_idx])
        ent_hidden_states_to_kg_vocab = self.entity_decoder(
            hidden_states[:, self.kg_start_idx : self.prot_start_idx]
        )
        prot_hidden_states_to_prot_vocab = self.prot_decoder(
            hidden_states[:, self.prot_start_idx :]
        )

        return (
            text_hidden_states_to_vocab,
            ent_hidden_states_to_kg_vocab,
            prot_hidden_states_to_prot_vocab,
        )


class ProtSTonKGsForPreTraining(BigBirdForPreTraining):
    """Create the pre-training part of the ProtSTonKGs model based, text and KG and protein sequence embeddings."""

    def __init__(
        self,
        config,  # the config is loaded from scratch later on anyways
        protstonkgs_model_type: str = PROTSTONKGS_MODEL_TYPE,
        lm_model_type: str = NLP_MODEL_TYPE,
        prot_start_idx: int = 1024,
        prot_model_type: str = PROT_SEQ_MODEL_TYPE,
        kg_start_idx: int = 768,
        kg_embedding_dict_path: str = EMBEDDINGS_PATH,
    ):
        """Initialize the model architecture components of ProtSTonKGs.

        The Transformer operates on a concatenation of [text, KG, protein]-based input sequences.

        :param config: Required for automated methods such as .from_pretrained in classes that inherit from this one
        :param protstonkgs_model_type: The type of Transformer used to construct ProtSTonKGs.
        :param lm_model_type: The type of (hf) model used to generate the initial text embeddings.
        :param kg_start_idx: The index at which the KG random walks start (and the text ends).
        :param kg_embedding_dict_path: The path specification for the node2vec embeddings used for the KG data.
        :param prot_start_idx: The index at which the protein sequences start (and the KG part ends).
        :param prot_model_type: The type of (hf) model used to generate the initial protein sequence embeddings.
        """
        # Initialize the three backbones for generating the initial embeddings for the three modalities (text, KG, prot)
        # 1. LM backbone for text (pre-trained BERT-based model to get the initial embeddings)
        # based on the specified protstonkgs_model_type (e.g. BioBERT)
        self.lm_tokenizer = BertTokenizer.from_pretrained(lm_model_type)
        self.lm_backbone = BertModel.from_pretrained(lm_model_type)

        # 2. Prot backbone for protein sequences (e.g. ProtBERT)
        self.prot_tokenizer = BertTokenizer.from_pretrained(prot_model_type)
        self.prot_backbone = BertModel.from_pretrained(prot_model_type)
        self.prot_start_idx = prot_start_idx

        # Initialize the ProtSTonKGs tokenizer
        self.protstonkgs_tokenizer = BigBirdTokenizer.from_pretrained(protstonkgs_model_type)

        # In order to initialize the KG backbone: First get the separator, mask and unknown token ids from all relevant
        # models (LM + Prot backbones + ProtSTonKGs model) and make sure they are the same (for compatibility reasons
        # in the forward function)
        if not all(
            [
                self.protstonkgs_tokenizer.sep_token_id
                == self.lm_tokenizer.sep_token_id
                == self.prot_tokenizer.sep_token_id,
                self.protstonkgs_tokenizer.mask_token_id
                == self.lm_tokenizer.mask_token_id
                == self.prot_tokenizer.mask_token_id,
                self.protstonkgs_tokenizer.unk_token_id
                == self.lm_tokenizer.unk_token_id
                == self.prot_tokenizer.unk_token_id,
            ]
        ):
            logger.warning(
                "Special tokens are inconsistent across the backbones and/or ProtSTonKGs. The model might not be "
                "trained properly."
            )

        # Save respective attributes
        self.sep_id = self.protstonkgs_tokenizer.sep_token_id  # usually 102
        self.mask_id = self.protstonkgs_tokenizer.mask_token_id  # usually 103
        self.unk_id = self.protstonkgs_tokenizer.unk_token_id  # usually 100

        # 3. KG backbone for KG entities (pretrained node2vec model)
        # Initialize the KG dict from the file here, rather than passing it as a parameter, so that it can
        # be loaded from a checkpoint
        kg_embedding_dict = prepare_df(kg_embedding_dict_path)
        # Get numeric indices for the KG embedding vectors except for the sep, unk, mask ids which are reserved for the
        # LM [SEP] embedding vectors (see below)
        numeric_indices = list(range(len(kg_embedding_dict) + 3))
        # Keep the numeric indices of the special tokens free, don't put the kg embeds there
        for special_token_id in [self.sep_id, self.mask_id, self.unk_id]:
            numeric_indices.remove(special_token_id)
        # Generate numeric indices for the KG node names (iterating .keys() is deterministic)
        self.kg_idx_to_name = {i: key for i, key in zip(numeric_indices, kg_embedding_dict.keys())}
        # Initialize KG index to embeddings based on the provided kg_embedding_dict
        self.kg_backbone = {
            i: torch.tensor(kg_embedding_dict[self.kg_idx_to_name[i]]).to(self.lm_backbone.device)
            for i in self.kg_idx_to_name.keys()
        }
        self.kg_start_idx = kg_start_idx
        # Add the MASK, SEP and UNK (LM backbone) embedding vectors to the KG backbone so that the labels are correctly
        # identified in the loss function later on
        # [0][0][0] is required to get the shape from batch x seq_len x hidden_size to hidden_size
        for special_token_id in [self.sep_id, self.mask_id, self.unk_id]:
            self.kg_backbone[special_token_id] = self.lm_backbone(
                torch.tensor([[special_token_id]]).to(self.lm_backbone.device),
            )[0][0][0]

        # Initialize the BigBird config for the model architecture
        config = BigBirdConfig.from_pretrained(protstonkgs_model_type)
        # Add the number of KG entities to the default config of a standard BigBird model
        config.update({"kg_vocab_size": len(kg_embedding_dict)})
        # Add the protein sequence vocabulary size to the default config as well
        config.update({"prot_vocab_size": self.prot_backbone.config.vocab_size})

        # Initialize the underlying BigBirdForPreTraining model that will be used to build the STonKGs
        # Transformer layers
        super().__init__(config)

        # Override the standard MLM head: In the underlying BigBirdForPreTraining model, change the MLM head to a custom
        # ProtSTonKGsELMPredictionHead so that it can be used on the concatenated text/entity/prot sequence input
        self.cls.predictions = ProtSTonKGsPELMPredictionHead(config)

        # Freeze the parameters of the LM and Prot backbones so that they're not updated during training
        # (We only want to train the ProtSTonKGs Transformer layers)
        for backbone in [self.lm_backbone, self.prot_backbone]:
            for param in backbone.parameters():
                param.requires_grad = False
