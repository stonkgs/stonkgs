# -*- coding: utf-8 -*-

"""ProtSTonKGs model architecture components."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import (
    BertModel,
    BigBirdConfig,
    BigBirdForPreTraining,
    BigBirdTokenizer,
)
from transformers.models.big_bird.modeling_big_bird import (
    BigBirdForPreTrainingOutput,
    BigBirdLMPredictionHead,
)
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


@dataclass
class BigBirdForPreTrainingOutputWithPooling(BigBirdForPreTrainingOutput):
    """Overriding the BigBirdForPreTrainingOutput class to further include the pooled output."""

    pooler_output: Optional[torch.FloatTensor] = None


class ProtSTonKGsPELMPredictionHead(BigBirdLMPredictionHead):
    """Custom masked protein, entity and language modeling (PELM) head for proteins, entities and text tokens."""

    def __init__(
        self,
        config,
        kg_start_idx: int = 768,
        prot_start_idx: int = 1024,
    ):
        """Initialize the ELM head based on the (hyper)parameters in the provided BertConfig."""
        super().__init__(config)
        # Initialize the KG and protein part start indices
        self.kg_start_idx = kg_start_idx
        self.prot_start_idx = prot_start_idx

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
        prot_vocab_size: int = 30,
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
        :param prot_vocab_size: Vocabulary size of the protein backbone.
        """
        # Initialize the KG dict from the file here, rather than passing it as a parameter, so that it can
        # be loaded from a checkpoint
        kg_embedding_dict = prepare_df(kg_embedding_dict_path)
        # Initialize the BigBird config for the model architecture
        config = BigBirdConfig.from_pretrained(protstonkgs_model_type)
        # Use gradient checkpointing to save memory at the expense of speed
        config.update({"gradient_checkpointing": True})
        # Add the number of KG entities to the default config of a standard BigBird model
        config.update({"kg_vocab_size": len(kg_embedding_dict)})
        # Add the protein sequence vocabulary size to the default config as well
        config.update({"prot_vocab_size": prot_vocab_size})

        # Initialize the underlying LongformerForPreTraining model that will be used to build the STonKGs
        # Transformer layers
        super().__init__(config)

        # Initialize the three backbones for generating the initial embeddings for the three modalities (text, KG, prot)
        # 1. LM backbone for text (pre-trained BERT-based model to get the initial embeddings)
        # based on the specified protstonkgs_model_type (e.g. BioBERT)
        self.lm_backbone = BertModel.from_pretrained(lm_model_type)

        # 2. Prot backbone for protein sequences (e.g. ProtBERT)
        # do_lower_case is required, see example in https://huggingface.co/Rostlab/prot_bert
        self.prot_backbone = BertModel.from_pretrained(prot_model_type)
        self.prot_start_idx = prot_start_idx

        # Initialize the ProtSTonKGs tokenizer
        self.protstonkgs_tokenizer = BigBirdTokenizer.from_pretrained(protstonkgs_model_type)

        # In order to initialize the KG backbone: First get the separator, mask and unknown token ids from the
        # ProtSTonKGs model base (BigBird)
        self.sep_id = self.protstonkgs_tokenizer.sep_token_id
        self.mask_id = self.protstonkgs_tokenizer.mask_token_id
        self.unk_id = self.protstonkgs_tokenizer.unk_token_id

        # 3. KG backbone for KG entities (pretrained node2vec model)
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
        with torch.no_grad():
            for special_token_id in [self.sep_id, self.mask_id, self.unk_id]:
                self.kg_backbone[special_token_id] = self.lm_backbone(
                    torch.tensor([[special_token_id]]).to(self.device),
                )[0][0][0]

        # Override the standard MLM head: In the underlying BigBirdForPreTraining model, change the MLM head to a
        # custom ProtSTonKGsELMPredictionHead so that it can be used on the concatenated text/entity/prot sequence input
        self.cls.predictions = ProtSTonKGsPELMPredictionHead(
            config,
            kg_start_idx=kg_start_idx,
            prot_start_idx=prot_start_idx,
        )

        # Freeze the parameters of the LM and Prot backbones so that they're not updated during training
        # (We only want to train the ProtSTonKGs Transformer layers + prot to hidden linear layer)
        for backbone in [self.lm_backbone, self.prot_backbone]:
            for param in backbone.parameters():
                param.requires_grad = False

        # Add another layer that transforms the hidden size of the protein model onto the LM/KG hidden size
        self.prot_to_lm_hidden_linear = nn.Linear(
            self.prot_backbone.config.hidden_size,
            self.lm_backbone.config.hidden_size,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        masked_lm_labels=None,
        ent_masked_lm_labels=None,
        prot_masked_lm_labels=None,
        return_dict=None,
        head_mask=None,
    ):
        """Perform one forward pass for a given sequence of text_input_ids + ent_input_ids + prot_input_ids.

        Due to having more than two parts (and a RoBERTa base in the default BigBird model), the NSP objective is
        omitted in this forward function.

        :param input_ids: Concatenation of text + KG (random walk) + protein sequence embeddings
        :param attention_mask: Attention mask of the combined input sequence
        :param token_type_ids: Token type IDs of the combined input sequence
        :param masked_lm_labels: Masked LM labels for only the text part
        :param ent_masked_lm_labels: Masked entity labels for only the KG part
        :param prot_masked_lm_labels: Masked protein labels for only the protein part
        :param return_dict: Whether the output should be returned as a dict or not
        :param head_mask: Used to cancel out certain heads in the Transformer

        :return: Loss, prediction_logits in a LongformerForPreTrainingOutputWithPooling format
        """
        # No backpropagation is needed for getting the initial embeddings from the backbones
        with torch.no_grad():
            # 1. Use the LM backbone to get the pre-trained token embeddings
            # batch x number_text_tokens x hidden_size
            # The first element of the returned tuple from the LM backbone forward() pass is the sequence of hidden
            # states
            text_embeddings = torch.cat(
                [
                    self.lm_backbone(
                        input_ids[
                            :, i * (self.kg_start_idx // 3) : (i + 1) * (self.kg_start_idx // 3)
                        ]
                    )[0]
                    for i in range(3)
                ],
                dim=1,
            )

            # 2. Use the KG backbone to obtain the pre-trained entity embeddings
            # batch x number_kg_tokens x hidden_size
            ent_embeddings = torch.stack(
                [
                    # for each numeric index in the random walks sequence: get the embedding vector from the KG backbone
                    torch.stack([self.kg_backbone[i.item()] for i in j])
                    # for each example in the batch: get the random walks sequence
                    for j in input_ids[:, self.kg_start_idx : self.prot_start_idx]
                ],
            )
            # 3. Use the Prot backbone to obtain the pre-trained entity embeddings
            # batch x number_prot_tokens x prot_hidden_size (prot_hidden_size != hidden_size)
            prot_embeddings_original_dim = self.prot_backbone(input_ids[:, self.prot_start_idx :])[
                0
            ]

        # Additional layer to project prot_hidden_size onto hidden_size
        prot_embeddings = self.prot_to_lm_hidden_linear(prot_embeddings_original_dim)

        # Concatenate token, KG and prot embeddings obtained from the LM, KG and prot backbones and cast to float
        # batch x seq_len x hidden_size
        inputs_embeds = (
            torch.cat(
                [
                    text_embeddings,
                    ent_embeddings.to(text_embeddings.device),
                    prot_embeddings.to(text_embeddings.device),
                ],
                dim=1,
            )
            .type(torch.FloatTensor)
            .to(self.device)
        )

        # Get the hidden states from the basic ProtSTonKGs Transformer layers
        # batch x seq_len x hidden_size
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=None,
        )
        # batch x seq_len x hidden_size
        # sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output
        sequence_output, pooled_output = outputs[:2]

        # Generate the prediction scores (mapping to text and entity vocab sizes + NSP) for the training objectives
        # prediction_scores = Text MLM, entity "MLM" and protein "MLM" scores
        prediction_scores, _ = self.cls(sequence_output, pooled_output)
        # The custom STonKGsELMPredictionHead returns a triple of prediction scores for tokens, entities,
        # and protein sequences, respectively
        (
            token_prediction_scores,
            entity_predictions_scores,
            prot_predictions_scores,
        ) = prediction_scores

        # Calculate the loss
        total_loss = None
        if (
            masked_lm_labels is not None
            and ent_masked_lm_labels is not None
            and prot_masked_lm_labels is not None
        ):
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
            # 3. Protein-based masked "language" (entity) modeling
            prot_masked_lm_loss = loss_fct(
                prot_predictions_scores.view(-1, self.config.prot_vocab_size),
                prot_masked_lm_labels.view(-1),
            )
            # Total loss = the sum of the individual training objective losses
            total_loss = masked_lm_loss + ent_masked_lm_loss + prot_masked_lm_loss

        if not return_dict:
            output = prediction_scores + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BigBirdForPreTrainingOutputWithPooling(
            loss=total_loss,
            prediction_logits=prediction_scores,
            hidden_states=sequence_output,
            attentions=outputs.attentions,
            pooler_output=pooled_output,
        )
