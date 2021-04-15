# -*- coding: utf-8 -*-

"""Prepares the pre-training data for STonKGs.

Run with:
python -m src.stonkgs.data.indra_for_pretraining
"""

import logging
import random

import pandas as pd

from stonkgs.constants import EMBEDDINGS_PATH, NLP_MODEL_TYPE, PRETRAINING_PATH, RANDOM_WALKS_PATH
from stonkgs.models.kg_baseline_model import _prepare_df
from typing import List, Optional
from transformers import BertTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _replace_mlm_tokens(
    tokens: List,
    vocab_len: int,
    mask_id: Optional[int] = 103,
    masked_tokens_percentage: Optional[float] = 0.15,
):
    """Applies masking to the given sequence with numeric indices and returns the manipulated sequence + labels."""
    # This code is taken from: https://d2l.ai/chapter_natural-language-processing-pretraining/
    # bert-dataset.html#defining-helper-functions-for-pretraining-tasks

    # Make a new copy of tokens for the input of a masked language model,
    # where the input may contain replaced '<mask>' or random tokens
    mlm_input_tokens = [token for token in tokens]

    # -100 indicates that these are NOT the labels that need to be predicted in MLM
    mlm_labels = [-100] * len(mlm_input_tokens)

    # Shuffle for getting 15% random tokens for prediction in the MLM task
    candidate_pred_positions = random.sample(
        range(len(mlm_input_tokens)),
        int(len(mlm_input_tokens) * masked_tokens_percentage),
    )

    for mlm_pred_position in candidate_pred_positions:
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = mask_id
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% of the time: replace the word with a random word
            else:
                masked_token = random.randint(0, vocab_len - 1)

        # Replace accordingly in the original sequence
        mlm_input_tokens[mlm_pred_position] = masked_token
        # Use the original token label for the list of masked labels
        mlm_labels[mlm_pred_position] = tokens[mlm_pred_position]

    return mlm_input_tokens, mlm_labels


def indra_to_pretraining_df(
    embedding_name_to_vector_path: Optional[str] = EMBEDDINGS_PATH,
    embedding_name_to_random_walk_path: Optional[str] = RANDOM_WALKS_PATH,
    pre_training_path: Optional[str] = PRETRAINING_PATH,
    nlp_model_type: Optional[str] = NLP_MODEL_TYPE,
    nsp_negative_proportion: Optional[float] = 0.5,
):
    """Preprocesses the INDRA statements from the pre-training file so that it contains all the necessary attributes."""

    # Load the KG embedding dict to convert the names to numeric indices
    kg_embed_dict = _prepare_df(embedding_name_to_vector_path)
    kg_name_to_idx = {key: i for i, key in enumerate(kg_embed_dict.keys())}

    # Load the random walks for each node
    random_walk_dict = _prepare_df(embedding_name_to_random_walk_path)
    # Convert random walk sequences to list of numeric indices
    random_walk_idx_dict = {k: [kg_name_to_idx[node] for node in v] for k, v in random_walk_dict.items()}

    # Load the pre-training dataframe
    pretraining_df = pd.read_csv(pre_training_path, sep='\t')

    # Check if all pretraining entities are covered by the embedding dict
    number_of_pre_training_nodes = len(set(pretraining_df["source"]).union(set(pretraining_df["target"])))
    if number_of_pre_training_nodes > len(kg_embed_dict):
        logger.warning('The learned KG embeddings do not cover all of the nodes in the pre-training data')
        return

    # Get the length of the text or entity embedding sequences (2 random walks = entity embedding sequence length)
    half_length = len(random_walk_idx_dict.values()[0]) * 2

    # Initialize a tokenizer used for getting the text token ids
    tokenizer = BertTokenizer.from_pretrained(nlp_model_type)

    for idx, row in pretraining_df.iterrows():
        # The features that we need to create here:
        # TODO: input_ids=None,
        # attention_mask=None, DONE
        # token_type_ids=None, DONE
        # TODO: position_ids=None,

        # TODO: masked_lm_labels=None,
        # TODO: ent_masked_lm_labels=None,
        # For MLM, the model expects a tensor of dimension(batch_size, seq_length) with each value
        # corresponding to the expected label of each individual token: the labels being the token ID
        # for the masked token, and values to be ignored for the rest (usually -100).

        # TODO: next_sentence_labels=None,

        # 1. "Token type IDs": 0 for text tokens, 1 for entity tokens
        token_type_ids = [0] * half_length + 1 * [half_length]

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

        # 3. Get the random walks sequence and the node indices
        random_walk = random_walk_idx_dict[row['source']] + random_walk_idx_dict[row['target']]

        # 4. Total attention mask (attention mask is all 1 for the entity sequence)
        attention_mask = text_attention_mask + [1] * half_length

        # TODO: DO THE MASKING HERE, ADD -1 in the entity sequences (and 103 in the text sequences)
        masked_lm_token_ids, masked_lm_labels = _replace_mlm_tokens(text_token_ids)  # TODO more parameters
        ent_masked_lm_token_ids, ent_masked_lm_labels = _replace_mlm_tokens()  # TODO more parameters

        input_ids = masked_lm_token_ids + ent_masked_lm_token_ids

    # return NotImplementedError()


if __name__ == "__main__":
    # Just simple testing
    indra_to_pretraining_df()
