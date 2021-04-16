# -*- coding: utf-8 -*-

"""Prepares the pre-training data for STonKGs.

Run with:
python -m src.stonkgs.data.indra_for_pretraining
"""

import logging
import os
import random
from typing import List, Optional

import numpy as np
import pandas as pd
from transformers import BertTokenizer

from stonkgs.constants import EMBEDDINGS_PATH, NLP_MODEL_TYPE, PRETRAINING_DIR, PRETRAINING_PATH, RANDOM_WALKS_PATH
from stonkgs.models.kg_baseline_model import _prepare_df

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _replace_mlm_tokens(
    tokens: List[int],
    vocab_len: int,
    mask_id: int = 103,
    masked_tokens_percentage: float = 0.15,
    unmasked_label_id: int = -100,
):
    """Applies masking to the given sequence with numeric indices and returns the manipulated sequence + labels."""
    # This code is taken from: https://d2l.ai/chapter_natural-language-processing-pretraining/
    # bert-dataset.html#defining-helper-functions-for-pretraining-tasks

    # Make a new copy of tokens for the input of a masked language model,
    # where the input may contain replaced '<mask>' or random tokens
    mlm_input_tokens = [token for token in tokens]

    # The unmasked_label_id indicates that these are NOT the labels that need to be predicted in MLM
    # For MLM, the model expects a tensor of dimension(batch_size, seq_length) with each value
    # corresponding to the expected label of each individual token: the labels being the token ID
    # for the masked token, and values to be ignored for the rest (usually -100).
    mlm_labels = [unmasked_label_id] * len(mlm_input_tokens)

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


def _add_negative_nsp_samples(
    processed_df: pd.DataFrame,
    nsp_negative_proportion: float = 0.5,
):
    """Generates non-matching text-entity pairs (negative NSP samples)."""
    negative_samples = []

    # Get the length of half a sequence
    half_length = len(processed_df.iloc[0]['input_ids']) // 2

    # First get the indices that serve as the basis for the negative examples
    negative_sample_idx = random.sample(
        range(len(processed_df)),
        int(len(processed_df) * nsp_negative_proportion),
    )

    # For each negative sample, get a random index that is not the current one for creating negative samples
    negative_sample_idx_partner = [
        random.choice([j for j in range(len(processed_df)) if j != i])
        for i in negative_sample_idx
    ]

    for idx, (i, j) in enumerate(zip(negative_sample_idx, negative_sample_idx_partner)):
        # Log the progress
        if idx % 1000 == 0:
            logger.info(f'Processing negative examples for row number {idx} of '
                        f'{int(len(processed_df) * nsp_negative_proportion)}')

        # Get the features from i
        text_features = processed_df.iloc[i]
        # Get the features from j
        entity_features = processed_df.iloc[j]
        # 1. Replace the second half of the input sequence
        # 2. Replace the entity mask labels
        # 3. Replace the NSP label
        new_entry = {
            'input_ids': text_features['input_ids'][:half_length] + entity_features['input_ids'][half_length:],
            'attention_mask': text_features['attention_mask'],
            'token_type_ids': text_features['token_type_ids'],
            'masked_lm_labels': text_features['masked_lm_labels'],
            'ent_masked_lm_labels': entity_features['ent_masked_lm_labels'],
            'next_sentence_labels': 1,  # 1 indicates it's a corrupted training sample
        }
        negative_samples.append(new_entry)

    negative_samples_df = pd.DataFrame(negative_samples)

    return negative_samples_df


def indra_to_pretraining_df(
    embedding_name_to_vector_path: str = EMBEDDINGS_PATH,
    embedding_name_to_random_walk_path: str = RANDOM_WALKS_PATH,
    pre_training_path: str = PRETRAINING_PATH,
    nlp_model_type: str = NLP_MODEL_TYPE,
    nsp_negative_proportion: float = 0.5,
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
    half_length = len(next(iter(random_walk_idx_dict.values()))) * 2

    # Initialize a tokenizer used for getting the text token ids
    tokenizer = BertTokenizer.from_pretrained(nlp_model_type)

    # Initialize the preprocessed data
    pre_training_preprocessed = []

    for idx, row in pretraining_df.iterrows():
        # Log the progress
        if idx % 1000 == 0:
            logger.info(f'Processing positive examples for row number {idx} of {len(pretraining_df)}')

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

        # 3. Get the random walks sequence and the node indices
        random_walks = random_walk_idx_dict[row['source']] + random_walk_idx_dict[row['target']]

        # 4. Total attention mask (attention mask is all 1 for the entity sequence)
        attention_mask = text_attention_mask + [1] * half_length

        # Apply the masking strategy to the text tokens and get the text MLM labels
        masked_lm_token_ids, masked_lm_labels = _replace_mlm_tokens(
            tokens=text_token_ids,
            vocab_len=len(tokenizer.vocab),
        )
        # Apply the masking strategy to the entity tokens and get the entity (E)LM labels
        # The mask_id is -1 for the entity vocabulary (handled by the STonKGs forward pass later on)
        ent_masked_lm_token_ids, ent_masked_lm_labels = _replace_mlm_tokens(
            tokens=random_walks,
            vocab_len=len(kg_embed_dict),
            mask_id=-1,
        )

        input_ids = masked_lm_token_ids + ent_masked_lm_token_ids

        # Add all the features to the preprocessed data
        pre_training_preprocessed.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'masked_lm_labels': masked_lm_labels,
            'ent_masked_lm_labels': ent_masked_lm_labels,
            'next_sentence_labels': 0  # 0 indicates the random walks belong to the text evidence
        })

    # Put the preprocessed data into a dataframe
    pre_training_preprocessed_df = pd.DataFrame(pre_training_preprocessed)

    # Generate the negative NSP training samples
    pre_training_negative_samples = _add_negative_nsp_samples(
        pre_training_preprocessed_df,
        nsp_negative_proportion=nsp_negative_proportion,
    )

    # And append them to the original data
    pre_training_preprocessed_df = pre_training_preprocessed_df.append(
        pre_training_negative_samples
    ).reset_index()

    # Shuffle the dataframe just to be sure
    pre_training_preprocessed_df_shuffled = pre_training_preprocessed_df.iloc[
        np.random.permutation(pre_training_preprocessed_df.index)
    ].reset_index(drop=True)

    # Save the final dataframe
    pre_training_preprocessed_df_shuffled.to_csv(
        os.path.join(PRETRAINING_DIR, 'pretraining_preprocessed.tsv'),
        sep='\t',
        index=False,
    )
    # Pickle it, too (easier for reading in the lists in the pandas dataframe)
    pre_training_preprocessed_df_shuffled.to_pickle(
        os.path.join(PRETRAINING_DIR, 'pretraining_preprocessed.pkl'),
    )

    return pre_training_preprocessed_df


if __name__ == "__main__":
    # Just simple testing
    indra_to_pretraining_df()
