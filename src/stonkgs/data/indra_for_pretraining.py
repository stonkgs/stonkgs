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
from transformers import BertTokenizer, BertTokenizerFast
from tqdm import tqdm

from stonkgs.constants import (
    EMBEDDINGS_PATH,
    NLP_MODEL_TYPE,
    PRETRAINING_DIR,
    PRETRAINING_PATH,
    RANDOM_WALKS_PATH,
    VOCAB_FILE,
)
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
    nsp_negative_proportion: float = 0.25,
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

    for i, j in tqdm(
        zip(negative_sample_idx, negative_sample_idx_partner),
        total=int(nsp_negative_proportion * len(processed_df)),
        desc='Generating negative samples',
    ):
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
    sep_id: int = 102,
):
    """Preprocesses the INDRA statements from the pre-training file so that it contains all the necessary attributes."""
    # Load the KG embedding dict to convert the names to numeric indices
    kg_embed_dict = _prepare_df(embedding_name_to_vector_path)
    kg_name_to_idx = {key: i for i, key in enumerate(kg_embed_dict.keys())}

    # Load the random walks for each node
    random_walk_dict = _prepare_df(embedding_name_to_random_walk_path)
    # Assert that embeddings and random walks are generated based on the same dataset
    assert len(kg_embed_dict) == len(random_walk_dict), 'Embeddings and random walks must cover the same entities'

    # Log the number of entities
    logger.info(f'There are {len(kg_embed_dict)} many entities in the pre-trained KG')

    # Convert random walk sequences to list of numeric indices
    random_walk_idx_dict = {k: [kg_name_to_idx[node] for node in v] for k, v in random_walk_dict.items()}

    # Load the pre-training dataframe
    pretraining_df = pd.read_csv(pre_training_path, sep='\t')

    # Check if all pretraining entities are covered by the embedding dict
    number_of_pre_training_nodes = len(set(pretraining_df["source"]).union(set(pretraining_df["target"])))
    if number_of_pre_training_nodes > len(kg_embed_dict):
        logger.warning('The learned KG embeddings do not cover all of the nodes in the pre-training data')
        return

    # Get the length of the text or entity embedding sequences (2 random walks + 2 = entity embedding sequence length)
    half_length = len(next(iter(random_walk_idx_dict.values()))) * 2 + 2

    # Initialize a FAST tokenizer if it's the default one (BioBERT)
    if nlp_model_type == NLP_MODEL_TYPE:
        # Initialize the fast tokenizer for getting the text token ids
        tokenizer = BertTokenizerFast(vocab_file=VOCAB_FILE)
    else:
        # Initialize a slow tokenizer used for getting the text token ids
        tokenizer = BertTokenizer.from_pretrained(nlp_model_type)

    # Initialize the preprocessed data
    pre_training_preprocessed = []

    # Log progress with a progress bar
    for idx, row in tqdm(
        pretraining_df.iterrows(),
        total=pretraining_df.shape[0],
        desc='Generating positive samples',
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
        random_walks = random_walk_idx_dict[row['source']] + [sep_id] + random_walk_idx_dict[row['target']] + [sep_id]

        # 4. Total attention mask (attention mask is all 1 for the entity sequence)
        attention_mask = text_attention_mask + [1] * half_length

        # Apply the masking strategy to the text tokens and get the text MLM labels
        masked_lm_token_ids, masked_lm_labels = _replace_mlm_tokens(
            tokens=text_token_ids,
            vocab_len=len(tokenizer.vocab),
        )
        # Apply the masking strategy to the entity tokens and get the entity (E)LM labels
        # Use the same mask_id as in the NLP model (handled appropriately by STonKGs later on)
        ent_masked_lm_token_ids, ent_masked_lm_labels = _replace_mlm_tokens(
            tokens=random_walks,
            vocab_len=len(kg_embed_dict),
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

    logger.info('Finished generating positive training examples')

    # Generate the negative NSP training samples
    pre_training_negative_samples = _add_negative_nsp_samples(
        pre_training_preprocessed_df,
        nsp_negative_proportion=nsp_negative_proportion,
    )

    # And append them to the original data
    pre_training_preprocessed_df = pre_training_preprocessed_df.append(
        pre_training_negative_samples
    ).reset_index()

    logger.info('Finished generating negative training examples')

    # Shuffle the dataframe just to be sure
    pre_training_preprocessed_df_shuffled = pre_training_preprocessed_df.iloc[
        np.random.permutation(pre_training_preprocessed_df.index)
    ].reset_index(drop=True)

    logger.info('Finished shuffling the data')

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

    logger.info(f'Saved the data under {PRETRAINING_DIR}')

    return pre_training_preprocessed_df


if __name__ == "__main__":
    # Just simple testing
    indra_to_pretraining_df()
