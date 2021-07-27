# -*- coding: utf-8 -*-

"""Embeddings from the pre-trained STonKGs model."""

import logging
from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertTokenizerFast

from stonkgs.constants import (
    EMBEDDINGS_PATH,
    NLP_MODEL_TYPE,
    RANDOM_WALKS_PATH,
    VOCAB_FILE,
)
from stonkgs.data.indra_for_pretraining import prepare_df, replace_mlm_tokens
from stonkgs.models.stonkgs_pretraining import STonKGsForPreTraining

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_file_for_embeddings(
    data_path: str,
    embedding_name_to_vector_path: str = EMBEDDINGS_PATH,
    embedding_name_to_random_walk_path: str = RANDOM_WALKS_PATH,
    nlp_model_type: str = NLP_MODEL_TYPE,
    sep_id: int = 102,
    unk_id: int = 100,
) -> pd.DataFrame:
    """Preprocesses a given file specified by data_path so that it's ready for embedding extraction by STonKGs."""
    # Load the KG embedding dict to convert the names to numeric indices
    kg_embed_dict = prepare_df(embedding_name_to_vector_path)
    kg_name_to_idx = {key: i for i, key in enumerate(kg_embed_dict.keys())}

    # Load the random walks for each node
    random_walk_dict = prepare_df(embedding_name_to_random_walk_path)

    # Log the number of entities
    logger.info(f"There are {len(kg_embed_dict)} many entities in the pre-trained KG")

    # Convert random walk sequences to list of numeric indices
    random_walk_idx_dict = {
        k: [kg_name_to_idx[node] for node in v] for k, v in random_walk_dict.items()
    }

    # Load the dataframe for which embeddings are going to be generated
    pretraining_df = pd.read_csv(data_path, sep="\t")

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
    for _, row in tqdm(
        pretraining_df.iterrows(),
        total=pretraining_df.shape[0],
        desc="Preprocessing the dataset",
    ):
        # 1. "Token type IDs": 0 for text tokens, 1 for entity tokens
        token_type_ids = [0] * half_length + [1] * half_length

        # 2. Tokenization for getting the input ids and attention masks for the text
        # Use encode_plus to also get the attention mask ("padding" mask)
        encoded_text = tokenizer.encode_plus(
            row["evidence"],
            padding="max_length",
            truncation=True,
            max_length=half_length,
        )
        text_token_ids = encoded_text["input_ids"]
        text_attention_mask = encoded_text["attention_mask"]
        random_walk_length = len(next(iter(random_walk_idx_dict.values())))

        # 3. Get the random walks sequence and the node indices, add the SEP (usually with id=102) in between
        # Use a sequence of UNK tokens if the node is not contained in the dictionary of the nodes from pre-training
        random_w_source = (
            random_walk_idx_dict[row["source"]]
            if row["source"] in random_walk_idx_dict.keys()
            else [unk_id] * random_walk_length
        )
        random_w_target = (
            random_walk_idx_dict[row["target"]]
            if row["target"] in random_walk_idx_dict.keys()
            else [unk_id] * random_walk_length
        )
        random_walks = random_w_source + [sep_id] + random_w_target + [sep_id]

        # 4. Total attention mask (attention mask is all 1 for the entity sequence)
        attention_mask = text_attention_mask + [1] * half_length

        # Apply the masking strategy to the text tokens and get the text MLM labels
        masked_lm_token_ids, masked_lm_labels = replace_mlm_tokens(
            tokens=text_token_ids,
            vocab_len=len(tokenizer.vocab),
        )
        # Apply the masking strategy to the entity tokens and get the entity (E)LM labels
        # Use the same mask_id as in the NLP model (handled appropriately by STonKGs later on)
        ent_masked_lm_token_ids, ent_masked_lm_labels = replace_mlm_tokens(
            tokens=random_walks,
            vocab_len=len(kg_embed_dict),
        )

        input_ids = masked_lm_token_ids + ent_masked_lm_token_ids

        # Add all the features to the preprocessed data
        pre_training_preprocessed.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "masked_lm_labels": masked_lm_labels,
                "ent_masked_lm_labels": ent_masked_lm_labels,
                "next_sentence_labels": 0,  # 0 indicates the random walks belong to the text evidence
            }
        )

    # Put the preprocessed data into a dataframe
    pre_training_preprocessed_df = pd.DataFrame(pre_training_preprocessed)

    return pre_training_preprocessed_df


def get_stonkgs_embeddings(
    preprocessed_df: pd.DataFrame,
    pretrained_stonkgs_model_name: Optional[str],
    list_of_indices: Optional[List],
) -> pd.DataFrame:
    """Extract embeddings for a preprocessed_df based on a pretrained_stonkgs_model_name."""
    all_embed_sequences = pd.DataFrame(columns=["embedding"])

    # Initialize the model
    if pretrained_stonkgs_model_name:
        stonkgs_model = STonKGsForPreTraining.from_pretrained(pretrained_stonkgs_model_name)
    else:
        stonkgs_model = STonKGsForPreTraining.from_default_pretrained()

    # Use all indices for extraction if they are unspecified
    if list_of_indices is None:
        list_of_indices = list(range(len(preprocessed_df)))

    for idx in tqdm(list_of_indices):
        data_entry = {
            key: torch.tensor([value]) for key, value in dict(preprocessed_df.iloc[idx]).items()
        }
        stonkgs_hidden_states = stonkgs_model(**data_entry, return_dict=True).pooler_output[0]
        all_embed_sequences = all_embed_sequences.append(
            {"embedding": stonkgs_hidden_states.tolist()},
            ignore_index=True,
        )

    return all_embed_sequences
