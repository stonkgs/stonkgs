# -*- coding: utf-8 -*-

"""Prepares the protein-specific pre-training data subset for ProtSTonKGs.

Run with:
python -m src.stonkgs.data.prot_indra_for_pretraining
"""

import logging
import os

import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertTokenizerFast
from tqdm import tqdm

from stonkgs.constants import (
    EMBEDDINGS_PATH,
    NLP_MODEL_TYPE,
    PRETRAINING_DIR,
    PRETRAINING_PATH,
    PROT_SEQ_MODEL_TYPE,
    RANDOM_WALKS_PATH,
    VOCAB_FILE,
)
from stonkgs.data.indra_for_pretraining import replace_mlm_tokens
from stonkgs.models.kg_baseline_model import prepare_df

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# TODO adapt function
def prot_indra_to_pretraining_df(
    embedding_name_to_vector_path: str = EMBEDDINGS_PATH,
    embedding_name_to_random_walk_path: str = RANDOM_WALKS_PATH,
    pre_training_path: str = PRETRAINING_PATH,
    kg_start_idx: int = 768,
    lm_model_type: str = NLP_MODEL_TYPE,
    prot_model_type: str = PROT_SEQ_MODEL_TYPE,
    sep_id: int = 102,
):
    """Preprocesses the INDRA statements from the protein-specific pre-training file for the ProtSTonKGs model."""
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

    # Load the pre-training dataframe
    pretraining_df = pd.read_csv(pre_training_path, sep="\t")

    # Initialize a FAST LM tokenizer if it's the default one (BioBERT)
    if lm_model_type == NLP_MODEL_TYPE:
        # Initialize the fast tokenizer for getting the text token ids
        lm_tokenizer = BertTokenizerFast(vocab_file=VOCAB_FILE)
    else:
        # Initialize a slow tokenizer used for getting the text token ids
        lm_tokenizer = BertTokenizer.from_pretrained(lm_model_type)

    # Initialize a Protein tokenizer as well (default = ProtSTonKGs)
    # do_lower_case is required, see example in https://huggingface.co/Rostlab/prot_bert
    prot_tokenizer = BertTokenizer.from_pretrained(prot_model_type, do_lower_case=False)

    # Initialize the preprocessed data
    pre_training_preprocessed = []

    # Log progress with a progress bar
    for idx, row in tqdm(
        pretraining_df.iterrows(),
        total=pretraining_df.shape[0],
        desc="Generating samples",
    ):
        # 2. Tokenization for getting the input ids and attention masks for the text
        # Use encode_plus to also get the attention mask ("padding" mask)
        # TODO: combine evidence, "source_description" and "target_description"
        encoded_text = lm_tokenizer.encode_plus(
            row["evidence"],
            padding="max_length",
            truncation=True,
            max_length=kg_start_idx//3,  # TODO: replace with function parameter
        )
        text_token_ids = encoded_text["input_ids"]
        text_attention_mask = encoded_text["attention_mask"]

        # 3. Get the random walks sequence and the node indices, add the SEP (usually with id=102) in between
        random_walks = (
            random_walk_idx_dict[row["source"]]
            + [sep_id]
            + random_walk_idx_dict[row["target"]]
            + [sep_id]
        )

        # TODO 4. Get the protein sequence and combine source and target with [SEP]
        prot_sequence = lm_tokenizer.encode_plus(
            row["source_prot"],
            padding="max_length",
            truncation=True,
            max_length=kg_start_idx//3,  # TODO: replace with function parameter
        )
        prot_sequence_ids = prot_sequence["input_ids"]
        prot_attention_mask = prot_sequence["attention_mask"]

        # Apply the masking strategy to the text tokens and get the text MLM labels
        masked_lm_token_ids, masked_lm_labels = replace_mlm_tokens(
            tokens=text_token_ids,
            vocab_len=len(lm_tokenizer.vocab),
        )
        # Apply the masking strategy to the entity tokens and get the entity (E)LM labels
        # Use the same mask_id as in the NLP model (handled appropriately by STonKGs later on)
        ent_masked_lm_token_ids, ent_masked_lm_labels = replace_mlm_tokens(
            tokens=random_walks,
            vocab_len=len(kg_embed_dict),
        )
        # Also apply the masking strategy to the protein tokens and get the protein (P)LM labels
        # Again, use the same mask_id as in the NLP model (handled appropriately by STonKGs later on)
        prot_masked_lm_token_ids, prot_masked_lm_labels = replace_mlm_tokens(
            tokens=prot_sequence_ids,
            vocab_len=len(prot_tokenizer.vocab)
        )

        # 4. Total attention mask (attention mask is all 1 for the entity sequence)
        attention_mask = text_attention_mask + [1] * len(random_walks) + prot_attention_mask

        # 5. Define the final concatenated input sequence
        input_ids = masked_lm_token_ids + ent_masked_lm_token_ids + prot_masked_lm_token_ids

        # Add all the features to the preprocessed data
        pre_training_preprocessed.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "masked_lm_labels": masked_lm_labels,
                "ent_masked_lm_labels": ent_masked_lm_labels,
                "prot_masked_lm_labels": prot_masked_lm_labels,
            }
        )

    # Put the preprocessed data into a dataframe
    pre_training_preprocessed_df = pd.DataFrame(pre_training_preprocessed)
    logger.info("Finished generating the training examples")

    # Save the final dataframe
    pre_training_preprocessed_df.to_csv(
        os.path.join(PRETRAINING_DIR, "pretraining_preprocessed_prot.tsv"),
        sep="\t",
        index=False,
    )
    # Pickle it, too (easier for reading in the lists in the pandas dataframe)
    pre_training_preprocessed_df.to_pickle(
        os.path.join(PRETRAINING_DIR, "pretraining_preprocessed_prot.pkl"),
    )

    logger.info(f"Saved the data under {PRETRAINING_DIR}")

    return pre_training_preprocessed_df
