# -*- coding: utf-8 -*-

"""Prepares the pre-training data based on TransE embeddings for STonKGs.

Run with:
python -m src.stonkgs.data.transe_indra_for_pretraining
"""

import logging
import os
from more_itertools import sliced

import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertTokenizerFast
from tqdm import tqdm

from stonkgs.constants import (
    TRANSE_EMBEDDINGS_PATH,
    NLP_MODEL_TYPE,
    PRETRAINING_DIR,
    PRETRAINING_PATH,
    VOCAB_FILE,
)
from stonkgs.data.indra_for_pretraining import _add_negative_nsp_samples, replace_mlm_tokens
from stonkgs.models.kg_baseline_model import prepare_df

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def indra_to_transe_pretraining_df(
    embedding_name_to_vector_path: str = TRANSE_EMBEDDINGS_PATH,
    pre_training_path: str = PRETRAINING_PATH,
    nlp_model_type: str = NLP_MODEL_TYPE,
    nsp_negative_proportion: float = 0.25,
    text_part_length: int = 256,
    sep_id: int = 102,
    chunk_size: int = 50000,
):
    """Preprocesses the INDRA statements from the pre-training file so that it contains all the necessary attributes."""
    # Load the KG embedding dict to convert the names to numeric indices
    kg_embed_dict = prepare_df(embedding_name_to_vector_path)
    kg_name_to_idx = {key: i for i, key in enumerate(kg_embed_dict.keys())}

    # Log the number of entities
    logger.info(f"There are {len(kg_embed_dict)} many entities in the pre-trained KG")

    # Load the pre-training dataframe
    pretraining_df = pd.read_csv(pre_training_path, sep="\t")
    # Read the length of the existing preprocessed dataframe if it's already partially preprocessed
    already_preprocessed_items = 0
    if os.path.exists(
        os.path.join(PRETRAINING_DIR, "pretraining_transe_preprocessed_positive.tsv")
    ):
        already_preprocessed_items = len(
            pd.read_csv(
                os.path.join(PRETRAINING_DIR, "pretraining_transe_preprocessed_positive.tsv"),
                sep="\t",
                index_col=None,
                usecols=[0],  # only read 1 column to determine the length
            )
        )
        logger.info(
            f"Found an existing file with {already_preprocessed_items} many preprocessed triples"
        )
    # Only deal with the part that hasn't been preprocessed yet
    pretraining_df_not_processed = pretraining_df.iloc[already_preprocessed_items:]
    pretraining_df_not_processed.reset_index()

    # Check if all pretraining entities are covered by the embedding dict
    number_of_pre_training_nodes = len(
        set(pretraining_df["source"]).union(set(pretraining_df["target"]))
    )
    if number_of_pre_training_nodes > len(kg_embed_dict):
        logger.warning(
            "The learned KG embeddings do not cover all of the nodes in the pre-training data"
        )
        return

    # Initialize a FAST tokenizer if it's the default one (BioBERT)
    if nlp_model_type == NLP_MODEL_TYPE:
        # Initialize the fast tokenizer for getting the text token ids
        tokenizer = BertTokenizerFast(vocab_file=VOCAB_FILE)
    else:
        # Initialize a slow tokenizer used for getting the text token ids
        tokenizer = BertTokenizer.from_pretrained(nlp_model_type)

    # Prepare chunk-wise processing of the dataframe
    chunks = [
        pretraining_df_not_processed[i : i + chunk_size]
        for i in range(0, pretraining_df_not_processed.shape[0], chunk_size)
    ]

    # Log progress with a progress bar
    for i, chunk in enumerate(
        tqdm(
            chunks,
            total=len(chunks),
            desc="Processing the dataframe chunk-wise",
        )
    ):
        pre_training_preprocessed_partial = []
        skip_count = 0

        for idx, row in tqdm(
            chunk.iterrows(),  # Different start index if partially processed
            total=chunk.shape[0],
            desc=f"Generating positive samples for chunk no. {i} out of {len(chunks)} chunks left",
            leave=True,
            position=1,
        ):
            # 1. "Token type IDs": 0 for text tokens, 1 for TransE embedding "tokens"
            token_type_ids = [0] * text_part_length + [1] * 4

            # 2. Tokenization for getting the input ids and attention masks for the text
            # Use encode_plus to also get the attention mask ("padding" mask)
            encoded_text = tokenizer.encode_plus(
                row["evidence"],
                padding="max_length",
                truncation=True,
                max_length=text_part_length,
            )
            text_token_ids = encoded_text["input_ids"]
            text_attention_mask = encoded_text["attention_mask"]

            # !! TransESTonKGs-specific !!
            # 3. Get the TransE-based input sequence part, add the SEP (usually with id=102) afterwards
            try:
                transe_embedding_part = (
                    [kg_name_to_idx[row["source"]]]
                    + [kg_name_to_idx[row["relation"]]]
                    + [kg_name_to_idx[row["target"]]]
                    + [sep_id]
                )
            except KeyError:
                skip_count += 1
                continue

            # 4. Total attention mask (attention mask is all 1 for the entity sequence)
            attention_mask = text_attention_mask + [1] * 4

            # Apply the masking strategy to the text tokens and get the text MLM labels
            masked_lm_token_ids, masked_lm_labels = replace_mlm_tokens(
                tokens=text_token_ids,
                vocab_len=len(tokenizer.vocab),
            )
            # Apply the masking strategy to the entity tokens and get the entity (E)LM labels
            # Use the same mask_id as in the NLP model (handled appropriately by STonKGs later on)
            ent_masked_lm_token_ids, ent_masked_lm_labels = replace_mlm_tokens(
                tokens=transe_embedding_part,
                vocab_len=len(kg_embed_dict),
            )

            input_ids = masked_lm_token_ids + ent_masked_lm_token_ids

            # Add all the features to the preprocessed data
            pre_training_preprocessed_partial.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "masked_lm_labels": masked_lm_labels,
                    "ent_masked_lm_labels": ent_masked_lm_labels,
                    "next_sentence_labels": 0,  # 0 indicates the random walks belong to the text evidence
                }
            )

        pre_training_preprocessed_df_partial = pd.DataFrame(pre_training_preprocessed_partial)

        # Keep track of how many triples need to be skipped
        logger.info(f"{skip_count} many examples were skipped")

        pre_training_preprocessed_df_partial.to_csv(
            os.path.join(PRETRAINING_DIR, "pretraining_transe_preprocessed_positive.tsv"),
            sep="\t",
            index=False,
            mode="a",
        )

    logger.info("Finished generating positive training examples")

    # Load the positive examples to save them as a pickle
    pre_training_preprocessed_df = pd.read_csv(
        os.path.join(PRETRAINING_DIR, "pretraining_transe_preprocessed_positive.tsv"),
        sep="\t",
    )
    pre_training_preprocessed_df.to_pickle(
        os.path.join(PRETRAINING_DIR, "pretraining_transe_preprocessed_positive.pkl"),
    )

    """
    # Use this in case the script crashes during the generation of negative samples
    # Load the positive examples
    pre_training_preprocessed_df = pd.read_pickle(
        os.path.join(PRETRAINING_DIR, 'pretraining_preprocessed_positive.pkl'),
    )
    """

    # Generate the negative NSP training samples
    pre_training_negative_samples = _add_negative_nsp_samples(
        pre_training_preprocessed_df,
        nsp_negative_proportion=nsp_negative_proportion,
    )

    # And append them to the original data
    pre_training_preprocessed_df = pre_training_preprocessed_df.append(
        pre_training_negative_samples
    ).reset_index(drop=True)

    logger.info("Finished generating negative training examples")

    # Shuffle the dataframe just to be sure
    pre_training_preprocessed_df_shuffled = pre_training_preprocessed_df.iloc[
        np.random.permutation(pre_training_preprocessed_df.index)
    ].reset_index(drop=True)

    logger.info("Finished shuffling the data")

    # Save the final dataframe
    pre_training_preprocessed_df_shuffled.to_csv(
        os.path.join(PRETRAINING_DIR, "pretraining_transe_preprocessed.tsv"),
        sep="\t",
        index=False,
    )
    # Pickle it, too (easier for reading in the lists in the pandas dataframe)
    pre_training_preprocessed_df_shuffled.to_pickle(
        os.path.join(PRETRAINING_DIR, "pretraining_transe_preprocessed.pkl"),
    )

    logger.info(f"Saved the data under {PRETRAINING_DIR}")

    return pre_training_preprocessed_df


if __name__ == "__main__":
    # Run pre-processing
    indra_to_transe_pretraining_df()
