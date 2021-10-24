# -*- coding: utf-8 -*-

"""Prepares the pre-training data based on TransE embeddings for STonKGs.

Run with:
python -m src.stonkgs.data.transe_indra_for_pretraining
"""

import logging
import os

import pandas as pd
from transformers import BertTokenizer, BertTokenizerFast
from tqdm import tqdm

from stonkgs.constants import (
    TRANSE_EMBEDDINGS_PATH,
    NLP_MODEL_TYPE,
    PRETRAINING_DIR,
    PRETRAINING_PREPROCESSED_DF_PATH,
    VOCAB_FILE,
)
from stonkgs.models.kg_baseline_model import prepare_df

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def indra_to_transe_pretraining_df(
    embedding_name_to_vector_path: str = TRANSE_EMBEDDINGS_PATH,
    preprocessed_path: str = PRETRAINING_PREPROCESSED_DF_PATH,
    nlp_model_type: str = NLP_MODEL_TYPE,
    text_part_length: int = 256,
    sep_id: int = 102,
):
    """Preprocesses the INDRA statements from the pre-training file so that it contains all the necessary attributes."""
    # Load the KG embedding dict to convert the names to numeric indices
    kg_embed_dict = prepare_df(embedding_name_to_vector_path)
    kg_name_to_idx = {key: i for i, key in enumerate(kg_embed_dict.keys())}
    # Keep track of how many triples need to be skipped
    skip_count = 0

    # Log the number of entities
    logger.info(f"There are {len(kg_embed_dict)} many entities in the pre-trained KG")

    # Load the pre-training dataframe
    pretraining_preprocessed_df = pd.read_csv(preprocessed_path, sep="\t")

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
        pretraining_preprocessed_df.iterrows(),
        total=pretraining_preprocessed_df.shape[0],
        desc="Altering the default STonKGs entries",
    ):
        # 1. Load the tokenized input IDs for the text part
        text_token_ids = row["input_ids"][:text_part_length]

        # !! TransESTonKGs-specific !!
        # 2. Create the TransE-based input sequence part, add the SEP (usually with id=102) afterwards
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

        # !! TransESTonKGs-specific !!
        # 3. And replace/extend the KG part of the input IDs and attention mask with TransE embeddings
        input_ids = text_token_ids + transe_embedding_part

        # Add all the features to the preprocessed data
        pre_training_preprocessed.append(
            {
                "input_ids": input_ids,
                "attention_mask": row["attention_mask"][:text_part_length] + 4 * [1],  # 1 for the TransE part
                "token_type_ids": row["token_type_ids"][:text_part_length] + 4 * [1],  # 1 for the TransE part
                "masked_lm_labels": row["masked_lm_labels"][:text_part_length],
                "ent_masked_lm_labels": [-100] * 4,
                "next_sentence_labels": row["next_sentence_labels"],
            }
        )

    # Put the preprocessed data into a dataframe
    pre_training_preprocessed_df = pd.DataFrame(pre_training_preprocessed)

    logger.info(f"{skip_count} many examples were skipped")

    # Save the final dataframe
    pre_training_preprocessed_df.to_csv(
        os.path.join(PRETRAINING_DIR, "pretraining_transe_preprocessed.tsv"),
        sep="\t",
        index=False,
    )
    # Pickle it, too (easier for reading in the lists in the pandas dataframe)
    pre_training_preprocessed_df.to_pickle(
        os.path.join(PRETRAINING_DIR, "pretraining_transe_preprocessed.pkl"),
    )

    logger.info(f"Saved the data under {PRETRAINING_DIR}")

    return pre_training_preprocessed_df


if __name__ == "__main__":
    # Run pre-processing
    indra_to_transe_pretraining_df()
