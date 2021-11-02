# -*- coding: utf-8 -*-

"""Fixes wrongly preprocessed lists of ints in the preprocessed pandas dataframes.

Run with:
python -m src.stonkgs.data.fix_broken_pretraining_dataset
"""
import os

import click
import pandas as pd
from tqdm import tqdm

from stonkgs.constants import (
    PRETRAINING_DIR,
)


@click.command()
@click.option(
    "--chunk_size",
    default=100000,
    help="Size of the chunks used for processing the corrupted file",
    type=int,
)
@click.option(
    "--input_path",
    default=os.path.join(PRETRAINING_DIR, "pretraining_transe_preprocessed.tsv"),
    help="Path to the corrupted file (a .tsv file)",
    type=str,
)
@click.option(
    "--output_path",
    default=os.path.join(PRETRAINING_DIR, "pretraining_transe_preprocessed_fixed.pkl"),
    help="Output file path specification (ending in .pkl)",
    type=str,
)
def convert_tsv_to_pkl(
    chunk_size: int,
    input_path: str,
    output_path: str,
):
    """Converts strings that look like lists into actual lists of ints and saves a pickle of the repaired dataframe."""
    converter_dict = {
        col: lambda x: [int(y) for y in x.strip("[]").split(", ")]
        for col in [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "masked_lm_labels",
            "ent_masked_lm_labels",
        ]
    }
    converter_dict["next_sentence_labels"] = lambda x: int(x)  # type: ignore  # noqa
    chunks = []

    # Process dataframe in bits with a progress bar
    for chunk in tqdm(
        pd.read_csv(
            input_path,
            sep="\t",
            chunksize=chunk_size,
            converters=converter_dict,
        )
    ):
        chunks.append(chunk)
    complete_df = pd.concat(chunks, axis=0)

    # Pickle the complete dataframe
    complete_df.to_pickle(output_path)


if __name__ == "__main__":
    # Fixing the dataset
    convert_tsv_to_pkl()
