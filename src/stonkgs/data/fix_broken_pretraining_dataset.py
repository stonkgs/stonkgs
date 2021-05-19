# -*- coding: utf-8 -*-

"""Fixes wrongly preprocessed lists of ints in the preprocessed pandas dataframes.

Run with:
python -m src.stonkgs.data.fix_broken_pretraining_dataset
"""
import os
import pandas as pd
from tqdm import tqdm

from stonkgs.constants import (
    PRETRAINING_DIR,
)


def convert_tsv_with_str_list(
    file_name: str,
    path: str = PRETRAINING_DIR,
):
    """Converts strings that look like lists into actual lists of ints and saves a pickle of the repaired dataframe."""
    converter = lambda x: [int(y) for y in x.strip("[]").split(", ")]
    converter_dict = {col: converter for col in
                      ['input_ids', 'attention_mask', 'token_type_ids', 'masked_lm_labels', 'ent_masked_lm_labels']
                      }
    chunks = []

    # Process dataframe in bits with a progress bar
    for chunk in tqdm(pd.read_csv(
        os.path.join(path, file_name + '.tsv'),
        sep="\t",
        chunksize=1000000,
        converters=converter_dict),
        total=14,  # Hard coded total number
    ):
        chunks.append(chunk)
    complete_df = pd.concat(chunks, axis=0)

    # Pickle the complete dataframe
    complete_df.to_pickle(
        os.path.join(PRETRAINING_DIR, file_name + '.pkl')
    )


if __name__ == "__main__":
    # Fixing the positive dataset
    convert_tsv_with_str_list('pretraining_preprocessed_positive')
