# -*- coding: utf-8 -*-

"""Prepares the pre-training data for STonKGs.

Run with:
python -m src.stonkgs.data.indra_for_pretraining
"""

import pandas as pd

from stonkgs.constants import EMBEDDINGS_PATH, RANDOM_WALKS_PATH, PRETRAINING_PATH
from stonkgs.models.kg_baseline_model import _prepare_df


def indra_to_pretraining_df(
    embedding_name_to_vector_path: str = EMBEDDINGS_PATH,
    embedding_name_to_random_walk_path: str = RANDOM_WALKS_PATH,
    pre_training_path: str = PRETRAINING_PATH,
    nsp_negative_proportion: float = 0.5,
):
    """Preprocesses the INDRA statements from the pre-training file so that it contains all the necessary attributes."""

    # Load the KG embedding dict to convert the names to numeric indices
    kg_embed_dict = _prepare_df(EMBEDDINGS_PATH)
    kg_name_to_idx = {key: i for i, key in enumerate(kg_embed_dict.keys())}

    # Load the random walks for each node
    random_walk_dict = _prepare_df(RANDOM_WALKS_PATH)
    # Convert random walk sequences to list of numeric indices
    random_walk_idx_dict = {k: [kg_name_to_idx[node] for node in v] for k, v in random_walk_dict.items()}

    # Load the pre-training dataframe
    pretraining_df = pd.read_csv(PRETRAINING_PATH, sep='\t')

    # TODO: add assertion here to see if all pretraining entities are covered by the embedding dict
    # print(len(set(pretraining_df["source"]).union(set(pretraining_df["target"]))))

    for idx, row in pretraining_df.iterrows():
        # Generate the random walks
        random_walk = random_walk_idx_dict[row['source']] + random_walk_idx_dict[row['target']]
        print(random_walk)

    # TODO: (Bio)BERT Tokenization at some point for getting the input ids for the text
    # return NotImplementedError()


if __name__ == "__main__":
    # Just simple testing
    indra_to_pretraining_df()
