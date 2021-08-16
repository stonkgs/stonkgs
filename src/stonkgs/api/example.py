# -*- coding: utf-8 -*-

"""Full example of how to use a fine tuned model."""

import pandas as pd

from stonkgs import get_stonkgs_embeddings, preprocess_df_for_embeddings
from stonkgs.api.datasets import ensure_embeddings, ensure_walks


def main():
    """Example application of the species model."""
    rows = []
    df = pd.DataFrame(rows, columns=['source', 'target', 'evidence'])

    walks_path = ensure_walks()
    embeddings_path = ensure_embeddings()

    preprocessed_df = preprocess_df_for_embeddings(
        df=df,
        embedding_name_to_vector_path=embeddings_path,
        embedding_name_to_random_walk_path=walks_path,
    )

    embeddings_df = get_stonkgs_embeddings(preprocessed_df=preprocessed_df)

    # TODO finish


if __name__ == '__main__':
    main()
