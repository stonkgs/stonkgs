# -*- coding: utf-8 -*-

"""Full example of how to use a fine tuned model."""

import click
import pandas as pd

from stonkgs import STonKGsForSequenceClassification, preprocess_df_for_embeddings
from stonkgs.api.constants import SPECIES_MODULE, ensure_embeddings, ensure_walks


def main():
    """Example application of the species model."""
    walks_path = ensure_walks()
    embeddings_path = ensure_embeddings()

    click.echo('Model loading')
    model = STonKGsForSequenceClassification.from_pretrained(
        SPECIES_MODULE.base,
        kg_embedding_dict_path=embeddings_path,
    )
    click.echo('Model loaded')

    rows = [
        [
            "p(HGNC:1748 ! CDH1)",
            "p(HGNC:6871 ! MAPK1)",
            "p(HGNC:3229 ! EGF)",
        ],
        [
            "p(HGNC:2515 ! CTNND1)",
            "p(HGNC:6018 ! IL6)",
            "p(HGNC:4066 ! GAB1)",
        ],
        [
            "Some example sentence about CDH1 and CTNND1.",
            "Another example about some interaction between MAPK and IL6.",
            "One last example in which Gab1 and EGF are mentioned.",
        ],
    ]

    df = pd.DataFrame(rows, columns=["source", "target", "evidence"])

    click.echo("Processing df for embeddings")
    preprocessed_df = preprocess_df_for_embeddings(
        df=df,
        embedding_name_to_vector_path=embeddings_path,
        embedding_name_to_random_walk_path=walks_path,
        vocab_file_path=...,
    )
    click.echo("done processing df for embeddings")

    # TODO finish
    model.predict()


if __name__ == "__main__":
    main()
