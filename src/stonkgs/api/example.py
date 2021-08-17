# -*- coding: utf-8 -*-

"""Full example of how to use a fine tuned model.

Run with: ``python -m stonkgs.api.example``
"""

import pickle
import time

import click
import pandas as pd
import pystow

from stonkgs import STonKGsForSequenceClassification
from stonkgs.api import ensure_embeddings, ensure_species, infer


def main():
    """Do an example application of the species model."""
    # Ensure that all the necessary files are loaded (embeddings, random walks, fine-tuned model)
    species_path = ensure_species()
    embeddings_path = ensure_embeddings()

    click.echo("Model loading")
    t = time.time()
    model = STonKGsForSequenceClassification.from_pretrained(
        species_path.parent,
        kg_embedding_dict_path=embeddings_path,
    )
    click.echo(f"Model {model.__class__.__name__} loaded in {time.time() - t:.2f} seconds")

    rows = [
        [
            "p(HGNC:1748 ! CDH1)",
            "p(HGNC:2515 ! CTNND1)",
            "Some example sentence about CDH1 and CTNND1.",
        ],
        [
            "p(HGNC:6871 ! MAPK1)",
            "p(HGNC:6018 ! IL6)",
            "Another example about some interaction between MAPK and IL6.",
        ],
        [
            "p(HGNC:3229 ! EGF)",
            "p(HGNC:4066 ! GAB1)",
            "One last example in which Gab1 and EGF are mentioned.",
        ],
    ]
    source_df = pd.DataFrame(rows, columns=["source", "target", "evidence"])

    # Save both the raw prediction results (as a pickle) as well as the processed probabilities (in a dataframe)
    raw_results, probabilities = infer(model, source_df)

    # Save as pickle
    pickle_path = pystow.join("stonkgs", "species", name="predictions.pkl")
    with pickle_path.open("wb") as file:
        pickle.dump(raw_results, file, protocol=pickle.HIGHEST_PROTOCOL)

    # Save as a dataframe
    df_path = pystow.join("stonkgs", "species", name="predictions.tsv")
    probabilities_df = pd.DataFrame(probabilities, columns=["mouse", "rat", "human"])
    output_df = pd.concat([source_df, probabilities_df], axis=1)
    output_df.to_csv(df_path, sep="\t", index=False)
    click.echo(f"Results at {df_path}")


if __name__ == "__main__":
    main()
