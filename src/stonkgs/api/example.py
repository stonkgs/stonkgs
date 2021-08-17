# -*- coding: utf-8 -*-

"""Full example of how to use a fine tuned model.

Run with: ``python -m stonkgs.api.example``
"""

import click
import pystow

from stonkgs import infer_species

SPECIES_PREDICTION_PATH = pystow.join("stonkgs", "species", name="predictions.tsv")


def main():
    """Do an example application of the species model."""
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
    output_df = infer_species(rows)
    output_df.to_csv(SPECIES_PREDICTION_PATH, sep="\t", index=False)
    click.echo(f"Results at {SPECIES_PREDICTION_PATH}")


if __name__ == "__main__":
    main()
