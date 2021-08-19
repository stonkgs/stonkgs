# -*- coding: utf-8 -*-

"""Demo of assessing a full EMMAA model."""

import csv
import gzip
import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import click
import matplotlib.pyplot as plt
import more_click
import pandas as pd
import pystow
import seaborn as sns
from indra.statements import Statement, stmts_from_json

import stonkgs

MARM_URL = "https://emmaa.s3.amazonaws.com/assembled/marm_model/statements_2021-08-17-17-31-53.gz"
RAS_URL = "https://emmaa.s3.amazonaws.com/assembled/rasmachine/statements_2021-08-16-19-22-38.gz"
COVID_URL = "https://emmaa.s3.amazonaws.com/assembled/covid19/statements_2021-08-16-20-29-07.gz"
NF_URL = "https://emmaa.s3.amazonaws.com/assembled/nf/statements_2021-08-16-18-37-34.gz"
VT_URL = "https://emmaa.s3.amazonaws.com/assembled/vitiligo/statements_2021-08-17-18-38-35.gz"


def get_statements(url: str) -> Tuple[Path, List[Statement]]:
    """Get EMMAA statements."""
    statements_path = pystow.ensure("stonkgs", "demos", "emmaa", url.split("/")[-2], url=url)
    with gzip.open(statements_path, "rt") as file:
        statements: List[Statement] = stmts_from_json(json.load(file))
    return statements_path, statements


def run_emmaa_demo(url: str):
    """Run the EMMAA demo."""
    statements_path, statements = get_statements(url)
    results_path = statements_path.with_suffix(".results.tsv")
    scatter_path = statements_path.with_suffix(".scatter.svg")
    curation_path = statements_path.with_suffix(".curation.pkl")

    it = iter(stonkgs.infer_correct_binary(statements))
    header = next(it)
    first = next(it)
    # why do two calls to next()? to make sure it's successful before opening the file.
    with results_path.open(mode="w") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(header)
        writer.writerow(first)
        writer.writerows(it)

    df = pd.read_csv(results_path, usecols=[0, 1, 6], sep="\t")
    belief_lower, belief_upper = 0.2, 0.85
    stonkgs_lower, stonkgs_upper = 0.2, 0.85
    idx = (
        ((df.belief < belief_lower) & (df.correct < stonkgs_lower))  # bad belief, bad stonkgs
        | ((df.belief < belief_lower) & (df.correct > stonkgs_upper))  # bad belief, good stonkgs
        | ((df.belief > belief_upper) & (df.correct < stonkgs_lower))  # good belief, bad stonkgs
        | ((df.belief > belief_upper) & (df.correct > stonkgs_upper))  # good belief, good stonkgs
    )
    curate_hashes = set(df.loc[idx].stmt_hash.unique())
    click.echo(f"Got {len(curate_hashes)} statements for curation")
    export_stmts = [statement for statement in statements if statement.get_hash() in curate_hashes]
    with curation_path.open("wb") as curation_file:
        pickle.dump(export_stmts, curation_file)

    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(data=df, x="correct", y="belief", ax=ax)
    fig.savefig(scatter_path)


@click.command()
@more_click.verbose_option
@click.option("--url", default=VT_URL)
def main(url: str):
    """Run the EMMAA demo from the CLI."""
    logging.getLogger("indra.assemblers.pybel.assembler").setLevel(logging.ERROR)
    run_emmaa_demo(url)


if __name__ == "__main__":
    main()
