# -*- coding: utf-8 -*-

"""Demo of assessing a full EMMAA model."""

import csv
import gzip
import json
import logging

import matplotlib.pyplot as plt
import pandas as pd
import pystow
import seaborn as sns
from indra.statements import stmts_from_json

import stonkgs

# URL = "https://emmaa.s3.amazonaws.com/assembled/marm_model/statements_2021-08-17-17-31-53.gz"
# URL = "https://emmaa.s3.amazonaws.com/assembled/covid19/statements_2021-08-16-20-29-07.gz"
URL = "https://emmaa.s3.amazonaws.com/assembled/nf/statements_2021-08-16-18-37-34.gz"
STATEMENTS_PATH = pystow.ensure("stonkgs", "demos", "emmaa", URL.split("/")[-2], url=URL)
RESULTS_PATH = STATEMENTS_PATH.with_suffix(".results.tsv")
SCATTER_PATH = STATEMENTS_PATH.with_suffix(".scatter.svg")


def main():
    """Run the EMMAA demo."""
    if RESULTS_PATH.is_file() and False:
        df = pd.read_csv(RESULTS_PATH, sep="\t")
    else:
        with gzip.open(STATEMENTS_PATH, "rt") as file:
            statements = stmts_from_json(json.load(file))
        with RESULTS_PATH.open("w") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerows(stonkgs.infer_correct_binary(statements))

    return
    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(data=df, x="correct", y="belief", ax=ax)
    fig.savefig(SCATTER_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("indra.assemblers.pybel.assembler").setLevel(logging.ERROR)
    main()
