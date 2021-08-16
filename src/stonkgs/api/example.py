# -*- coding: utf-8 -*-

"""Full example of how to use a fine tuned model.

Run with: ``python -m stonkgs.api.example``
"""

import pickle
import time

import click
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers.trainer_utils import PredictionOutput

from stonkgs import STonKGsForSequenceClassification, preprocess_df_for_embeddings
from stonkgs.api.constants import ensure_embeddings, ensure_species, ensure_walks


def main():
    """Example application of the species model."""
    # Ensure that all the necessary files are loaded (embeddings, random walks, fine-tuned model)
    species_path = ensure_species()
    walks_path = ensure_walks()
    embeddings_path = ensure_embeddings()

    click.echo("Model loading")
    t = time.time()
    model = STonKGsForSequenceClassification.from_pretrained(
        species_path.parent,
        kg_embedding_dict_path=embeddings_path,
    )
    click.echo(f"Model {model.__class__.__name__} loaded in {time.time() - t:.2f} seconds")
    click.echo(repr(model))

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

    source_df = pd.DataFrame(rows, columns=["source", "target", "evidence"])

    click.echo("Processing df for embeddings")
    t = time.time()
    preprocessed_df = preprocess_df_for_embeddings(
        df=source_df,
        embedding_name_to_vector_path=embeddings_path,
        embedding_name_to_random_walk_path=walks_path,
    )[["input_ids", "attention_mask", "token_type_ids"]]
    click.echo(f"done processing df for embeddings after {time.time() - t:.2f} seconds")

    dataset = Dataset.from_pandas(preprocessed_df)
    dataset.set_format('torch')

    # Three entries in this named tuple: predictions, label_ids, metrics
    results = []
    for idx, row in tqdm(preprocessed_df.iterrows(), desc="Inferring"):
        # Process each row at once
        data_entry = {
            key: torch.tensor([value]) for key, value in dict(preprocessed_df.iloc[idx]).items()
        }

        prediction_output: PredictionOutput = model(**data_entry, return_dict=True)
        results.append(prediction_output)

    with open("/home/hbalabin/Downloads/results.pkl", "wb") as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

    # output_df = pd.DataFrame({
    #     "predictions": prediction_output.predictions,
    #     "label_ids": prediction_output.label_ids,
    # })
    # output_df.to_csv('/Users/cthoyt/Desktop/results.tsv', sep='\t', index=False)


if __name__ == "__main__":
    main()
