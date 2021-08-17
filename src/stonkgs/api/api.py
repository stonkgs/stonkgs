# -*- coding: utf-8 -*-

"""Functionality for ensuring the fine-tuned models are ready to use."""

import time
from pathlib import Path
from typing import List, Union

import click
import pandas as pd
import pystow
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers.trainer_utils import PredictionOutput

from ..models.stonkgs_finetuning import STonKGsForSequenceClassification
from ..models.stonkgs_for_embeddings import preprocess_df_for_embeddings

STONKGS = pystow.module("stonkgs")

SPECIES_RECORD = "5205530"
LOCATION_RECORD = "5205553"
DISEASE_RECORD = "5205592"
CORRECT_MULTICLASS_RECORD = "5206139"
CORRECT_BINARY_RECORD = "5205989"
CELL_LINE_RECORD = "5205915"

WALKS_URL = "https://zenodo.org/record/5205687/files/random_walks_best_model.tsv"
EMBEDDINGS_URL = "https://zenodo.org/record/5205687/files/embeddings_best_model.tsv"


def ensure_walks() -> Path:
    """Ensure the walks file is downloaded from zenodo."""
    return STONKGS.ensure(url=WALKS_URL)


def ensure_embeddings() -> Path:
    """Ensure the embeddings file is downloaded from zenodo."""
    return STONKGS.ensure(url=EMBEDDINGS_URL)


def _ensure_fine_tuned(submodule, record) -> Path:
    STONKGS.ensure(submodule, url=f"https://zenodo.org/record/{record}/files/config.json")
    STONKGS.ensure(submodule, url=f"https://zenodo.org/record/{record}/files/training_args.bin")
    return STONKGS.ensure(
        submodule, url=f"https://zenodo.org/record/{record}/files/pytorch_model.bin"
    )


def ensure_species() -> Path:
    """Ensure that the species model is downloaded from `Zenodo <https://zenodo.org/record/5205530>`_.

    :returns: The path to the model binary

    .. warning:: It's pretty big - 1.4GB. Be patient!
    """
    return _ensure_fine_tuned("species", SPECIES_RECORD)


def ensure_location() -> Path:
    """Ensure that the location model is downloaded."""
    return _ensure_fine_tuned("location", LOCATION_RECORD)


def ensure_disease() -> Path:
    """Ensure that the disease model is downloaded."""
    return _ensure_fine_tuned("disease", DISEASE_RECORD)


def ensure_correct_multiclass() -> Path:
    """Ensure that the correct (multiclass) model is downloaded."""
    return _ensure_fine_tuned("correct_multiclass", CORRECT_MULTICLASS_RECORD)


def ensure_correct_binary() -> Path:
    """Ensure that the correct (binary) model is downloaded."""
    return _ensure_fine_tuned("correct_binary", CORRECT_BINARY_RECORD)


def ensure_cell_binary() -> Path:
    """Ensure that the cell line model is downloaded."""
    return _ensure_fine_tuned("cell_line", CELL_LINE_RECORD)


def infer(model: STonKGsForSequenceClassification, source_df: Union[pd.DataFrame, List]):
    """Run inference on a given model."""
    if isinstance(source_df, pd.DataFrame):
        pass
    elif isinstance(source_df, list):
        source_df = pd.DataFrame(source_df, columns=["source", "target", "evidence"])
    else:
        raise TypeError
    click.echo("Processing df for embeddings")
    t = time.time()
    preprocessed_df = preprocess_df_for_embeddings(
        df=source_df,
        embedding_name_to_vector_path=ensure_embeddings(),
        embedding_name_to_random_walk_path=ensure_walks(),
    )[["input_ids", "attention_mask", "token_type_ids"]]
    click.echo(f"done processing df for embeddings after {time.time() - t:.2f} seconds")

    dataset = Dataset.from_pandas(preprocessed_df)
    dataset.set_format("torch")

    # Save both the raw prediction results (as a pickle) as well as the processed probabilities (in a dataframe)
    raw_results = []
    probabilities = []
    for idx, _ in tqdm(preprocessed_df.iterrows(), desc="Inferring"):
        # Process each row at once
        data_entry = {
            key: torch.tensor([value]) for key, value in dict(preprocessed_df.iloc[idx]).items()
        }
        prediction_output: PredictionOutput = model(**data_entry, return_dict=True)
        probabilities.append(
            torch.nn.functional.softmax(prediction_output.logits, dim=1)[0].tolist()
        )
        raw_results.append(prediction_output)

    return raw_results, probabilities
