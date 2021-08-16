# -*- coding: utf-8 -*-

"""Functionality for ensuring the fine-tuned models are ready to use."""

from pathlib import Path

import pystow

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
