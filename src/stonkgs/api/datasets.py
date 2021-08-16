# -*- coding: utf-8 -*-

"""Functionality for ensuring the fine-tuned models are ready to use."""

from pathlib import Path

import pystow

STONKGS = pystow.module('stonkgs')

SPECIES_RECORD = '5205530'
SPECIES_CONFIG = 'https://zenodo.org/record/5205530/files/config.json'
SPECIES_BIN = 'https://zenodo.org/record/5205530/files/pytorch_model.bin'
SPECIES_TRAINING = 'https://zenodo.org/record/5205530/files/training_args.bin'
SPECIES_MODULE = STONKGS.submodule('species')

WALKS_RECORD = ''
WALKS_URL = ''

EMBEDDINGS_RECORD = ''
EMBEDDINGS_URL = ''


def ensure_walks() -> Path:
    """Ensure the walks file is downloaded from zenodo."""
    return STONKGS.ensure(url=WALKS_URL)


def ensure_embeddings() -> Path:
    """Ensure the embeddings file is downloaded from zenodo."""
    return STONKGS.ensure(url=EMBEDDINGS_URL)


def ensure_species() -> Path:
    """Ensure that the species model is downloaded from `Zenodo <https://zenodo.org/record/5205530>`_.

    :returns: The path to the model binary

    .. warning:: It's pretty big - 1.4GB. Be patient!
    """
    SPECIES_MODULE.ensure(url=SPECIES_CONFIG)
    SPECIES_MODULE.ensure(url=SPECIES_TRAINING)
    return SPECIES_MODULE.ensure(url=SPECIES_BIN)
