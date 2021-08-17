# -*- coding: utf-8 -*-

"""Code for downstream users to take advantage of the pre-trained and fine-tuned models."""

from .api import (  # noqa:F401
    ensure_cell_line,
    ensure_correct_binary,
    ensure_correct_multiclass,
    ensure_disease,
    ensure_embeddings,
    ensure_location,
    ensure_species,
    ensure_walks,
    get_cell_line_model,
    get_correct_binary_model,
    get_correct_multiclass_model,
    get_disease_model,
    get_location_model,
    get_species_model,
    infer,
    infer_cell_lines,
    infer_concat,
    infer_correct_binary,
    infer_correct_multiclass,
    infer_diseases,
    infer_locations,
    infer_species,
)
