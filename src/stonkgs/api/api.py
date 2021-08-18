# -*- coding: utf-8 -*-

"""Functionality for ensuring the fine-tuned models are ready to use."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

import pandas as pd
import pybel.constants as pc
import pystow
import torch
import torch.nn.functional
from indra.assemblers.pybel import PybelAssembler
from indra.statements import Statement
from tqdm import tqdm
from transformers.trainer_utils import PredictionOutput

from ..models.stonkgs_finetuning import STonKGsForSequenceClassification
from ..models.stonkgs_for_embeddings import preprocess_df_for_embeddings_iter

logger = logging.getLogger(__name__)

InferenceHint = Union[pd.DataFrame, List[List[str]], List[Statement]]

STONKGS = pystow.module("stonkgs")

SPECIES_RECORD = "5205530"
LOCATION_RECORD = "5205553"
DISEASE_RECORD = "5205592"
CORRECT_MULTICLASS_RECORD = "5206139"
CORRECT_BINARY_RECORD = "5205989"
CELL_LINE_RECORD = "5205915"

WALKS_URL = "https://zenodo.org/record/5205687/files/random_walks_best_model.tsv"
EMBEDDINGS_URL = "https://zenodo.org/record/5205687/files/embeddings_best_model.tsv"

POLARITY_COLUMNS = ["down", "up"]
INTERACTION_COLUMNS = ["direct_interaction", "indirect_interaction"]
SPECIES_COLUMNS = ["mouse", "rat", "human"]
LOCATION_COLUMNS = [
    "extracellular_space",
    "cell_membrane",
    "cell_nucleus",
    "extracellular_matrix",
    "cytoplasm",
]
DISEASE_COLUMNS = [
    "neuroblastoma",
    "multiple_myeloma",
    "lung_non-small_cell_carcinomaleukemia",
    "breast_cancer",
    "lung_cancer",
    "atherosclerosis",
    "osteosarcoma",
    "melanoma",
    "leukemia",
    "colon_cancer",
]
CORRECT_MULTICLASS_COLUMNS = [
    "act_vs_amt",
    "grounding",
    "hypothesis",
    "entity_boundaries",
    "no_relation",
    "correct",
    "wrong_relation",
    "polarity",
]
CORRECT_BINARY_COLUMNS = ["incorrect", "correct"]
CELL_LINE_COLUMNS = [
    "HeLa",
    "THP-1",
    "LNCAP",
    "COS-1",
    "DMS_114",
    "NIH-3T3",
    "HEK293",
    "MCF7",
    "Hep_G2",
    "U-937",
]


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


def _get_model(f: Callable[[], Path]) -> STonKGsForSequenceClassification:
    path = f().parent
    logger.info("loading STonKGs sequence classifier model from %s", path)
    rv = STonKGsForSequenceClassification.from_pretrained(
        path,
        kg_embedding_dict_path=ensure_embeddings(),
    )
    logger.info("done loading STonKGs sequence classifier model")
    return rv


def ensure_species() -> Path:
    """Ensure that the species model is downloaded from `Zenodo <https://zenodo.org/record/5205530>`_.

    :returns: The path to the model binary

    .. warning:: It's pretty big - 1.4GB. Be patient!
    """
    return _ensure_fine_tuned("species", SPECIES_RECORD)


@lru_cache(maxsize=1)
def get_species_model() -> STonKGsForSequenceClassification:
    """Get the species model."""
    return _get_model(ensure_species)


def infer_species(data: InferenceHint) -> pd.DataFrame:
    """Infer the species for the given input."""
    return infer_concat(get_species_model(), data, columns=SPECIES_COLUMNS)


def ensure_location() -> Path:
    """Ensure that the location model is downloaded."""
    return _ensure_fine_tuned("location", LOCATION_RECORD)


@lru_cache(maxsize=1)
def get_location_model() -> STonKGsForSequenceClassification:
    """Get the location model."""
    return _get_model(ensure_location)


def infer_locations(data: InferenceHint) -> pd.DataFrame:
    """Infer the locations for the given input."""
    return infer_concat(get_location_model(), data, columns=LOCATION_COLUMNS)


def ensure_disease() -> Path:
    """Ensure that the disease model is downloaded."""
    return _ensure_fine_tuned("disease", DISEASE_RECORD)


@lru_cache(maxsize=1)
def get_disease_model() -> STonKGsForSequenceClassification:
    """Get the disease model."""
    return _get_model(ensure_disease)


def infer_diseases(data: InferenceHint) -> pd.DataFrame:
    """Infer the diseases for the given input."""
    return infer_concat(get_disease_model(), data, columns=DISEASE_COLUMNS)


def ensure_correct_multiclass() -> Path:
    """Ensure that the correct (multiclass) model is downloaded."""
    return _ensure_fine_tuned("correct_multiclass", CORRECT_MULTICLASS_RECORD)


@lru_cache(maxsize=1)
def get_correct_multiclass_model() -> STonKGsForSequenceClassification:
    """Get the correct (multiclass) model."""
    return _get_model(ensure_correct_multiclass)


def infer_correct_multiclass(data: InferenceHint) -> pd.DataFrame:
    """Infer the correct multiclass output for the given input."""
    return infer_concat(get_correct_multiclass_model(), data, columns=CORRECT_MULTICLASS_COLUMNS)


def ensure_correct_binary() -> Path:
    """Ensure that the correct (binary) model is downloaded."""
    return _ensure_fine_tuned("correct_binary", CORRECT_BINARY_RECORD)


@lru_cache(maxsize=1)
def get_correct_binary_model() -> STonKGsForSequenceClassification:
    """Get the correct (binary) model."""
    return _get_model(ensure_correct_binary)


def infer_correct_binary(data: InferenceHint) -> pd.DataFrame:
    """Infer the correct binary output for the given input.

    :param data: A pandas dataframe or rows to a dataframe with source, target, and evidence as columns
    :return: A pandas dataframe with source, target, evidence, incorrect probability, and correct probability based
        on :data:`CORRECT_BINARY_COLUMNS`.

    >>> from stonkgs import infer_correct_binary
    >>> rows = [
    ...     [
    ...         "p(HGNC:17927 ! SENP1)",
    ...         "p(HGNC:4910 ! HIF1A)",
    ...         "Hence, deSUMOylation of HIF-1alpha by SENP1 could prevent degradation of HIF-1alpha "],
    ...     ],
    ... ]
    >>> df = infer_correct_binary(rows)
    """
    return infer_concat(get_correct_binary_model(), data, columns=CORRECT_BINARY_COLUMNS)


def ensure_cell_line() -> Path:
    """Ensure that the cell line model is downloaded."""
    return _ensure_fine_tuned("cell_line", CELL_LINE_RECORD)


@lru_cache(maxsize=1)
def get_cell_line_model() -> STonKGsForSequenceClassification:
    """Get the cell line model."""
    return _get_model(ensure_cell_line)


def infer_cell_lines(data: InferenceHint) -> pd.DataFrame:
    """Infer the cell lines for the given input."""
    return infer_concat(get_cell_line_model(), data, columns=CELL_LINE_COLUMNS)


KEEP_COLUMNS = ["input_ids", "attention_mask", "token_type_ids"]


def infer_concat(
    model: STonKGsForSequenceClassification,
    data: InferenceHint,
    *,
    columns: Optional[List[str]] = None,
    as_dataframe: bool = False,
) -> Union[Iterable, pd.DataFrame]:
    """Run inference and return the input with output columns concatenated."""
    rv = iter(infer_concat_iter(model, data, columns=columns))
    if as_dataframe:
        return pd.DataFrame(rv, columns=next(rv))
    return rv


def infer_concat_iter(
    model: STonKGsForSequenceClassification,
    data: InferenceHint,
    columns: Optional[List[str]] = None,
) -> Iterable:
    """Run inference and return the input with output columns concatenated."""
    _data: pd.DataFrame = _prepare_df(data)
    if columns is not None:
        header = *_data.columns, *columns
        logger.info("header for output: %s", header)
        yield header
    for row, (_raw_results, probabilities) in zip(_data.values, infer_iter(model, _data)):
        yield *row, *probabilities


INDRA_DF_COLUMNS = [
    "stmt_hash",
    "belief",
    "source",
    "target",
    "evidence",
]


def _convert_indra_statements(statements: Iterable[Statement]) -> pd.DataFrame:
    logger.info("assembling INDRA statements to PyBEL")
    assembler = PybelAssembler(statements)
    bel_graph = assembler.make_model()
    logger.info("done assembling INDRA statements to PyBEL")
    rows = []
    for u, v, data in bel_graph.edges(data=True):
        if pc.ANNOTATIONS not in data:
            continue
        rows.append(
            (
                list(data[pc.ANNOTATIONS]["stmt_hash"].keys())[0],
                list(data[pc.ANNOTATIONS]["belief"].keys())[0],
                str(u),
                str(v),
                data[pc.EVIDENCE],
            )
        )
    return pd.DataFrame(rows, columns=INDRA_DF_COLUMNS)


def _prepare_df(data: InferenceHint) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if not isinstance(data, list):
        raise TypeError(f"source df has invalid type: {type(data)}")
    if isinstance(data[0], (list, tuple)):
        return pd.DataFrame(data, columns=["source", "target", "evidence"])
    elif isinstance(data[0], Statement):
        return _convert_indra_statements(data)
    else:
        raise TypeError(f"row has invalid type: {type(data[0])}")


def infer(model: STonKGsForSequenceClassification, data: InferenceHint):
    """Run inference on a given model."""
    # Save both the raw prediction results (as a pickle) as well as the processed probabilities (in a dataframe)
    raw_results, probabilities = [], []
    for r, p in infer_iter(model, data):
        raw_results.append(r)
        probabilities.append(p)
    return raw_results, probabilities


def infer_iter(
    model: STonKGsForSequenceClassification, data: InferenceHint
) -> Iterable[Tuple[PredictionOutput, torch.FloatTensor]]:
    """Run inference on a given model."""
    df = _prepare_df(data)
    rows = preprocess_df_for_embeddings_iter(
        rows=df[["source", "target", "evidence"]].values,
        embedding_name_to_vector_path=ensure_embeddings(),
        embedding_name_to_random_walk_path=ensure_walks(),
    )
    for row in tqdm(rows, total=len(df.index), desc="Inferring"):
        prediction_output: PredictionOutput = model.forward(
            input_ids=torch.tensor([row["input_ids"]]),
            attention_mask=torch.tensor([row["attention_mask"]]),
            token_type_ids=torch.tensor([row["token_type_ids"]]),
            return_dict=True,
        )
        probability = torch.nn.functional.softmax(prediction_output.logits, dim=1)[0].tolist()
        yield prediction_output, probability
