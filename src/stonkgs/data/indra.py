# -*- coding: utf-8 -*-

"""Reads and prepares triples/statements for INDRA.

Run with:
python -m src.stonkgs.data.indra
"""

import logging
import os
from typing import Any, Dict, List

import pandas as pd
import pybel
from pybel.constants import (
    ANNOTATIONS,
    EVIDENCE,
    RELATION,
    CITATION,
    INCREASES,
    DIRECTLY_INCREASES,
    DECREASES,
    DIRECTLY_DECREASES,
    REGULATES,
    BINDS,
    CORRELATION,
    NO_CORRELATION,
    NEGATIVE_CORRELATION,
    POSITIVE_CORRELATION,
    ASSOCIATION,
    PART_OF,
)
from pybel.dsl import CentralDogma, ComplexAbundance, Abundance, CompositeAbundance, MicroRna

from stonkgs.constants import (
    DUMMY_EXAMPLE_INDRA,
    MISC_DIR,
    INPUT_DIR,
    SPECIES_DIR,
    ORGAN_DIR,
    CELL_TYPE_DIR,
    CELL_LINE_DIR,
    LOCATION_DIR,
    DISEASE_DIR,
    RELATION_TYPE_DIR,
)

logger = logging.getLogger(__name__)

DIRECT_RELATIONS = {
    DIRECTLY_INCREASES,
    DIRECTLY_DECREASES,
    BINDS,
}

INDIRECT_RELATIONS = {
    REGULATES,
    CORRELATION,
    DECREASES,
    INCREASES,
    NO_CORRELATION,
    NEGATIVE_CORRELATION,
    POSITIVE_CORRELATION,
    ASSOCIATION,
    PART_OF,
}

UP_RELATIONS = {
    INCREASES,
    POSITIVE_CORRELATION,
    DIRECTLY_INCREASES
}

DOWN_RELATIONS = {
    DECREASES,
    NEGATIVE_CORRELATION,
    DIRECTLY_DECREASES
}


def binarize_triple_direction(graph: pybel.BELGraph) -> Dict[str, Any]:
    """Binarize triples depending on the type of direction."""
    triples = []

    summary = {'context': '(in)direct relations'}

    # Iterate through the graph and infer a subgraph
    for u, v, data in graph.edges(data=True):

        if EVIDENCE not in data or not data[EVIDENCE]:
            logger.warning(f'not evidence found in {data}')
            continue

        # todo: check this we will focus only on molecular interactions
        if not any(
            isinstance(u, class_to_check)
            for class_to_check in (CentralDogma, ComplexAbundance, Abundance, CompositeAbundance, MicroRna)
        ):
            continue

        if not any(
            isinstance(v, class_to_check)
            for class_to_check in (CentralDogma, ComplexAbundance, Abundance, CompositeAbundance, MicroRna)
        ):
            continue

        if data[RELATION] in UP_RELATIONS:
            class_label = 'up'
        elif data[RELATION] in DOWN_RELATIONS:
            class_label = 'down'
        # TODO: add more
        elif data[RELATION] == REGULATES:
            class_label = 'regulates'
        else:
            continue

        triples.append({
            'source': u,
            'relation': data[RELATION],
            'target': v,
            'evidence': data[EVIDENCE],
            'pmid': data[CITATION],
            'class': class_label,
        })

    df = pd.DataFrame(triples)

    summary['number_of_triples'] = df.shape[0]
    summary['number_of_labels'] = df['class'].unique().size
    summary['labels'] = df['class'].value_counts().to_dict()

    df.to_csv(os.path.join(RELATION_TYPE_DIR, f'relation_type.tsv'), sep='\t', index=False)

    return summary


def create_polarity_annotations(graph: pybel.BELGraph) -> Dict[str, Any]:
    """Group triples depending on the type of polarity."""
    triples = []

    summary = {'context': 'polarity'}

    # Iterate through the graph and infer a subgraph
    for u, v, data in graph.edges(data=True):

        if EVIDENCE not in data or not data[EVIDENCE]:
            logger.warning(f'not evidence found in {data}')
            continue

        # todo: check this we will focus only on molecular interactions
        if not any(
            isinstance(u, class_to_check)
            for class_to_check in (CentralDogma, ComplexAbundance, Abundance, CompositeAbundance, MicroRna)
        ):
            continue

        if not any(
            isinstance(v, class_to_check)
            for class_to_check in (CentralDogma, ComplexAbundance, Abundance, CompositeAbundance, MicroRna)
        ):
            continue

        class_label = 'indirect' if data[RELATION] in INDIRECT_RELATIONS else 'direct'

        triples.append({
            'source': u,
            'relation': data[RELATION],
            'target': v,
            'evidence': data[EVIDENCE],
            'pmid': data[CITATION],
            'class': class_label,
        })

    df = pd.DataFrame(triples)

    summary['number_of_triples'] = df.shape[0]
    summary['number_of_labels'] = df['class'].unique().size
    summary['labels'] = df['class'].value_counts().to_dict()

    df.to_csv(os.path.join(RELATION_TYPE_DIR, f'relation_type.tsv'), sep='\t', index=False)

    return summary


def create_context_type_specific_subgraph(graph: pybel.BELGraph, context_annotations: List[str]) -> pybel.BELGraph:
    """Create a subgraph based on context annotations."""
    subgraph = graph.child()
    subgraph.name = f'INDRA graph contextualized for {context_annotations}'

    edges_to_remove = []

    # Iterate through the graph and infer a subgraph with edges that contain the annotation of interest
    for u, v, k, data in graph.edges(data=True, keys=True):
        if ANNOTATIONS in data and any(
            annotation in data[ANNOTATIONS]
            for annotation in context_annotations
        ):
            subgraph.add_edge(u, v, k, **data)
            # Triples to be removed
            edges_to_remove.append((u, v, k))

    number_of_edges_before = graph.number_of_edges()
    # graph.remove_edges_from(edges_to_remove)
    # number_of_edges_after_removing_annotations = graph.number_of_edges()

    # logger.info(
    #     f'Original graph was reduced from {number_of_edges_before} to {number_of_edges_after_removing_annotations}
    #     edges'
    # )

    string = f'Number of nodes/edges in the inferred subgraph "{context_annotations}": \
    {subgraph.number_of_nodes()} {subgraph.number_of_edges()}'

    logger.info(string)

    return subgraph


def dump_edgelist(
    graph: pybel.BELGraph,
    annotations: List[str],
    name: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Dump tsv file for ml purposes."""
    triples = []

    summary = {
        'context': name,
    }

    # Iterate through the graph and infer a subgraph with edges that contain the annotation of interest
    for u, v, data in graph.edges(data=True):

        if not data[EVIDENCE]:
            continue

        # Multiple annotations
        for annotation in data[ANNOTATIONS]:

            if annotation not in annotations:
                continue

            # Skip multiple classes in the triple for the same annotation
            if len(data[ANNOTATIONS][annotation]) > 1:
                logger.warning(f'triple has more than one label -> {data[ANNOTATIONS][annotation]}')
                continue

            for label_annotation in data[ANNOTATIONS][annotation]:
                triples.append(
                    {
                        'source': u,
                        'relation': data[RELATION],
                        'target': v,
                        'evidence': data[EVIDENCE],
                        'pmid': data[CITATION],
                        'class': label_annotation,
                    },
                )

    if not triples:
        return {
            'context': name,
            'number_of_triples': '0',
            'number_of_labels': '0',
            'labels': '0',
        }

    df = pd.DataFrame(triples)

    """Removing labels that appear less than X% from the total"""
    label_counts = df['class'].value_counts().to_dict()
    # 0.05%
    percentage = 0.05
    cutoff = int(df.shape[0] * percentage)

    labels_to_remove = {
        label: count
        for label, count in label_counts.items()
        if count < cutoff
    }

    logger.warning(f'labels removed due to low occurrence {labels_to_remove}')

    logger.info(f'raw triples {df.shape[0]}')
    df = df[df['class'].isin(list(labels_to_remove.keys()))]

    logger.info(f'triples after filtering {df.shape[0]}')

    final_labels = df["class"].unique()
    logger.info(f' final number of classes {final_labels.size}')

    summary['number_of_triples'] = df.shape[0]
    summary['number_of_labels'] = final_labels.size
    summary['labels'] = df['class'].value_counts().to_dict()

    df.to_csv(os.path.join(output_dir, f'{name}.tsv'), sep='\t', index=False)

    return summary


def read_indra_triples(
    path: str = DUMMY_EXAMPLE_INDRA,  # TODO: change
):
    """Parse indra statements in JSON and returns context specific graphs."""
    #: Read file INDRA KG
    indra_kg = pybel.io.indra.from_indra_statements_json_file(DUMMY_EXAMPLE_INDRA)

    #: Summarize content of the KG
    logger.info(indra_kg.summarize())

    # Print all the annotations in the graph
    all_annotations = set()

    for _, _, data in indra_kg.edges(data=True):
        if ANNOTATIONS in data:
            for key in data[ANNOTATIONS]:
                all_annotations.add(key)

    # Summarize all annotations
    logger.info(all_annotations)

    """
    Split the KG into two big chunks:

    1. Pre-training dataset -> Any triple + text evidence that is not in any of the 6+1 annotation types above
    The rationale is that we want to have a common pre-trained model that can be used as a basis for all the fine tuning
    classification tasks.

    2. Fine-tuning datasets (benchmark for our models). 7 different train-validation-test splits for each of the 
    annotation types (each of them will contain multiple classes).

    We would like to note that the pre-training dataset is significantly larger than the fine-tuning dataset.
    Naturally, STonKGs requires a pre-training similar to the other two baselines models (i.e., KG-based has been
    trained based on node2vec on the INDRA KG and BioBERT was trained on PubMed).
    """
    organ_subgraph = create_context_type_specific_subgraph(indra_kg, ['MeSHAnatomy'])
    species_subgraph = create_context_type_specific_subgraph(indra_kg, ['TAX_ID'])
    disease_subgraph = create_context_type_specific_subgraph(indra_kg, ['MeSHDisease', 'Disease'])
    cell_type_subgraph = create_context_type_specific_subgraph(indra_kg, ['Cell'])
    cell_line_subgraph = create_context_type_specific_subgraph(indra_kg, ['CellLine'])
    location_subgraph = create_context_type_specific_subgraph(indra_kg, ['CellStructure'])

    #: Dump the 6+1 annotation type specific subgraphs (triples)
    organ_summary = dump_edgelist(
        graph=organ_subgraph,
        annotations=['MeSHAnatomy'],
        name='organ',
        output_dir=ORGAN_DIR,
    )
    species_summary = dump_edgelist(
        graph=species_subgraph,
        annotations=['TAX_ID'],
        name='species',
        output_dir=SPECIES_DIR,
    )
    disease_summary = dump_edgelist(
        graph=disease_subgraph,
        annotations=['MeSHDisease', 'Disease'],
        name='disease',
        output_dir=DISEASE_DIR,
    )
    cell_type_summary = dump_edgelist(
        graph=cell_type_subgraph,
        annotations=['Cell'],
        name='cell_type',
        output_dir=CELL_TYPE_DIR,
    )
    cell_line_summary = dump_edgelist(
        graph=cell_line_subgraph,
        annotations=['CellLine'],
        name='cell_line',
        output_dir=CELL_LINE_DIR,
    )
    location_summary = dump_edgelist(
        graph=location_subgraph,
        annotations=['CellStructure'],
        name='location',
        output_dir=LOCATION_DIR,
    )

    polarity_summary = binarize_triple_direction(indra_kg)
    directionality_summary = binarize_triple_direction(indra_kg)

    summary_df = pd.DataFrame([
        organ_summary,
        species_summary,
        disease_summary,
        cell_type_summary,
        cell_line_summary,
        location_summary,
        directionality_summary,
        polarity_summary,
    ])
    summary_df.to_csv(os.path.join(MISC_DIR, 'summary.tsv'), sep='\t', index=False)

    """Dump pre training dataset."""
    triples = []

    # Iterate through the graph and infer a subgraph with edges that contain the annotation of interest
    for u, v, data in indra_kg.edges(data=True):
        # Skip relations without evidences
        # TODO: skip specific relations
        if not EVIDENCE in data:
            continue

        triples.append({
            'source': u,
            'relation': data[RELATION],
            'target': v,
            'evidence': data[EVIDENCE],
            'pmid': data[CITATION],
        })

    pretraining_triples = pd.DataFrame(triples)
    del triples
    pretraining_triples.to_csv(os.path.join(INPUT_DIR, 'pretraining_triples.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    read_indra_triples()
