# -*- coding: utf-8 -*-

"""Reads and prepares triples/statements for INDRA.

Run with:
python -m src.data.indra
"""

import os
import logging

import pandas as pd
import pybel
from pybel.constants import ANNOTATIONS, EVIDENCE, RELATION, CITATION

from ..constants import DATA_DIR, DUMMY_EXAMPLE_INDRA

logger = logging.getLogger(__name__)


def create_context_subgraph(graph: pybel.BELGraph, context_annotation: str) -> pybel.BELGraph:
    """Create a subgraph based on context annotations."""
    subgraph = graph.child()
    subgraph.name = f'INDRA graph contextualized for {context_annotation}'

    # Iterate through the graph and infer a subgraph with edges that contain the annotation of interest
    for u, v, k, data in graph.edges(data=True, keys=True):
        if ANNOTATIONS in data and context_annotation in data[ANNOTATIONS]:
            subgraph.add_edge(u, v, k, **data)

    string = f'Number of nodes/edges in the inferred subgraph "{context_annotation}": \
    {subgraph.number_of_nodes()} {subgraph.number_of_edges()}'

    logger.info(string)

    return subgraph


def dump_edgelist(graph: pybel.BELGraph, name: str) -> None:
    """Dump tsv file for ml purposes."""
    triples = []

    # Iterate through the graph and infer a subgraph with edges that contain the annotation of interest
    for u, v, data in graph.edges(data=True):

        if not data[EVIDENCE]:
            logger.warning(f'not evidence found in {data}')
            continue

        if len(data[ANNOTATIONS][name]) > 1:
            logger.warning(f'triple has more than one label -> {data[ANNOTATIONS][name]}')
            continue

        for label_annotation in data[ANNOTATIONS][name]:
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

    df = pd.DataFrame(triples)
    df.to_csv(os.path.join(DATA_DIR, f'{name}.tsv'), sep='\t', index=False)


def read_indra_triples(
    path: str = DUMMY_EXAMPLE_INDRA,  # TODO: change
):
    """Parse indra statements in JSON and returns context specific graphs."""
    indra_kg = pybel.io.indra.from_indra_statements_json_file(DUMMY_EXAMPLE_INDRA)

    logger.info(indra_kg.summarize())

    # Print all the annotations in the graph
    all_annotations = set()

    for _, _, data in indra_kg.edges(data=True):
        if ANNOTATIONS in data:
            for key in data[ANNOTATIONS]:
                all_annotations.add(key)

    logger.info(all_annotations)

    organ_subgraph = create_context_subgraph(indra_kg, 'organ')
    species_subgraph = create_context_subgraph(indra_kg, 'species')
    disease_subgraph = create_context_subgraph(indra_kg, 'disease')
    cell_type_subgraph = create_context_subgraph(indra_kg, 'cell_type')
    cell_line_subgraph = create_context_subgraph(indra_kg, 'cell_line')
    location_subgraph = create_context_subgraph(indra_kg, 'location')

    dump_edgelist(organ_subgraph, 'organ')
    dump_edgelist(species_subgraph, 'species')
    dump_edgelist(disease_subgraph, 'disease')
    dump_edgelist(cell_type_subgraph, 'cell_type')
    dump_edgelist(cell_line_subgraph, 'cell_line')
    dump_edgelist(location_subgraph, 'location')


if __name__ == '__main__':
    read_indra_triples()
