# -*- coding: utf-8 -*-

"""Reads and prepares triples/statements for INDRA.

Run with:
python -m src.data.indra
"""

import logging
import os
from typing import List

import pandas as pd
import pybel
from pybel.constants import ANNOTATIONS, EVIDENCE, RELATION, CITATION

from ..constants import DATA_DIR, DUMMY_EXAMPLE_INDRA

logger = logging.getLogger(__name__)


def create_context_subgraph(graph: pybel.BELGraph, context_annotations: List[str]) -> pybel.BELGraph:
    """Create a subgraph based on context annotations."""
    subgraph = graph.child()
    subgraph.name = f'INDRA graph contextualized for {context_annotations}'

    # Iterate through the graph and infer a subgraph with edges that contain the annotation of interest
    for u, v, k, data in graph.edges(data=True, keys=True):
        if ANNOTATIONS in data and any(
            annotation in data[ANNOTATIONS]
            for annotation in context_annotations
        ):
            subgraph.add_edge(u, v, k, **data)

    string = f'Number of nodes/edges in the inferred subgraph "{context_annotations}": \
    {subgraph.number_of_nodes()} {subgraph.number_of_edges()}'

    logger.info(string)

    return subgraph


def dump_edgelist(graph: pybel.BELGraph, annotations: List[str], name: str) -> None:
    """Dump tsv file for ml purposes."""
    triples = []

    # Iterate through the graph and infer a subgraph with edges that contain the annotation of interest
    for u, v, data in graph.edges(data=True):

        if not data[EVIDENCE]:
            logger.warning(f'not evidence found in {data}')
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

    df = pd.DataFrame(triples)

    logger.info(df['class'].value_counts())

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

    organ_subgraph = create_context_subgraph(indra_kg, ['MeSHAnatomy'])
    species_subgraph = create_context_subgraph(indra_kg, ['TAX_ID'])
    disease_subgraph = create_context_subgraph(indra_kg, ['MeSHDisease', 'Disease'])
    cell_type_subgraph = create_context_subgraph(indra_kg, ['Cell'])
    cell_line_subgraph = create_context_subgraph(indra_kg, ['CellLine'])
    location_subgraph = create_context_subgraph(indra_kg, ['CellStructure'])

    dump_edgelist(organ_subgraph, ['MeSHAnatomy'], 'organ')
    dump_edgelist(species_subgraph, ['TAX_ID'], 'species')
    dump_edgelist(disease_subgraph, ['MeSHDisease', 'Disease'], 'disease')
    dump_edgelist(cell_type_subgraph, ['Cell'], 'cell_type')
    dump_edgelist(cell_line_subgraph, ['CellLine'], 'cell_line')
    dump_edgelist(location_subgraph, ['CellStructure'], 'location')


if __name__ == '__main__':
    read_indra_triples()
