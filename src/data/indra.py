# -*- coding: utf-8 -*-

"""Reads and prepares triples/statements for INDRA.

Run with:
python -m src.data.indra
"""

import logging

import pybel
from pybel.constants import ANNOTATIONS
from ..constants import DUMMY_EXAMPLE_INDRA

logger = logging.getLogger(__name__)


def read_indra_triples(
    path: str = DUMMY_EXAMPLE_INDRA,  # TODO: change
):
    """Parse indra statements in JSON and returns context specific graphs."""
    indra_kg = pybel.io.indra.from_indra_statements_json_file(DUMMY_EXAMPLE_INDRA)

    logger.info(indra_kg.summarize())

    context_annotations = set()

    for _, _, data in indra_kg.edges(data=True):
        if ANNOTATIONS in data:
            for key in data[ANNOTATIONS]:
                context_annotations.add(key)

    #TODO: "context" annotation is missing
    logger.info(context_annotations)
    
    
    # TODO: get a subgraph for each of the four annnotations of interest
    
    # TODO: summarize each of the subgraphs
    
    # TODO: export them in edgelist with a fourth column with the label/class (e.g., human, rat...)


if __name__ == '__main__':
    read_indra_triples()
