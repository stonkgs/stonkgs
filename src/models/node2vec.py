# -*- coding: utf-8 -*-

"""Node2vec model."""

import logging
import random
import numpy as np
import click
from nodevectors import Node2Vec
import networkx as nx
from typing import Optional

logger = logging.getLogger(__name__)


@click.group()
# TODO: add parameters
def run_node2vec(
    positive_graph_path: str,
    sep: str = '\t',
    seed: Optional[int] = None,
):
    """CLI to run node2vec."""
    if seed is None:
        seed = random.randint(1, 2 ** 32 - 1)
        logger.info(f'random seed given, setting to: {seed}')
    else:
        logger.info(f'random seed set given is: {seed}')

    np.random.seed(seed)
    random.seed(seed)

    # Read graphs
    #: TODO check if it is a directed graph
    indra_kg_pos = nx.read_edgelist(positive_graph_path, sep=sep)

    # TODO: get negative sampling from https://pykeen.readthedocs.io/en/latest/reference/negative_sampling.html
    # TODO: or xswap from himmelstein (https://github.com/drug2ways/drug2ways/blob/master/src/drug2ways/permute.py)
    indra_kg_neg = ...

    # Parameters
    # TODO: what are the best? look at other papers / make grid search
    # see https://github.com/seffnet/seffnet/blob/master/src/seffnet/optimization.py
    n_components: int = 50
    walklen: int = 10
    epochs: int = 20
    return_weight: float = 2.0
    neighbor_weight: float = 2.0
    window: int = 8
    negative: int = 5
    iterations: int = 15
    batch_words: int = 1000

    # Fit embedding model to graph
    g2v = Node2Vec(
        n_components=n_components,
        walklen=walklen,
        epochs=epochs,
        return_weight=return_weight,
        neighbor_weight=neighbor_weight,
        threads=0,
        keep_walks=True,
        verbose=True,
        w2vparams={
            'window': window,
            'negative': negative,
            'iter': iterations,
            'batch_words': batch_words,
        },
    )

    # Fit
    g2v.fit(indra_kg)

    # TODO: predictions
    g2v.predict(42)

    # Save and load whole node2vec model
    g2v.save('node2vec')

    # Save model to gensim.KeyedVector format
    g2v.save_vectors("wheel_model.bin")

    # TODO: save walks

