# -*- coding: utf-8 -*-

"""Node2vec model."""

import logging
import random
import pickle
import os
from typing import Optional

import numpy as np
import pandas as pd
import optuna
import networkx as nx
from nodevectors import Node2Vec
from stellargraph.data import EdgeSplitter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from ..constants import DUMMY_EXAMPLE_TRIPLES, KG_HPO_DIR

logger = logging.getLogger(__name__)


def run_link_prediction(
    kg: nx.DiGraph,
    model: Node2Vec
) -> float:
    """Link prediction task for a given KG and node2vec model."""
    # Preprocessing step: Generate positive and negative triples
    # 1. Method employed -> https://stellargraph.readthedocs.io/en/latest/_modules/stellargraph/data/edge_splitter.html
    # 2. Alternative from PyKEEN -> https://pykeen.readthedocs.io/en/latest/reference/negative_sampling.html
    # 3. Alternative from xswap -> https://github.com/hetio/xswap

    # follow https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/node2vec-link-prediction.html
    _, indra_kg_triples, edge_labels = EdgeSplitter(kg).train_test_split()

    # 1. generate embeddings
    # 2. use dot product between two entities from negative triple
    # iterate row-wise through the negative samples and return the hadamard product of the embeddings of the
    # two entities from each triple
    triple_embeds = np.asarray([
        model.predict(source) * model.predict(target)
        for source, target in indra_kg_triples
    ])

    # 3. use a LogReg classifier for final classification on test
    # first generate a train test split for the classifier
    embeds_train, embeds_test, labels_train, labels_test = train_test_split(
        triple_embeds,
        edge_labels,
        stratify=edge_labels)

    link_prediction_model = LogisticRegression()
    link_prediction_model.fit(embeds_train, labels_train)
    predictions_test = link_prediction_model.predict(embeds_test)

    # 4. return sklearn roc curve score for this classifier
    return roc_auc_score(labels_test, predictions_test)


# TODO: add parameters/click later on
# @click.group()
def run_node2vec(
    positive_graph_path: str = DUMMY_EXAMPLE_TRIPLES,
    sep: str = '\t',
    seed: Optional[int] = None,
    delete_database: bool = True
):
    """CLI to run node2vec."""
    if seed is None:
        seed = random.randint(1, 2 ** 32 - 1)
        logger.info(f'random seed given, setting to: {seed}')
    else:
        logger.info(f'random seed set given is: {seed}')

    np.random.seed(seed)
    random.seed(seed)

    # Read graph, first read the triples into a dataframe
    triples_df = pd.read_csv(positive_graph_path, sep=sep)
    # TODO check if it is a directed graph
    # Initialize empty Graph and fill it with the triples from the df
    indra_kg_pos = nx.empty_graph()
    for _, row in triples_df[["source", "target"]].iterrows():
        indra_kg_pos.add_edge(row["source"], row["target"])

    # indra_kg_pos = nx.read_edgelist(positive_graph_path, delimiter=sep)

    # Hyperparameters
    # see https://github.com/seffnet/seffnet/blob/master/src/seffnet/optimization.py
    negative: int = 5
    iterations: int = 15
    batch_words: int = 1000
    walk_length: int = 128
    # has to be the same as the embedding dimension of the NLP model
    # TODO double check later on
    dimensions: int = 768

    # define HPO function for optuna
    def objective(
        trial: optuna.trial.Trial
    ) -> float:
        """Runs HPO on the link prediction task on the KG, based on a LogReg classifier and the auc score."""
        epochs = trial.suggest_categorical('epochs', [8, 16, 32, 64, 128, 256])
        window_size = trial.suggest_int('window_size', 3, 7)
        # TODO: check best q/p values
        p = trial.suggest_uniform('p', 0, 4.0)
        q = trial.suggest_uniform('q', 0, 4.0)

        # train the KGE model
        node2vec_model = Node2Vec(
            n_components=dimensions,
            walklen=walk_length,
            epochs=epochs,
            # use inverse, see https://github.com/VHRanger/nodevectors/blob/master/nodevectors/node2vec.py#L46
            return_weight=1 / p,
            neighbor_weight=1 / q,
            threads=0,
            keep_walks=True,
            verbose=True,
            w2vparams={
                'window': window_size,
                'negative': negative,
                'iter': iterations,
                'batch_words': batch_words,
            },
        )

        node2vec_model.fit(indra_kg_pos)

        # Save a trained model to a file
        with open(os.path.join(KG_HPO_DIR, "node2vec_model_{}.pickle".format(trial.number)), "wb") as fout:
            pickle.dump(node2vec_model, fout)

        # return the auc score for a LogReg classifier on the link prediction task with negative samples
        return run_link_prediction(
            kg=indra_kg_pos,
            model=node2vec_model
        )

    # create study and set number of trials
    n_trials = 2
    study = optuna.create_study(
        study_name="Node2vec HPO on INDRA KG",
        storage=f"sqlite:///{KG_HPO_DIR}/kge_indra_hpo.db",
        direction='maximize',
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=n_trials
    )

    # Remove all the models except for the best one :)
    for filename in os.listdir(KG_HPO_DIR):
        if filename != "node2vec_model_{}.pickle".format(study.best_trial.number):
            os.remove(os.path.join(KG_HPO_DIR, filename))

    # Load the best model
    with open(os.path.join(KG_HPO_DIR, "node2vec_model_{}.pickle".format(study.best_trial.number)), "rb") as fin:
        best_clf: Node2Vec = pickle.load(fin)

    # Delete the HPO database by default
    if delete_database:
        optuna.delete_study(study_name="Node2vec HPO on INDRA KG", storage=f"sqlite:///{KG_HPO_DIR}/kge_indra_hpo.db")

    """Save the embeddings"""
    wv = best_clf.model.wv
    sorted_vocab_items = sorted(wv.vocab.items(), key=lambda item: item[1].count, reverse=True)
    vectors = wv.vectors

    with open(os.path.join(KG_HPO_DIR, "embeddings_best_model.tsv"), "wb") as emb_file:
        for word, vocab_ in sorted_vocab_items:
            # Write to vectors file
            print(word, *(repr(val) for val in vectors[vocab_.index]), sep='\t', file=emb_file)

    randomwalks = best_clf.walks

    print(randomwalks)
    print(type(randomwalks))

    with open(os.path.join(KG_HPO_DIR, "random_walks_best_model.tsv", "wb")) as emb_file:
        for word, vocab_ in sorted_vocab_items:
            # Write to vectors file
            print(word, *(repr(val) for val in vectors[vocab_.index]), sep='\t', file=emb_file)


if __name__ == "__main__":
    run_node2vec()
