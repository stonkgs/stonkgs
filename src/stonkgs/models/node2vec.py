# -*- coding: utf-8 -*-

"""Node2vec model.

Run with: ``python -m src.stonkgs.models.node2vec``
"""

import logging
import os
import pickle
import random
from typing import Optional, Union

import csrgraph as cg
import networkx as nx
import numpy as np
import optuna
import pandas as pd
from nodevectors import Node2Vec
from optuna.integration import MLflowCallback
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from stellargraph.data import EdgeSplitter
from tqdm import tqdm

from stonkgs.constants import KG_HPO_DIR, MLFLOW_TRACKING_URI, MODELS_DIR, PRETRAINING_PROT_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_link_prediction(
    kg: Union[nx.DiGraph, cg.csrgraph],
    model: Node2Vec,
) -> float:
    """Link prediction task for a given KG and node2vec model."""
    # Preprocessing step: Generate positive and negative triples
    # 1. Method employed -> https://stellargraph.readthedocs.io/en/latest/_modules/stellargraph/data/edge_splitter.html
    # 2. Alternative from PyKEEN -> https://pykeen.readthedocs.io/en/latest/reference/negative_sampling.html
    # 3. Alternative from xswap -> https://github.com/hetio/xswap

    if isinstance(kg, cg.csrgraph):
        # The sparse matrix needs to be converted first
        kg_triples = kg.mat  # noqa: F841
        raise NotImplementedError("negative sampling has not yet been implemented")
    else:
        # follow https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/node2vec-link-prediction.html
        _, indra_kg_triples, edge_labels = EdgeSplitter(kg).train_test_split()

    # 1. generate embeddings
    # 2. use dot product between two entities from negative triple
    # iterate row-wise through the negative samples and return the hadamard product of the embeddings of the
    # two entities from each triple
    triple_embeds = np.asarray(
        [model.predict(source) * model.predict(target) for source, target in indra_kg_triples]
    )

    # 3. use a LogReg classifier for final classification on test
    # first generate a train test split for the classifier
    embeds_train, embeds_test, labels_train, labels_test = train_test_split(
        triple_embeds, edge_labels, stratify=edge_labels
    )

    link_prediction_model = LogisticRegression()
    link_prediction_model.fit(embeds_train, labels_train)
    predictions_test = link_prediction_model.predict(embeds_test)

    # 4. return sklearn roc curve score for this classifier
    return roc_auc_score(labels_test, predictions_test)


# TODO: add parameters/click later on
# @click.group()
def run_node2vec_hpo(
    positive_graph_path: Optional[str] = PRETRAINING_PROT_PATH,
    sep: Optional[str] = "\t",
    delete_database: Optional[bool] = True,
    mlflow_tracking_uri: Optional[str] = MLFLOW_TRACKING_URI,
    n_optimization_trials: Optional[int] = 1,  # TODO change later to 20
    n_threads: Optional[int] = 96,  # hard coded to the cluster, change if necessary
    seed: Optional[int] = None,
):
    """CLI to run node2vec."""
    if seed is None:
        seed = random.randint(1, 2 ** 32 - 1)  # noqa: S311
        logger.info(f"random seed given, setting to: {seed}")
    else:
        logger.info(f"random seed set given is: {seed}")

    np.random.seed(seed)
    random.seed(seed)

    # Read graph, first read the triples into a dataframe
    triples_df = pd.read_csv(positive_graph_path, sep=sep)
    # Initialize empty Graph and fill it with the triples from the df
    indra_kg_pos = nx.DiGraph()
    for _, row in tqdm(triples_df[["source", "target"]].iterrows(), total=triples_df.shape[0]):
        # FIXME add double relation for some cases
        indra_kg_pos.add_edge(row["source"], row["target"])

    logger.info("Finished loading the KG")

    # indra_kg_pos = nx.read_edgelist(positive_graph_path, delimiter=sep)

    # Hyperparameters
    # see https://github.com/seffnet/seffnet/blob/master/src/seffnet/optimization.py
    negative: int = 5
    # only use 1 iteration for word2vec, see
    # https://datascience.stackexchange.com/questions/9819/number-of-epochs-in-gensim-word2vec-implementation
    iterations: int = 1
    batch_words: int = 1000  # batch words default value in Gensim is 10,000
    walk_length: int = 127
    # has to be the same as the embedding dimension of the NLP model
    dimensions: int = 768
    # TODO: Discuss if one should use these or just set them to 1, see
    #  "embedding a large graph" https://github.com/VHRanger/nodevectors (replace with inverse if != 1.0)
    p = 1.0
    q = 1.0

    # define HPO function for optuna
    def objective(
        trial: optuna.trial.Trial,
    ) -> float:
        """Run HPO on the link prediction task on the KG, based on a LogReg classifier and the auc score."""
        # Leave out large values for epochs due to runtime constraints
        epochs = trial.suggest_categorical("epochs", [2, 4, 8])  # 8, 16])  # 32, 64, 128, 256])
        # Same for window size
        window_size = trial.suggest_int("window_size", 3, 5)  # 7)

        # train the KGE model
        node2vec_model = Node2Vec(
            n_components=dimensions,
            walklen=walk_length,
            epochs=epochs,
            return_weight=p,
            neighbor_weight=q,
            threads=n_threads,
            keep_walks=True,
            verbose=True,
            w2vparams={
                "window": window_size,
                "negative": negative,
                "iter": iterations,
                "batch_words": batch_words,
            },
        )

        node2vec_model.fit(indra_kg_pos)

        # Save a trained model to a file
        with open(
            os.path.join(KG_HPO_DIR, "node2vec_model_{}.pickle".format(trial.number)), "wb"
        ) as fout:
            pickle.dump(node2vec_model, fout)

        # return the auc score for a LogReg classifier on the link prediction task with negative samples
        return run_link_prediction(
            kg=indra_kg_pos,
            model=node2vec_model,
        )

    # Add mlflow callback to log the HPO in mlflow
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow_tracking_uri,
        metric_name="ROC AUC Score",
    )

    logger.info("Created the MLflow Callback")

    # Create study and run it
    study = optuna.create_study(
        study_name="Node2vec HPO on INDRA KG",
        storage=f"sqlite:///{MODELS_DIR}/kge_indra_hpo.db",
        direction="maximize",
        load_if_exists=True,
    )

    logger.info("Created the study")

    study.optimize(
        objective,
        n_trials=n_optimization_trials,
        callbacks=[mlflow_callback],
    )

    # Remove all the models except for the best one :)
    for filename in os.listdir(KG_HPO_DIR):
        if filename != "node2vec_model_{}.pickle".format(study.best_trial.number):
            os.remove(os.path.join(KG_HPO_DIR, filename))

    # Load the best model
    with open(
        os.path.join(KG_HPO_DIR, "node2vec_model_{}.pickle".format(study.best_trial.number)), "rb"
    ) as fin:
        best_clf: Node2Vec = pickle.load(fin)

    # Delete the HPO database by default
    if delete_database:
        optuna.delete_study(
            study_name="Node2vec HPO on INDRA KG",
            storage=f"sqlite:///{MODELS_DIR}/kge_indra_hpo.db",
        )

    """Save the embeddings"""
    wv = best_clf.model.wv
    sorted_vocab_items = sorted(wv.vocab.items(), key=lambda item: item[1].count, reverse=True)
    vectors = wv.vectors

    with open(os.path.join(KG_HPO_DIR, "embeddings_best_model.tsv"), "w") as emb_file:
        for word, vocab_ in sorted_vocab_items:
            # Write to vectors file
            embeddings = "\t".join(repr(val) for val in vectors[vocab_.index])
            emb_file.write(f"{word}\t{embeddings}\n")

    """Save the random walks"""
    all_random_walks = best_clf.walks

    with open(os.path.join(KG_HPO_DIR, "random_walks_best_model.tsv"), "w") as random_walk_file:
        for node, random_walks in zip(wv.index2entity, all_random_walks):
            random_walks_str = "\t".join(random_walks)
            random_walk_file.write(f"{node}\t{random_walks_str}\n")


def run_node2vec(
    positive_graph_path: Optional[str] = PRETRAINING_PROT_PATH,
    sep: Optional[str] = "\t",
    n_threads: Optional[int] = 96,  # hard coded to the cluster, change if necessary
):
    """Run node2vec with no HPO."""
    # Double check the number of expected embeddings
    triples_df = pd.read_csv(positive_graph_path, sep=sep)
    n_expected_nodes = len(set(triples_df["source"]).union(set(triples_df["target"])))
    logger.info(f"{n_expected_nodes} node embeddings are expected")

    # Use CSRGraph for speedup
    indra_kg_pos = cg.read_edgelist(
        positive_graph_path,
        directed=False,
        sep=sep,
        usecols=["source", "target"],
        header=0,
    )
    logger.info("Finished loading the KG")

    # Hyperparameters
    # see https://github.com/seffnet/seffnet/blob/master/src/seffnet/optimization.py
    negative: int = 5
    # only use 1 iteration for word2vec, see
    # https://datascience.stackexchange.com/questions/9819/number-of-epochs-in-gensim-word2vec-implementation
    iterations: int = 1
    batch_words: int = 10000  # batch words default value in Gensim is 10,000
    walk_length: int = 127
    # has to be the same as the embedding dimension of the NLP model
    dimensions: int = 768
    epochs = 4
    window_size = 3

    # train the KGE model
    node2vec_model = Node2Vec(
        n_components=dimensions,
        walklen=walk_length,
        epochs=epochs,
        # use inverse, see https://github.com/VHRanger/nodevectors/blob/master/nodevectors/node2vec.py#L46
        return_weight=1.0,
        neighbor_weight=1.0,
        threads=n_threads,
        keep_walks=True,
        verbose=True,
        w2vparams={
            "window": window_size,
            "negative": negative,
            "iter": iterations,
            "batch_words": batch_words,
            "min_count": 1,  # do not sort out any words (KG entities)!
        },
    )
    logger.info("Successfully created the model")

    node2vec_model.fit(indra_kg_pos)
    logger.info("Successfully trained the model")

    # Save the trained model to a file
    with open(os.path.join(KG_HPO_DIR, "node2vec_model_prot_no_hpo.pickle"), "wb") as fout:
        pickle.dump(node2vec_model, fout)

    # Save the embeddings
    wv = node2vec_model.model.wv
    sorted_vocab_items = sorted(wv.vocab.items(), key=lambda item: item[1].count, reverse=True)
    logger.info(f"{len(sorted_vocab_items)} embeddings were learned")
    vectors = wv.vectors

    with open(os.path.join(KG_HPO_DIR, "embeddings_prot_best_model.tsv"), "w") as emb_file:
        for word, vocab_ in sorted_vocab_items:
            # Write to vectors file
            embeddings = "\t".join(repr(val) for val in vectors[vocab_.index])
            emb_file.write(f"{word}\t{embeddings}\n")

    logger.info("Successfully saved the embeddings")

    # Save the random walks
    all_random_walks = node2vec_model.walks
    logger.info(f"{len(all_random_walks)} random walks were learned")

    with open(
        os.path.join(KG_HPO_DIR, "random_walks_prot_best_model.tsv"), "w"
    ) as random_walk_file:
        for node, random_walks in zip(wv.index2entity, all_random_walks):
            random_walks_str = "\t".join(random_walks)
            random_walk_file.write(f"{node}\t{random_walks_str}\n")

    logger.info("Successfully saved the walks")


if __name__ == "__main__":
    run_node2vec()
