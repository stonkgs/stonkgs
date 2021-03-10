# -*- coding: utf-8 -*-

"""KG baseline model on the fine-tuning classification task, assuming the model embeddings are pre-trained."""

import os
from typing import List

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def get_train_test_splits(
    train_data: pd.DataFrame,
    type_column_name: str = "class",
    random_seed: int = 42,
    n_splits: int = 5
) -> List:
    """ Returns train/test indices for n_splits many splits based on the fine-tuning dataset that is passed"""

    # Leave out the label in the dataset
    X = train_data.drop(type_column_name, axis=1)
    y = train_data[type_column_name]

    # TODO: think about whether a validation split is necessary
    # For now: implement stratified train/test splits
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=False)

    return [[train_idx, test_idx] for train_idx, test_idx in skf.split(X, y)]

    # TODO: 1. load the tsv file with the annotation types you want to test and make the splits
    # TODO: 2. load the embeddings from node2vec and random walks
    # TODO: 3. Initialize the model pooling classification (make a class)


def run_kg_baseline_classification_cv(embedding_path, random_walks_path):
    """"""


if __name__ == "__main__":
    run_kg_baseline_classification_cv(
        embedding_path='',
        random_walks_path='',
    )
