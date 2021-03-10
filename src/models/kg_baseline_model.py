# -*- coding: utf-8 -*-

"""KG baseline model on the fine-tuning classification task, assuming the model embeddings are pre-trained."""

import os
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _prepare_df(embedding_path: str, sep: str = '\t') -> Dict[str, List[str]]:
    """Prepare dataframe to node->embeddings/random walks."""
    # Load embeddings
    df = pd.read_csv(
        embedding_path,
        sep=sep,
        header=None,
        index_col=0,
    )
    # node id -> embeddings
    return {
        index: row.values
        for index, row in df.iterrows()
    }


def get_train_test_splits(
    data: pd.DataFrame,
    label_column_name: str = "class",
    random_seed: int = 42,
    n_splits: int = 5
) -> List:
    """Returns deterministic train/test indices for n_splits based on the fine-tuning dataset that is passed."""
    # Leave out the label in the dataset
    data_no_labels = data.drop(label_column_name, axis=1)
    labels = data[label_column_name]

    # For now: implement stratified train/test splits with no validation split (since there's no HPO)
    # It is shuffled deterministically (determined by random_seed)
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=True)

    return [{"train_idx": train_idx, "test_idx": test_idx} for train_idx, test_idx in skf.split(data_no_labels, labels)]


def run_kg_baseline_classification_cv(triples_path, embedding_path, random_walks_path):
    """Run KG baseline classification."""

    # Step 1. load the tsv file with the annotation types you want to test and make the splits
    triples_df = pd.read_csv(
        triples_path,
        sep='\t',
        usecols=[
            'source',
            # 'relation',
            'target',
            # 'evidence',
            # 'pmid',
            'class'
        ],
    )

    train_test_splits = get_train_test_splits(triples_df)

    embeddings_dict = _prepare_df(embedding_path)
    random_walks_dict = _prepare_df(random_walks_path)

    print(embeddings_dict)
    print(random_walks_dict)

    # Number of embeddings and random walks
    print(len(random_walks_dict[list(random_walks_dict.keys())[0]]))
    print(len(embeddings_dict[list(embeddings_dict.keys())[0]]))

    # TODO: 3. Initialize the model pooling classification (make a class)


if __name__ == "__main__":
    run_kg_baseline_classification_cv(
        triples_path='',  # fixme
        embedding_path='',
        random_walks_path='',
    )
