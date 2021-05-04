# -*- coding: utf-8 -*-

"""STonKGs model on the fine-tuning classification task, assuming the model embeddings are pre-trained."""

import logging
from typing import List

import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def filter_for_majority_classes(
    df: pd.DataFrame,
    n_classes: int = 5,
) -> pd.DataFrame:
    """Filter out data entries that occur infrequently, return the dataframe with only n_class many majority classes."""
    # Remove any classes that are not in the n_classes most common classes
    # (value counts returns classes in descending order)
    labels_to_remove = df['class'].value_counts()[n_classes:].to_dict()

    # Some statistics
    logger.info(f'labels removed due to low occurrence {labels_to_remove}')
    logger.info(f'raw triples {df.shape[0]}')

    # Remove all the entries that are labelled with a class that should be removed
    df = df[~df['class'].isin(list(labels_to_remove.keys()))]

    # Final length of the filtered df
    logger.info(f'triples after filtering {df.shape[0]}')

    return df


def get_train_test_splits(
    train_data: pd.DataFrame,
    type_column_name: str = "class",
    random_seed: int = 42,
    n_splits: int = 5,
) -> List:
    """Return train/test indices for n_splits many splits based on the fine-tuning dataset that is passed."""
    # Leave out the label in the dataset
    X = train_data.drop(type_column_name, axis=1)  # noqa: N806
    y = train_data[type_column_name]

    # TODO: think about whether a validation split is necessary
    # For now: implement stratified train/test splits
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=False)

    return [[train_idx, test_idx] for train_idx, test_idx in skf.split(X, y)]
