# -*- coding: utf-8 -*-

"""STonKGs model on the fine-tuning classification task, assuming the model embeddings are pre-trained."""

import logging
import os
from typing import List, Optional

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from stonkgs.constants import (
    CELL_LINE_DIR,
    CELL_TYPE_DIR,
    DISEASE_DIR,
    LOCATION_DIR,
    ORGAN_DIR,
    SPECIES_DIR,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def filter_for_majority_classes(
    df: pd.DataFrame,
    n_classes: int = 5,
    name: str = '',
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Filter out data entries that occur infrequently, return the dataframe with only n_class many majority classes."""
    # Remove any classes that are not in the n_classes most common classes
    # (value counts returns classes in descending order)
    labels_to_remove = df['class'].value_counts()[n_classes:].to_dict()
    labels_to_keep = df['class'].value_counts()[:n_classes].to_dict()

    # Some statistics
    # logger.info(f'labels removed due to low occurrence {labels_to_remove}')
    logger.info(f'{name} majority class occurrences {labels_to_keep}')
    logger.info(f'{name} raw triples {df.shape[0]}')

    # Remove all the entries that are labelled with a class that should be removed
    df = df[~df['class'].isin(list(labels_to_remove.keys()))]

    # Final length of the filtered df
    logger.info(f'{name} triples after filtering for {n_classes} classes: {df.shape[0]} \n')

    # Optional: Save the filtered df
    if output_path and len(name) > 0:
        df.to_csv(os.path.join(output_path, name + '_filtered.tsv'), sep='\t')

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


if __name__ == "__main__":
    # Load all the unfiltered dataframes
    cell_line_df = pd.read_csv(os.path.join(CELL_LINE_DIR, 'cell_line.tsv'), sep='\t')
    cell_type_df = pd.read_csv(os.path.join(CELL_TYPE_DIR, 'cell_type.tsv'), sep='\t')
    disease_df = pd.read_csv(os.path.join(DISEASE_DIR, 'disease.tsv'), sep='\t')
    location_df = pd.read_csv(os.path.join(LOCATION_DIR, 'location.tsv'), sep='\t')
    organ_df = pd.read_csv(os.path.join(ORGAN_DIR, 'organ.tsv'), sep='\t')
    species_df = pd.read_csv(os.path.join(SPECIES_DIR, 'species.tsv'), sep='\t')

    # See how many entries there are for each filtered dataframe
    # Empirically determined number of classes for now
    filter_for_majority_classes(cell_line_df, name='cell_line', n_classes=10, output_path=CELL_LINE_DIR)
    filter_for_majority_classes(cell_type_df, name='cell_type', n_classes=10, output_path=CELL_TYPE_DIR)
    filter_for_majority_classes(disease_df, name='disease', n_classes=10, output_path=DISEASE_DIR)
    filter_for_majority_classes(location_df, name='location', n_classes=5, output_path=LOCATION_DIR)
    filter_for_majority_classes(organ_df, name='organ', n_classes=10, output_path=ORGAN_DIR)
    filter_for_majority_classes(species_df, name='species', n_classes=3, output_path=SPECIES_DIR)
