# -*- coding: utf-8 -*-

"""Filter out duplicate text evidences (the ones that appear more than two times)."""

import logging
import os
from collections import Counter
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from stonkgs.constants import (
    CELL_LINE_DIR,
    CELL_TYPE_DIR,
    DISEASE_DIR,
    EMBEDDINGS_PATH,
    LOCATION_DIR,
    ORGAN_DIR,
    SPECIES_DIR,
    RELATION_TYPE_DIR,
)
from stonkgs.data.indra_for_pretraining import _prepare_df

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def filter_out_duplicates(
    df: pd.DataFrame,
    name: str = '',
) -> pd.DataFrame:
    """Filter for unique text evidences in the data entries, avoiding repeating evidences."""

    # Record the length before for logging purposes
    len_before = len(df)

    # Filter for unique evidences
    df = df.drop_duplicates(subset='evidence')

    # Record the length after and log it
    len_after = len(df)
    logger.info(f'{name}: {len_before} (before), {len_after} (after), {len_before-len_after} (# removed)')

    return df


def apply_kg_filtering(
    df: pd.DataFrame,
    embedding_name_to_vector_path: str = EMBEDDINGS_PATH,
    name: str = '',
) -> pd.DataFrame:
    """Filters out entries in the fine-tuning dataset that contain nodes which are not part of the pre-trained KG."""
    kg_embed_dict = _prepare_df(embedding_name_to_vector_path)
    original_length = len(df)
    df = df[
        df['source'].isin(kg_embed_dict.keys()) & df['target'].isin(kg_embed_dict.keys())
    ].reset_index(drop=True)
    new_length = len(df)
    logger.info(f'For {name}, {original_length - new_length} out of {original_length} triples are left out because '
                f'they contain nodes which are not present in the pre-training data')
    return df


def reduce_dataset_size(
    df: pd.DataFrame,
    max_dataset_size: int = 10000,
    class_name: str = "class",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Creates a stratified subset of the original dataset if it exceeds max_dataset_size."""
    # Cut the dataset down to max_dataset_size (deterministically!) using StratifiedShuffleSplit if needed:
    # (this is not an actual train/test split, this is just for getting a dataset of size max_dataset_size in a
    # stratified and deterministic manner)
    if len(df) > max_dataset_size:
        if class_name == "class":
            df = train_test_split(
                df,
                train_size=max_dataset_size,
                random_state=random_seed,
                stratify=df[class_name],
            )[0]
        else:
            df = train_test_split(
                df,
                train_size=max_dataset_size * 2,
                random_state=random_seed,
                stratify=df["interaction"],
            )[0]

            df = train_test_split(
                df,
                train_size=max_dataset_size,
                random_state=random_seed,
                stratify=df["polarity"],
            )[0]

            # Just some extra info on the relation type distribution afterwards
            if name == 'relation_type':
                logger.info(f"Polarity: {Counter(df['polarity'])}")
                logger.info(f"Interaction: {Counter(df['interaction'])}")

    return df


def filter_out_special_character_sequences(
    df: pd.DataFrame,
    evidence_col_name: str = "evidence",
    name: str = '',
) -> pd.DataFrame:
    counter = 0
    for idx, row in df.iterrows():
        if any([x in row[evidence_col_name] for x in ["[", "]", "XREF", "\\u"]]):
            counter += 1
            row[evidence_col_name].replace("[", "").replace("]", "").replace(r"\\u", "").replace("XREF", "")
            df.iloc[idx] = row
    logger.info(f'For {name}, {counter} out of {len(df)} many entries contained the specified special characters')

    return df


if __name__ == "__main__":
    all_names = [
        'cell_line',
        'cell_type',
        'disease',
        'location',
        'organ',
        'species',
        'relation_type',
    ]
    all_dirs = [
        CELL_LINE_DIR,
        CELL_TYPE_DIR,
        DISEASE_DIR,
        LOCATION_DIR,
        ORGAN_DIR,
        SPECIES_DIR,
        RELATION_TYPE_DIR,
    ]

    for name, directory in zip(all_names, all_dirs):
        # Load the unfiltered dataframe
        if name == "relation_type":
            task_specific_df = pd.read_csv(os.path.join(directory, name + '.tsv'), sep='\t')
        else:
            task_specific_df = pd.read_csv(os.path.join(directory, name + '_filtered_less_classes.tsv'), sep='\t')

        # 1. Remove all entries with nodes that are not in the KG
        task_specific_df = apply_kg_filtering(task_specific_df, name=name)
        # 2. Remove special characters in the evidences
        task_specific_df = filter_out_special_character_sequences(task_specific_df, name=name)
        # 3. Remove duplicates
        task_specific_df = filter_out_duplicates(task_specific_df, name=name)
        # 4. Reduce dataset size if needed
        if name == "relation_type":
            task_specific_df = reduce_dataset_size(task_specific_df, class_name="interaction")
        else:
            task_specific_df = reduce_dataset_size(task_specific_df, class_name="class")

        # Save the filtered df
        task_specific_df.to_csv(os.path.join(directory, name + '_less_classes_no_duplicates.tsv'), sep='\t', index=None)