# -*- coding: utf-8 -*-

"""Filter the fine-tuning datasets by their majority classes."""

import logging
import os
from typing import Optional

import pandas as pd

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
    # Remove the "-1" label (str) (since it can't be mapped to any ontology)
    df = df[df['class'] != '-1']

    # Manually merge EFO:0000887 into UBERON:0002107, since it's deprecated
    df.replace('0000887', '0002107', inplace=True)

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
        df.to_csv(os.path.join(output_path, name + '_filtered_more_classes.tsv'), sep='\t', index=None)

    return df


if __name__ == "__main__":
    # Load all the unfiltered dataframes
    cell_line_df = pd.read_csv(os.path.join(CELL_LINE_DIR, 'cell_line.tsv'), sep='\t')
    cell_type_df = pd.read_csv(os.path.join(CELL_TYPE_DIR, 'cell_type.tsv'), sep='\t')
    disease_df = pd.read_csv(os.path.join(DISEASE_DIR, 'disease.tsv'), sep='\t')
    location_df = pd.read_csv(os.path.join(LOCATION_DIR, 'location.tsv'), sep='\t')
    organ_df = pd.read_csv(os.path.join(ORGAN_DIR, 'organ.tsv'), sep='\t')
    species_df = pd.read_csv(os.path.join(SPECIES_DIR, 'species.tsv'), sep='\t')

    # See how many entries there are for each filtered dataframe
    filter_for_majority_classes(cell_line_df, name='cell_line', n_classes=20, output_path=CELL_LINE_DIR)
    filter_for_majority_classes(cell_type_df, name='cell_type', n_classes=20, output_path=CELL_TYPE_DIR)
    filter_for_majority_classes(disease_df, name='disease', n_classes=20, output_path=DISEASE_DIR)
    filter_for_majority_classes(location_df, name='location', n_classes=5, output_path=LOCATION_DIR)
    filter_for_majority_classes(organ_df, name='organ', n_classes=20, output_path=ORGAN_DIR)
    filter_for_majority_classes(species_df, name='species', n_classes=3, output_path=SPECIES_DIR)