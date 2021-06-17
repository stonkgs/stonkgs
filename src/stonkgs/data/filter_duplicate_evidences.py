# -*- coding: utf-8 -*-

"""Filter out duplicate text evidences (the ones that appear more than two times)."""

import logging
import os
from collections import Counter
from typing import Optional

import pandas as pd

from stonkgs.constants import (
    CELL_LINE_DIR,
    CELL_TYPE_DIR,
    DISEASE_DIR,
    LOCATION_DIR,
    ORGAN_DIR,
    SPECIES_DIR,
    RELATION_TYPE_DIR,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def filter_out_duplicates(
    df: pd.DataFrame,
    name: str = '',
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Filter for unique text evidences in the data entries, avoiding repeating evidences."""

    # Record the length before for logging purposes
    len_before = len(df)

    # Filter for unique evidences
    df = df.drop_duplicates(subset='evidence')

    # Record the length after and log it
    len_after = len(df)
    logger.info(f'{name}: {len_before} (before), {len_after} (after), {len_before-len_after} (# removed)')

    # Just some extra info on the relation type distribution afterwards
    if name == 'relation_type':
        logger.info(f"Polarity: {Counter(df['polarity'])}")
        logger.info(f"Interaction: {Counter(df['interaction'])}")

    # Optional: Save the filtered df
    if output_path and len(name) > 0:
        df.to_csv(os.path.join(output_path, name + '_no_duplicates.tsv'), sep='\t', index=None)

    return df


if __name__ == "__main__":
    # Load all the unfiltered dataframes
    cell_line_df = pd.read_csv(os.path.join(CELL_LINE_DIR, 'cell_line_filtered.tsv'), sep='\t')
    cell_type_df = pd.read_csv(os.path.join(CELL_TYPE_DIR, 'cell_type_filtered.tsv'), sep='\t')
    disease_df = pd.read_csv(os.path.join(DISEASE_DIR, 'disease_filtered.tsv'), sep='\t')
    location_df = pd.read_csv(os.path.join(LOCATION_DIR, 'location_filtered.tsv'), sep='\t')
    organ_df = pd.read_csv(os.path.join(ORGAN_DIR, 'organ_filtered.tsv'), sep='\t')
    species_df = pd.read_csv(os.path.join(SPECIES_DIR, 'species_filtered.tsv'), sep='\t')
    relation_df = pd.read_csv(os.path.join(RELATION_TYPE_DIR, 'relation_type.tsv'), sep='\t')

    # See how many entries there are for each filtered dataframe
    # Empirically determined number of classes for now
    filter_out_duplicates(cell_line_df, name='cell_line', output_path=CELL_LINE_DIR)
    filter_out_duplicates(cell_type_df, name='cell_type', output_path=CELL_TYPE_DIR)
    filter_out_duplicates(disease_df, name='disease', output_path=DISEASE_DIR)
    filter_out_duplicates(location_df, name='location', output_path=LOCATION_DIR)
    filter_out_duplicates(organ_df, name='organ', output_path=ORGAN_DIR)
    filter_out_duplicates(species_df, name='species', output_path=SPECIES_DIR)
    filter_out_duplicates(relation_df, name='relation_type', output_path=RELATION_TYPE_DIR)