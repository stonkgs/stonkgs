# -*- coding: utf-8 -*-

"""Add the protein sequences to text-triple pairs that have descriptions."""

import logging
import os

import pandas as pd
from protmapper.uniprot_client import get_id_from_entrez, get_sequence
from tqdm import tqdm

from stonkgs.constants import (
    CELL_LINE_DIR,
    CELL_TYPE_DIR,
    CORRECT_DIR,
    DISEASE_DIR,
    LOCATION_DIR,
    ORGAN_DIR,
    SPECIES_DIR,
    RELATION_TYPE_DIR,
)

# Initialize the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def add_protein_sequences_per_task(
    input_file: str,
):
    """Add the protein sequences to an existing dataframe consisting of the text-triple pairs with descriptions."""
    # Read the input file and create the resulting df
    input_df = pd.read_csv(input_file, sep="\t", index_col=None)
    result_df = pd.DataFrame(columns=list(input_df.columns) + ["source_prot", "target_prot"])

    # Iterate through all rows of the input file/df
    for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):
        # Try to get the Uniprot IDs for getting the protein sequences from the Entrez IDs
        source_id = get_id_from_entrez(str(row["source_id"]))
        target_id = get_id_from_entrez(str(row["target_id"]))

        # Try to get the protein sequences if there are Uniprot IDs for both nodes in a text-triple pair
        if source_id is not None and target_id is not None:
            # Reduce possible multiple matches to the first one
            source_id = source_id.split(",")[0]
            target_id = target_id.split(",")[0]

            # See if there are protein sequences
            source_prot = get_sequence(source_id)
            target_prot = get_sequence(target_id)

            # If so: Add them to the dataframe
            if source_prot is not None and target_prot is not None:
                row["source_prot"] = source_prot
                row["target_prot"] = target_prot
                result_df = result_df.append(row, ignore_index=True)

    logger.info(
        f"{len(result_df)}/{len(input_df)} many text-triple pairs have protein sequences for both nodes"
    )

    # Return the dataframe that only contains the entries that have protein sequences for both nodes
    return result_df


if __name__ == "__main__":
    add_protein_sequences_per_task(os.path.join(CELL_LINE_DIR, "cell_line_no_duplicates_ppi.tsv"))
