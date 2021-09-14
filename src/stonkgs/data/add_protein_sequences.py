# -*- coding: utf-8 -*-

"""Add the protein sequences to text-triple pairs that have descriptions."""

import logging
import os

import numpy as np
import pandas as pd
from protmapper.uniprot_client import get_id_from_entrez, get_sequence
from tqdm import tqdm, trange

from stonkgs.constants import (
    CELL_LINE_DIR,
    CELL_TYPE_DIR,
    CORRECT_DIR,
    DISEASE_DIR,
    LOCATION_DIR,
    ORGAN_DIR,
    PRETRAINING_DIR,
    SPECIES_DIR,
    RELATION_TYPE_DIR,
)

# Initialize the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def add_protein_sequences_per_task(
    input_file: str,
    output_file: str,
    chunk_size: int = 10000,
):
    """Add the protein sequences to an existing dataframe consisting of the text-triple pairs with descriptions."""
    # Read the input file
    input_df = pd.read_csv(input_file, sep="\t", index_col=None)
    # See if a partially processed df already exists
    try:
        logger.info("Existing file found")
        result_df = pd.read_csv(output_file, sep="\t", index_col=None)
        last_row = result_df.iloc[-1][["source_id", "target_id", "evidence"]]
        # Find the last processed row in the original input file
        input_index = input_df.index[
            (input_df['source_id'] == last_row["source_id"]) &
            (input_df['target_id'] == last_row["target_id"]) &
            (input_df['evidence'] == last_row["evidence"])
        ][0]
        logger.info(f"The original file has been processed until line no. {input_index}")
        # Define from which batch to start again
        begin_cn = int(input_index // chunk_size) + 1
        logger.info(f"Starting from batch no. {begin_cn}")
    except FileNotFoundError:
        logger.info("Creating new results file")
        begin_cn = 0

    # Define batches
    cn = len(input_df) // chunk_size + 1

    # Iterate batch-wise through all rows of the input file/df
    for i in trange(begin_cn, cn, desc="Batch-wise processing of the input file"):
        partial_result_df = pd.DataFrame(columns=list(input_df.columns) + ["source_prot", "target_prot"])

        chunk_df = input_df.iloc[chunk_size * i: np.min([chunk_size * (i + 1), len(input_df)])]

        # Find the protein sequences for each row in the batch
        for _, row in chunk_df.iterrows():
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
                    partial_result_df = partial_result_df.append(row, ignore_index=True)

        # Append the resulting df with the partial result
        partial_result_df.to_csv(output_file, sep="\t", index=False, mode='a', header=(i == 0))

    # Read result for the final overview of the number of text-triple pairs with protein sequences
    result_df = pd.read_csv(output_file, sep="\t", index_col=None)

    logger.info(
        f"{len(result_df)}/{len(input_df)} many text-triple pairs have protein sequences for both nodes"
    )
    return


def add_protein_sequences(
    chunk_size: int = 10000,
):
    # Define all the task and file names
    task_names = [
        "cell_line",
        "cell_type",
        "correct_incorrect_binary",
        "correct_incorrect_multiclass",
        "disease",
        "location",
        "organ",
        "relation_type",
        "species",
        "pretraining",
    ]
    directories = [
        CELL_LINE_DIR,
        CELL_TYPE_DIR,
        CORRECT_DIR,
        CORRECT_DIR,
        DISEASE_DIR,
        LOCATION_DIR,
        ORGAN_DIR,
        RELATION_TYPE_DIR,
        SPECIES_DIR,
        PRETRAINING_DIR,
    ]
    original_file_names = [i + "_no_duplicates_ppi.tsv" for i in task_names[:-1]] + [
        task_names[-1] + "_triples_ppi.tsv"
    ]
    new_file_names = [i + "_ppi_prot.tsv" for i in task_names]

    # Add the protein sequences for each file
    for task_name, directory, original_file_name, new_file_name in zip(
        task_names,
        directories,
        original_file_names,
        new_file_names,
    ):
        logger.info(f"Processing {task_name} data")

        add_protein_sequences_per_task(
            os.path.join(directory, original_file_name),
            os.path.join(directory, new_file_name),
            chunk_size=chunk_size,
        )
        # Save the df with the protein sequences
        # prot_df.to_csv(os.path.join(directory, new_file_name), sep="\t", index=None)

        logger.info(f"Processing {task_name} data complete")


if __name__ == "__main__":
    # add_protein_sequences()
    add_protein_sequences_per_task(
        os.path.join(PRETRAINING_DIR, "pretraining_triples_ppi.tsv"),
        os.path.join(PRETRAINING_DIR, "pretraining_ppi_prot.tsv"),
    )
