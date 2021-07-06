# -*- coding: utf-8 -*-

"""Checks if all entities in the fine-tuning datasets are contained in the pre-training data.

Run with:
python -m src.stonkgs.data.indra_check_overlaps
"""

import logging
import os
from typing import Dict, Set

import pandas as pd

from stonkgs.constants import (
    PRETRAINING_PATH,
    CELL_LINE_DIR,
    DISEASE_DIR,
    LOCATION_DIR,
    RELATION_TYPE_DIR,
    SPECIES_DIR,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_entities(
    df_path: str,
) -> Set[str]:
    """Returns all entities in a given dataset."""
    df = pd.read_csv(df_path, sep='\t')
    logger.info(f'Number of triples: {len(df)}')
    all_entities = set(df['source']).union(set(df['target']))
    logger.info(f'Successfully loaded {len(all_entities)} many entities \n')

    return all_entities


def find_missing_entities(
    pre_training_entities: Set[str],
    fine_tuning_entities_dict: Dict[str, Set[str]],
):
    """Identifies the entities (if any) present in the fine tuning BUT NOT in the pretraining dataset."""
    for name, fine_tuning_entities in fine_tuning_entities_dict.items():
        new_entities = fine_tuning_entities.difference(pre_training_entities)
        logger.info(f'For {name}, there are {len(new_entities)} many entities present in the fine-tuning set that are '
                    f'not in the pre-training data')
        # logger.info(new_entities)
        logger.info('\n')


def load_text(
    df_path: str,
) -> Set[str]:
    """Returns all text evidences in a given dataset."""
    df = pd.read_csv(df_path, sep='\t')
    all_text = set(df['evidence'])
    logger.info(f'Successfully loaded {len(all_text)} many text evidences')

    return all_text


def find_information_leakage(
    pre_training_text_evidences: Set[str],
    fine_tuning_evidences_dict: Dict[str, Set[str]],
):
    """Identifies how many text evidences from the fine-tuning sets are present in the pretraining dataset."""
    for name, fine_tuning_ev in fine_tuning_evidences_dict.items():
        new_entities = fine_tuning_ev.intersection(pre_training_text_evidences)
        logger.info(f'For {name}, there are {len(new_entities)} out of {len(fine_tuning_ev)} many entities that are '
                    f'also present in the pre-training data')
        # logger.info(new_entities)


if __name__ == '__main__':
    # 1. ENTITY OVERLAP
    # Get all the pre-training and fine-tuning datasets
    pre_training_ents = load_entities(PRETRAINING_PATH)

    # Iterate through all the fine-tuning stuff for ENTITIES
    paths = [CELL_LINE_DIR, DISEASE_DIR, LOCATION_DIR, RELATION_TYPE_DIR, SPECIES_DIR]
    names = [
        'cell_line_filtered',
        'disease_filtered',
        'location_filtered',
        'relation_type',
        'species_filtered',
    ]

    fine_tuning_dict = dict()
    for path, annot_name in zip(paths, names):
        fine_tuning_dict[annot_name] = load_entities(os.path.join(path, annot_name + '.tsv'))

    # Get an estimate of the entities that are not in the pretraining data
    find_missing_entities(
        pre_training_entities=pre_training_ents,
        fine_tuning_entities_dict=fine_tuning_dict,
    )

    # 2. INFORMATION LEAKAGE
    # Make sure that there is no information leakage, i.e. text evidence that is in both pre-training and one of
    # the fine-tuning datasets
    # Get the pre-training texts
    pre_training_texts = load_text(PRETRAINING_PATH)

    # Iterate through all the fine-tuning stuff for EVIDENCES
    fine_tuning_dict_evidences = {}
    for path, annot_name in zip(paths, names):
        fine_tuning_dict_evidences[annot_name] = load_text(os.path.join(path, annot_name + '.tsv'))

    # Get an estimate of the entities that are not in the pretraining data
    find_information_leakage(
        pre_training_text_evidences=pre_training_texts,
        fine_tuning_evidences_dict=fine_tuning_dict_evidences,
    )



