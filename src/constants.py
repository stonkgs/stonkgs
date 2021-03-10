# -*- coding: utf-8 -*-
#
# """Constants."""

import os

HERE = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))
PROJECT_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))

DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# Sub-directories of data
RAW_DIR = os.path.join(DATA_DIR, 'raw')
INPUT_DIR = os.path.join(DATA_DIR, 'input')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
MISC_DIR = os.path.join(DATA_DIR, 'misc')

# Directories for each annotation type
ORGAN_DIR = os.path.join(INPUT_DIR, 'organ')
DISEASE_DIR = os.path.join(INPUT_DIR, 'disease')
LOCATION_DIR = os.path.join(INPUT_DIR, 'location')
CELL_TYPE_DIR = os.path.join(INPUT_DIR, 'cell_type')
CELL_LINE_DIR = os.path.join(INPUT_DIR, 'cell_line')
SPECIES_DIR = os.path.join(INPUT_DIR, 'species')
RELATION_TYPE_DIR = os.path.join(INPUT_DIR, 'relation_type')

MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
NLP_BL_OUTPUT_DIR = os.path.join(MODELS_DIR, 'nlp-baseline')
KG_HPO_DIR = os.path.join(MODELS_DIR, 'kg-hpo')
KG_BL_OUTPUT_DIR = os.path.join(MODELS_DIR, 'kg-baseline')
STONKGS_OUTPUT_DIR = os.path.join(MODELS_DIR, 'stonkgs')

LOG_DIR = os.path.join(PROJECT_DIR, 'logs')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MISC_DIR, exist_ok=True)

os.makedirs(ORGAN_DIR, exist_ok=True)
os.makedirs(DISEASE_DIR, exist_ok=True)
os.makedirs(LOCATION_DIR, exist_ok=True)
os.makedirs(CELL_TYPE_DIR, exist_ok=True)
os.makedirs(CELL_LINE_DIR, exist_ok=True)
os.makedirs(SPECIES_DIR, exist_ok=True)
os.makedirs(RELATION_TYPE_DIR, exist_ok=True)

# Download from https://emmaa.s3.amazonaws.com/assembled/covid19/statements_2021-03-08-18-24-29.gz
DUMMY_EXAMPLE_INDRA = os.path.join(RAW_DIR, 'statements_2021-03-08-18-24-29.json')
# Can be created by running python -m src.data.indra
DUMMY_EXAMPLE_TRIPLES = os.path.join(DATA_DIR, 'location.tsv')

# Specify the (huggingface) language model that is used in this project
NLP_MODEL_TYPE = "monologg/biobert_v1.1_pubmed"
