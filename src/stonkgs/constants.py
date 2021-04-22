# -*- coding: utf-8 -*-

"""Constants."""

import os

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))
PROJECT_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))

# Move to parent folder by os.sep.join(PROJECT_DIR.split(os.sep)[:-1]) to get to the data folder
DATA_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), 'data')

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

# Path for the pretraining data
PRETRAINING_DIR = os.path.join(INPUT_DIR, 'pretraining')
PRETRAINING_PATH = os.path.join(PRETRAINING_DIR, 'pretraining_triples.tsv')
PRETRAINING_PREPROCESSED_DF_PATH = os.path.join(PRETRAINING_DIR, 'pretraining_preprocessed.pkl')

# Move to parent folder by os.sep.join(PROJECT_DIR.split(os.sep)[:-1]) to get to the models folder
MODELS_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), 'models')
NLP_BL_OUTPUT_DIR = os.path.join(MODELS_DIR, 'nlp-baseline')
KG_HPO_DIR = os.path.join(MODELS_DIR, 'kg-hpo')
KG_BL_OUTPUT_DIR = os.path.join(MODELS_DIR, 'kg-baseline')
STONKGS_PRETRAINING_DIR = os.path.join(MODELS_DIR, 'stonkgs-pretraining')
STONKGS_OUTPUT_DIR = os.path.join(MODELS_DIR, 'stonkgs')

EMBEDDINGS_PATH = os.path.join(KG_HPO_DIR, 'embeddings_best_model.tsv')
RANDOM_WALKS_PATH = os.path.join(KG_HPO_DIR, 'random_walks_best_model.tsv')

# Move to parent folder by os.sep.join(PROJECT_DIR.split(os.sep)[:-1]) to get to the logs folder
LOG_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), 'logs')

# Load constants from environment variables (set in the .env file)
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
# Load a constant for distinguishing between local and cluster execution (default = True)
LOCAL_EXECUTION = os.getenv("LOCAL_EXECUTION") or "True"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MISC_DIR, exist_ok=True)
os.makedirs(KG_HPO_DIR, exist_ok=True)
os.makedirs(KG_BL_OUTPUT_DIR, exist_ok=True)
os.makedirs(STONKGS_OUTPUT_DIR, exist_ok=True)
os.makedirs(NLP_BL_OUTPUT_DIR, exist_ok=True)

os.makedirs(ORGAN_DIR, exist_ok=True)
os.makedirs(DISEASE_DIR, exist_ok=True)
os.makedirs(LOCATION_DIR, exist_ok=True)
os.makedirs(CELL_TYPE_DIR, exist_ok=True)
os.makedirs(CELL_LINE_DIR, exist_ok=True)
os.makedirs(SPECIES_DIR, exist_ok=True)
os.makedirs(RELATION_TYPE_DIR, exist_ok=True)

# Download from https://emmaa.s3.amazonaws.com/assembled/covid19/statements_2021-03-08-18-24-29.gz
DUMMY_EXAMPLE_INDRA = os.path.join(RAW_DIR, 'statements_2021-01-30-17-21-54.json')
# Can be created by running python -m src.stonkgs.data.indra
DUMMY_EXAMPLE_TRIPLES = os.path.join(LOCATION_DIR, 'location.tsv')

# Specify the (huggingface) language model that is used in this project
NLP_MODEL_TYPE = "dmis-lab/biobert-v1.1"
