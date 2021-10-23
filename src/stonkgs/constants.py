# -*- coding: utf-8 -*-

"""Constants."""

import os

import pystow
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))
PROJECT_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))

# Move to parent folder by os.sep.join(PROJECT_DIR.split(os.sep)[:-1]) to get to the data folder
DATA_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "data")

# Sub-directories of data
RAW_DIR = os.path.join(DATA_DIR, "raw")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
MISC_DIR = os.path.join(DATA_DIR, "misc")

# Directories for each annotation type
CORRECT_DIR = os.path.join(INPUT_DIR, "correct_incorrect")
DISEASE_DIR = os.path.join(INPUT_DIR, "disease")
LOCATION_DIR = os.path.join(INPUT_DIR, "location")
CELL_LINE_DIR = os.path.join(INPUT_DIR, "cell_line")
CELL_TYPE_DIR = os.path.join(INPUT_DIR, "cell_type")
ORGAN_DIR = os.path.join(INPUT_DIR, "organ")
SPECIES_DIR = os.path.join(INPUT_DIR, "species")
RELATION_TYPE_DIR = os.path.join(INPUT_DIR, "relation_type")
NDD_DIR = os.path.join(INPUT_DIR, "ndd")

# Path for the pretraining data
PRETRAINING_DIR = os.path.join(INPUT_DIR, "pretraining")
PRETRAINING_PATH = os.path.join(PRETRAINING_DIR, "pretraining_triples.tsv")
PRETRAINING_PROT_PATH = os.path.join(PRETRAINING_DIR, "pretraining_ppi_prot.tsv")
PRETRAINING_PREPROCESSED_PROT_DF_PATH = os.path.join(
    PRETRAINING_DIR, "pretraining_ppi_prot_preprocessed.pkl"
)
PRETRAINING_PROT_DUMMY_PATH = os.path.join(PRETRAINING_DIR, "pretraining_ppi_prot_dummy.tsv")
PRETRAINING_PREPROCESSED_DF_PATH = os.path.join(PRETRAINING_DIR, "pretraining_preprocessed.pkl")
PRETRAINING_PREPROCESSED_POSITIVE_DF_PATH = os.path.join(
    PRETRAINING_DIR, "pretraining_preprocessed_positive.pkl"
)

# Move to parent folder by os.sep.join(PROJECT_DIR.split(os.sep)[:-1]) to get to the models folder
MODELS_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "models")
NLP_BL_OUTPUT_DIR = os.path.join(MODELS_DIR, "nlp-baseline")
KG_HPO_DIR = os.path.join(MODELS_DIR, "kg-hpo")
KG_BL_OUTPUT_DIR = os.path.join(MODELS_DIR, "kg-baseline")
STONKGS_PRETRAINING_DIR = os.path.join(MODELS_DIR, "stonkgs-pretraining")
PROTSTONKGS_PRETRAINING_DIR = os.path.join(MODELS_DIR, "protstonkgs-pretraining")
STONKGS_PRETRAINING_NO_NSP_DIR = os.path.join(MODELS_DIR, "stonkgs-pretraining-no-nsp")
PRETRAINED_STONKGS_DUMMY_PATH = os.path.join(
    STONKGS_PRETRAINING_DIR, "pretrained-stonkgs-dummy-model"
)
PRETRAINED_PROTSTONKGS_PATH = os.path.join(PROTSTONKGS_PRETRAINING_DIR, "pretrained-protstonkgs")
STONKGS_OUTPUT_DIR = os.path.join(MODELS_DIR, "stonkgs")
PROT_STONKGS_OUTPUT_DIR = os.path.join(MODELS_DIR, "protstonkgs")
DEEPSPEED_CONFIG_PATH = os.path.join(MODELS_DIR, "deepspeed_config_zero2.json")

EMBEDDINGS_PATH = os.path.join(KG_HPO_DIR, "embeddings_best_model.tsv")
TRANSE_EMBEDDINGS_PATH = os.path.join(KG_HPO_DIR, "transe_embeddings_best_model.tsv")
PROT_EMBEDDINGS_PATH = os.path.join(KG_HPO_DIR, "embeddings_prot_best_model.tsv")
RANDOM_WALKS_PATH = os.path.join(KG_HPO_DIR, "random_walks_best_model.tsv")
PROT_RANDOM_WALKS_PATH = os.path.join(KG_HPO_DIR, "random_walks_prot_best_model.tsv")

# Move to parent folder by os.sep.join(PROJECT_DIR.split(os.sep)[:-1]) to get to the logs folder
LOG_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "logs")

# Use dotenv to properly load the device-dependent mlflow tracking URI
load_dotenv()
# Load constants from environment variables (set in the .env file)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_FINETUNING_TRACKING_URI = os.getenv("MLFLOW_FINETUNING_TRACKING_URI")
# Load a constant for distinguishing between local and cluster execution (default = True)
LOCAL_EXECUTION = os.getenv("LOCAL_EXECUTION") or "True"

# Directory for visualizations, save it in the notebooks dir for now
NOTEBOOKS_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "notebooks")
VISUALIZATIONS_DIR = os.path.join(NOTEBOOKS_DIR, "visualization")

"""Create directories"""
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
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

os.makedirs(PRETRAINING_DIR, exist_ok=True)
os.makedirs(DISEASE_DIR, exist_ok=True)
os.makedirs(LOCATION_DIR, exist_ok=True)
os.makedirs(CELL_LINE_DIR, exist_ok=True)
os.makedirs(SPECIES_DIR, exist_ok=True)
os.makedirs(RELATION_TYPE_DIR, exist_ok=True)

# Specify the raw (complete) INDRA json file
INDRA_RAW_JSON = os.path.join(RAW_DIR, "raw_statements.json")
# Download from https://emmaa.s3.amazonaws.com/assembled/covid19/statements_2021-03-08-18-24-29.gz
DUMMY_EXAMPLE_INDRA = os.path.join(RAW_DIR, "statements_2021-01-30-17-21-54.json")
# Can be created by running python -m src.stonkgs.data.indra
DUMMY_EXAMPLE_TRIPLES = os.path.join(LOCATION_DIR, "location.tsv")

# Specify the (huggingface) language model that is used as a basis for STonKGs
NLP_MODEL_TYPE = "dmis-lab/biobert-v1.1"
# Specify the (huggingface) language model that is used as a basis for ProtSTonKGs
PROTSTONKGS_MODEL_TYPE = "google/bigbird-roberta-base"
# Specify the protein bert backbone model type
PROT_SEQ_MODEL_TYPE = "Rostlab/prot_bert"

# Specify the vocab file of the language model that is used in this project
# (the file can be obtained here: https://huggingface.co/dmis-lab/biobert-v1.1/tree/main)
VOCAB_URL = "https://huggingface.co/dmis-lab/biobert-v1.1/raw/main/vocab.txt"
VOCAB_FILE = pystow.ensure("stonkgs", "misc", url=VOCAB_URL)
