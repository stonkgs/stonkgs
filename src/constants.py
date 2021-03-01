# -*- coding: utf-8 -*-
#
# """Constants."""

import os

HERE = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))
PROJECT_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')

# Download from https://emmaa.s3.amazonaws.com/assembled/brca/statements_2021-01-30-17-21-54.gz
DUMMY_EXAMPLE_INDRA = os.path.join(DATA_DIR, 'statements_2021-01-30-17-21-54.json')
