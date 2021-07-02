# -*- coding: utf-8 -*-

"""Script for visualizing the embeddings of 1) NLP baseline 2) KG baseline and 3) STonKGs."""

# Imports
import logging
import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import umap
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from stonkgs.constants import (
    EMBEDDINGS_PATH,
    MISC_DIR,
    NLP_MODEL_TYPE,
    PRETRAINED_STONKGS_DUMMY_PATH,
    RANDOM_WALKS_PATH,
    SPECIES_DIR,
    VISUALIZATIONS_DIR,
)
from stonkgs.models.kg_baseline_model import _prepare_df, INDRAEntityDataset
from stonkgs.models.stonkgs_model import STonKGsForPreTraining

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_stonkgs_data(
    unprocessed_df: pd.DataFrame,
) -> pd.DataFrame:
    """Preprocess the finetuning data for the stonkgs model."""
    sep_id = 102
    unk_id = 100  # does not matter, will not be used here
    kg_name_to_idx = {key: i for i, key in enumerate(embeddings_dict.keys())}

    # Convert random walk sequences to list of numeric indices
    random_walk_idx_dict = {k: [kg_name_to_idx[node] for node in v] for k, v in random_walks_dict.items()}

    # Get the length of the text or entity embedding sequences (2 random walks + 2 = entity embedding sequence length)
    random_walk_length = len(next(iter(random_walk_idx_dict.values())))
    half_length = random_walk_length * 2 + 2

    # Initialize the preprocessed data
    fine_tuning_preprocessed = []

    # Log progress with a progress bar
    for _, row in tqdm(
        unprocessed_df.iterrows(),
        total=unprocessed_df.shape[0],
        desc='Preprocessing the fine-tuning dataset',
    ):
        # 1. "Token type IDs": 0 for text tokens, 1 for entity tokens
        token_type_ids = [0] * half_length + [1] * half_length

        # 2. Tokenization for getting the input ids and attention masks for the text
        # Use encode_plus to also get the attention mask ("padding" mask)
        encoded_text = tokenizer.encode_plus(
            row['evidence'],
            padding='max_length',
            truncation=True,
            max_length=half_length,
        )
        text_token_ids = encoded_text['input_ids']
        text_attention_mask = encoded_text['attention_mask']

        # 3. Get the random walks sequence and the node indices, add the SEP (usually with id=102) in between
        # Use a sequence of UNK tokens if the node is not contained in the dictionary of the nodes from pre-training
        random_w_source = random_walk_idx_dict[
            row['source']
        ] if row['source'] in random_walk_idx_dict.keys() else [unk_id] * random_walk_length
        random_w_target = random_walk_idx_dict[
            row['target']
        ] if row['target'] in random_walk_idx_dict.keys() else [unk_id] * random_walk_length
        random_w_ids = random_w_source + [sep_id] + random_w_target + [sep_id]

        # 4. Total attention mask (attention mask is all 1 for the entity sequence)
        attention_mask = text_attention_mask + [1] * half_length

        # 5. Total input_ids = half text ids + half entity ids
        input_ids = text_token_ids + random_w_ids

        # Add all the features to the preprocessed data
        fine_tuning_preprocessed.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,  # Remove the MLM, ELM and NSP labels since it's not needed anymore
        })

    # Put the preprocessed data into a dataframe
    fine_tuning_preprocessed_df = pd.DataFrame(fine_tuning_preprocessed)

    return fine_tuning_preprocessed_df


def get_nlp_embeddings(
    list_of_indices: List,
) -> pd.DataFrame:
    """Returns a data frame of with embedding_sequence, label columns."""
    all_embed_sequences = pd.DataFrame(columns=["embedding", "label"])

    for idx in tqdm(list_of_indices):
        nlp_evidence = tokenizer(sampled_df.iloc[idx]["evidence"], return_tensors="pt", padding='max_length',
                                 truncation=True)
        nlp_hidden_states = nlp_baseline(**nlp_evidence, output_hidden_states=True).pooler_output[0]
        all_embed_sequences = all_embed_sequences.append(
            {"embedding": nlp_hidden_states.tolist(), "label": sampled_df.iloc[idx]["class"]},
            ignore_index=True,
        )

    return all_embed_sequences


def get_kg_embeddings(
    list_of_indices: List,
) -> pd.DataFrame:
    """Returns a data frame of with embedding_sequence, label columns."""
    all_embed_sequences = pd.DataFrame(columns=["embedding", "label"])

    for idx in tqdm(list_of_indices):
        all_embed_sequences = all_embed_sequences.append(
            {"embedding": torch.max(kg_baseline[idx][0], dim=0).values.tolist(), "label": kg_baseline[idx][1].item()},
            ignore_index=True,
        )

    return all_embed_sequences


def get_stonkgs_embeddings(
    list_of_indices: List,
) -> pd.DataFrame:
    """Returns a data frame of with embedding_sequence, label columns."""
    all_embed_sequences = pd.DataFrame(columns=["embedding", "label"])

    for idx in tqdm(list_of_indices):
        data_entry = {key: torch.tensor([value]) for key, value in dict(stonkgs_data.iloc[idx]).items()}
        stonkgs_hidden_states = stonkgs(**data_entry, return_dict=True).pooler_output[0]
        all_embed_sequences = all_embed_sequences.append(
            {"embedding": stonkgs_hidden_states.tolist(), "label": sampled_df.iloc[idx]["class"]},
            ignore_index=True,
        )

    return all_embed_sequences


def generate_umap_plot(
    embedding_df: pd.DataFrame,
    name: str,
    task: str = "species",
):
    # Convert the embeddings into the required format
    embedding_series = embedding_df["embedding"]
    embedding_matrix = pd.DataFrame.from_dict(dict(zip(embedding_series.index, embedding_series.values))).values.T

    labels = embedding_df["label"]

    # Initialize UMAP
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1)

    # Create the 2-dimensional embeddings
    two_dim_embedding = reducer.fit_transform(embedding_matrix)

    # Plot it
    plt.figure(figsize=(16, 8))
    plt.scatter(
        two_dim_embedding[:, 0],
        two_dim_embedding[:, 1],
        c=[sns.color_palette()[x] for x in labels.map({9606: 0, 10116: 1, 10090: 2})]
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'UMAP projection of the {name} model for the {task} dataset', fontsize=24)
    plt.savefig(os.path.join(MISC_DIR, name + "_embedding_figure.png"), dpi=300)


if __name__ == "__main__":
    # Initialize the task specific dataset
    """    task_dir = SPECIES_DIR
    number_unique_tags = 3
    dataset_version = "species_no_duplicates.tsv"
    number_entries = 800

    # Load the dataset + remove the unnecessary column
    task_specific_dataset = pd.read_csv(os.path.join(task_dir, dataset_version), sep="\t", index_col=None)
    if "Unnamed: 0" in task_specific_dataset.columns.values:
        task_specific_dataset.drop(columns=["Unnamed: 0"], inplace=True)

    # Load the embedding and random walk dicts
    embeddings_dict = _prepare_df(EMBEDDINGS_PATH)
    random_walks_dict = _prepare_df(RANDOM_WALKS_PATH)

    logger.info('Finished loading the KG embedding dictionaries')

    # Filter out unseen nodes
    task_specific_dataset = task_specific_dataset[
        task_specific_dataset['source'].isin(embeddings_dict.keys()) & task_specific_dataset['target'].isin(
            embeddings_dict.keys())
        ].reset_index(drop=True)

    logger.info(task_specific_dataset['class'].value_counts())

    # Sample the classes equally
    sampled_df = pd.DataFrame()

    for cls in np.unique(task_specific_dataset["class"]):
        cls_specific_samples = task_specific_dataset[task_specific_dataset['class'] == cls].sample(
            n=number_entries // number_unique_tags)
        sampled_df = sampled_df.append(cls_specific_samples)

    sampled_df.reset_index(drop=True, inplace=True)

    logger.info(sampled_df['class'].value_counts())

    # Initialize the models and some additional required stuff
    tokenizer = BertTokenizer.from_pretrained(NLP_MODEL_TYPE, model_max_length=512)
    labels = sampled_df["class"].tolist()
    # nlp_baseline = BertModel.from_pretrained(NLP_MODEL_TYPE)
    kg_baseline = INDRAEntityDataset(
        embeddings_dict,
        random_walks_dict,
        sampled_df["source"],
        sampled_df["target"],
        sampled_df["class"],
    )
    stonkgs = STonKGsForPreTraining.from_pretrained(
        pretrained_model_name_or_path=PRETRAINED_STONKGS_DUMMY_PATH,
    )
    stonkgs_data = preprocess_stonkgs_data(sampled_df)"""

    # Get the embeddings
    # 1. NLP
    # nlp_embeds = get_nlp_embeddings(list(range(len(sampled_df))))
    # nlp_embeds.to_csv(os.path.join(MISC_DIR, "nlp_embeds_visualization.tsv"), sep="\t", index=None)

    # 2. KG
    # kg_embeds = get_kg_embeddings(list(range(len(sampled_df))))
    # kg_embeds.to_csv(os.path.join(MISC_DIR, "kg_embeds_visualization.tsv"), sep="\t", index=None)

    # 3. STonKGs
    # stonkgs_embeds = get_stonkgs_embeddings(list(range(len(sampled_df))))
    # stonkgs_embeds.to_csv(os.path.join(MISC_DIR, "stonkgs_embeds_visualization.tsv"), sep="\t", index=None)

    # Load them
    # 1. NLP
    nlp_embeds = pd.read_csv(
        os.path.join(MISC_DIR, "nlp_embeds_visualization.tsv"),
        index_col=None,
        sep="\t",
        converters={"embedding": lambda x: x.strip("[]").split(", ")}
    )
    generate_umap_plot(nlp_embeds, "nlp")

    # 2. KG
    kg_embeds = pd.read_csv(
        os.path.join(MISC_DIR, "kg_embeds_visualization.tsv"),
        index_col=None,
        sep="\t",
        converters={"embedding": lambda x: x.strip("[]").split(", ")}
    )
    generate_umap_plot(kg_embeds, "kg")

    # 3. STonKGs
    stonkgs_embeds = pd.read_csv(
        os.path.join(MISC_DIR, "stonkgs_embeds_visualization.tsv"),
        index_col=None,
        sep="\t",
        converters={"embedding": lambda x: x.strip("[]").split(", ")}
    )
    generate_umap_plot(stonkgs_embeds, "stonkgs")
