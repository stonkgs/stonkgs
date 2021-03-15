# -*- coding: utf-8 -*-
"""
KG baseline model on the fine-tuning classification task, assuming the model embeddings are pre-trained.

Run with:
python -m src.models.kg_baseline_model
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from constants import DUMMY_EXAMPLE_TRIPLES, RANDOM_WALKS_PATH, EMBEDDINGS_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class KGEClassificationModel(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        d_in: int = 768,
    ):
        """
        Initialize the components of the KGE based classification model, consisting of
        1) "Max-Pooling" (embedding-dimension-wise max)
        2) Linear layer (d_in x num_classes)
        3) Softmax
        """
        super(KGEClassificationModel, self).__init__()
        self.pooling = torch.max
        self.linear = torch.nn.Linear(d_in, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        """
        A forward pass consisting of pooling (dimension-wise max), and a linear layer followed by softmax, in result
        returning the class probabilities.
        """
        h_pooled = self.pooling(x, dim=1).values
        linear_output = self.linear(h_pooled)
        y_pred = self.softmax(linear_output)
        return y_pred


class INDRAEntityDataset(torch.utils.data.Dataset):
    """Custom Dataset class for INDRA data."""

    def __init__(self, embedding_dict, random_walk_dict, sources, targets, labels, max_len=256):
        self.max_length = max_len
        # Two entities (source, target) of each triple
        self.sources = sources
        self.targets = targets
        # Initialize dictionary of node name -> embedding vector
        self.embedding_dict = embedding_dict
        # Initialize dictionary of node name -> random walk node names
        self.random_walk_dict = random_walk_dict
        # Get the embedding sequences for each triple
        self.embeddings = self.get_embeddings()
        # Assumes that the labels are numerically encoded
        self.labels = labels

    def __getitem__(self, idx):
        """Get embeddings and labels for given indices."""
        # Get embeddings (of random walk sequences of source + target) for given indices
        item = torch.tensor(self.embeddings[idx, :, :], dtype=float)
        # Get labels for given indices
        labels = torch.tensor(self.labels[idx])
        return item, labels

    def __len__(self):
        return len(self.labels)

    def get_embeddings(self):
        """Get the embedding sequences for each triple in the dataset (node embeddings from sources +
        targets random walks)."""
        # Number of total examples in the dataset
        number_of_triples = len(self.sources)
        # Get the embedding dimension by accessing a random element
        embedding_dim = len(next(iter(self.embedding_dict.values())))

        # Initialize the embedding array of dimension n x random_walk_length x embedding_dim
        embeddings = np.empty((number_of_triples, self.max_length, embedding_dim))

        # 1. Iterate through all triples: Get random walks for sources and targets using random_walk_dict
        for idx, (source, target) in enumerate(zip(self.sources, self.targets)):
            random_walk_source = self.random_walk_dict[source]
            random_walk_target = self.random_walk_dict[target]
            # 2. Concatenate and shorten the random walks if needed
            # The total random walk has the length max_length. Therefore its split half into the random walk of source
            # and half target.
            random_walk = [source] + random_walk_source.tolist()[:(self.max_length // 2 - 1)] + \
                          [target] + random_walk_target.tolist()[:(self.max_length // 2 - 1)]
            # 3. Get embeddings for each node using embedding_dict stated by its index in each random walk
            embeds_random_walk = np.stack([self.embedding_dict[node] for node in random_walk], axis=0)
            # The final embedding sequence for a given triple has the dimension max_length x embedding_dim
            embeddings[idx, :, :] = embeds_random_walk

        return embeddings


def _prepare_df(embedding_path: str, sep: str = '\t') -> Dict[str, List[str]]:
    """Prepare dataframe to node->embeddings/random walks."""
    # Load embeddings
    df = pd.read_csv(
        embedding_path,
        sep=sep,
        header=None,
        index_col=0,
    )
    # Node id -> embeddings
    return {
        index: row.values
        for index, row in df.iterrows()
    }


def get_train_test_splits(
    data: pd.DataFrame,
    label_column_name: str = "class",
    random_seed: int = 42,
    n_splits: int = 5
) -> List:
    """Returns deterministic train/test indices for n_splits based on the fine-tuning dataset that is passed."""
    # Leave out the label in the dataset
    data_no_labels = data.drop(label_column_name, axis=1)
    labels = data[label_column_name]

    # For now: implement stratified train/test splits with no validation split (since there's no HPO)
    # It is shuffled deterministically (determined by random_seed)
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=True)

    # Return a list of dictionaries for train and test indices
    return [{"train_idx": train_idx, "test_idx": test_idx} for train_idx, test_idx in skf.split(data_no_labels, labels)]


def run_kg_baseline_classification_cv(
    triples_path=DUMMY_EXAMPLE_TRIPLES,
    embedding_path=EMBEDDINGS_PATH,
    random_walks_path=RANDOM_WALKS_PATH,
    n_splits=5,
    epochs=200,
    batch_size=32,
    lr=1e-4
) -> Dict:
    """Run KG baseline classification."""

    # Step 1. load the tsv file with the annotation types you want to test and make the splits
    triples_df = pd.read_csv(
        triples_path,
        sep='\t',
        usecols=[
            'source',
            # 'relation',
            'target',
            # 'evidence',
            # 'pmid',
            'class'
        ],
    )
    # Numerically encode labels
    unique_tags = set(label for label in triples_df["class"])
    tag2id = {label: number for number, label in enumerate(unique_tags)}
    id2tag = {value: key for key, value in tag2id.items()}

    # Cross entropy loss weights based on the inverse of the class counts (Inverse Number of Samples, INS)
    weights = [1 / len([i for i in triples_df["class"] if i == id2tag[id_num]]) for id_num in range(len(unique_tags))]
    labels = pd.Series([int(tag2id[label]) for label in triples_df["class"]])

    # Get the train/test split indices
    train_test_splits = get_train_test_splits(triples_df, n_splits=n_splits)

    # Prepare embeddings and random walks
    embeddings_dict = _prepare_df(embedding_path)
    random_walks_dict = _prepare_df(random_walks_path)

    # Initialize f1-scores
    f1_scores = []

    # Train and test the model in a cv setting
    for indices in train_test_splits:
        kg_embeds = INDRAEntityDataset(
            embeddings_dict,
            random_walks_dict,
            triples_df["source"],
            triples_df["target"],
            labels
        )

        # Sample elements randomly from a given list of ids, no replacement
        train_subsampler = torch.utils.data.SubsetRandomSampler(indices["train_idx"])
        test_subsampler = torch.utils.data.SubsetRandomSampler(indices["test_idx"])

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            kg_embeds,
            batch_size=batch_size,
            sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            kg_embeds,
            batch_size=batch_size,
            sampler=test_subsampler)

        model = KGEClassificationModel(num_classes=len(triples_df["class"].unique()))

        # Initialize loss (with weights due to possibly imbalanced classes) and optimizer
        criterion = torch.nn.CrossEntropyLoss(reduction="mean", weight=torch.tensor(weights))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # Train the model for epoch many epochs
        # TODO: tqdm
        for epoch in range(epochs):
            # Iterate over the DataLoader for training data
            for train_data in trainloader:
                train_inputs, train_labels = train_data
                # Zero the gradients
                optimizer.zero_grad()
                # Perform forward pass
                train_outputs = model(train_inputs.float())

                # Compute loss
                loss = criterion(train_outputs, train_labels)
                # Perform backward pass
                loss.backward()
                # Perform optimization
                optimizer.step()

        # Predict
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            all_true_labels = []
            all_pred_labels = []

            # Get labels for each batch
            for test_data in testloader:
                # Get inputs
                test_inputs, test_labels = test_data
                all_true_labels = all_true_labels + test_labels.tolist()
                # Generate outputs
                test_outputs = model(test_inputs.float())
                # Class probabilities to labels
                _, predicted_labels = torch.max(test_outputs.data, 1)
                all_pred_labels = all_pred_labels + predicted_labels.tolist()

        # Append f1 score per split based on the macro average
        f1_scores.append(f1_score(all_true_labels, all_pred_labels, average="macro"))

    # Log mean and std f1-scores from the cross validation procedure
    logger.info(f'Mean f1-score: {np.mean(f1_scores)}')
    logger.info(f'Std f1-score: {np.std(f1_scores)}')

    return {"f1_score_mean": np.mean(f1_scores), "f1_score_std": np.std(f1_scores)}


if __name__ == "__main__":
    run_kg_baseline_classification_cv()
