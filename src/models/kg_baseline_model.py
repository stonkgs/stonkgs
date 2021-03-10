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
from sklearn.model_selection import StratifiedKFold

from constants import DUMMY_EXAMPLE_TRIPLES, RANDOM_WALKS_PATH, EMBEDDINGS_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class KGEClassificationModel(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 5,  # the 5 does not mean anything, it is randomly chosen
        d_in: int = 768,
    ):
        """
        Initialize the components of the KGE based classification model, consisting of
        1) "Max-Pooling" (embedding-dimension-wise max)
        2) Linear layer (d_in x num_classes)
        3) Softmax
        """
        super(KGEClassificationModel, self).__init__()
        self.pooling = torch.max(dim=1)
        self.linear = torch.nn.Linear(d_in, num_classes)
        # TODO: make sure the dim parameters are correct
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        """
        The pooling component of this model is not specified as a separate nn Layer, since there are no parameters
        that need to be learned in this specific type of pooling. This layer returns the class probabilities.
        """
        h_pooled = self.pooling(x)
        linear_output = self.linear(h_pooled)
        y_pred = self.softmax(linear_output)
        return y_pred


# TODO: Own dataset class
class INDRAEntityDataset(torch.utils.data.Dataset):
    """Custom Dataset class for INDRA data."""

    def __init__(self, encodings, labels):
        # Assumes that the labels are numerically encoded
        # TODO: do all the embedding mappings in here
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def _prepare_df(embedding_path: str, sep: str = '\t') -> Dict[str, List[str]]:
    """Prepare dataframe to node->embeddings/random walks."""
    # Load embeddings
    df = pd.read_csv(
        embedding_path,
        sep=sep,
        header=None,
        index_col=0,
    )
    # node id -> embeddings
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

    return [{"train_idx": train_idx, "test_idx": test_idx} for train_idx, test_idx in skf.split(data_no_labels, labels)]


def run_kg_baseline_classification_cv(triples_path, embedding_path, random_walks_path, epochs=10) -> Dict:
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

    train_test_splits = get_train_test_splits(triples_df)

    embeddings_dict = _prepare_df(embedding_path)
    random_walks_dict = _prepare_df(random_walks_path)

    # Initialize the KGE-based model
    model = KGEClassificationModel(num_classes=len(triples_df["class"].unique()))
    f1_scores = []

    # TODO: 4. Train and test the model in a cv setting
    for indices in train_test_splits:
        # TODO: change to KG data processing
        # train_evidences = tokenizer(evidences_text[indices["train_idx"]].tolist(), truncation=True, padding=True)
        # test_evidences = tokenizer(evidences_text[indices["test_idx"]].tolist(), truncation=True, padding=True)

        # TODO: define dataloaders?

        # TODO: Change to CE loss for multiclass classification
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Train the model for epoch many epochs
        # TODO: tqdm?
        for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing x to the model
            # y_pred = model(x)

            # Compute and print loss
            # loss = criterion(y_pred, y)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            # loss.backward()
            optimizer.step()

        # TODO: predict

        # Use macro average for now
        # f1_scores.append(f1_score(test_labels, predicted_labels, average="macro"))

    logger.info(f'Mean f1-score: {np.mean(f1_scores)}')
    logger.info(f'Std f1-score: {np.std(f1_scores)}')

    return {"f1_score_mean": np.mean(f1_scores), "f1_score_std": np.std(f1_scores)}


if __name__ == "__main__":
    run_kg_baseline_classification_cv(
        triples_path=DUMMY_EXAMPLE_TRIPLES,
        embedding_path=EMBEDDINGS_PATH,
        random_walks_path=RANDOM_WALKS_PATH,
    )
