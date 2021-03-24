# -*- coding: utf-8 -*-
"""
KG baseline model on the fine-tuning classification task, assuming the model embeddings are pre-trained.

Run with:
python -m src.stonkgs.models.kg_baseline_model
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from ..constants import DUMMY_EXAMPLE_TRIPLES, EMBEDDINGS_PATH, RANDOM_WALKS_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class KGEClassificationModel(pl.LightningModule):
    """KGE baseline model."""

    def __init__(
        self,
        num_classes,
        class_weights,
        d_in: int = 768,
        lr: float = 1e-4,
    ):
        """Initialize the components of the KGE based classification model.

        :param num_classes: number of classes
        :param class_weights: class weights
        :param d_in: dimensions
        :param lr: learning rate

        The model consists of
        1) "Max-Pooling" (embedding-dimension-wise max)
        2) Linear layer (d_in x num_classes)
        3) Softmax
        (Not part of the model, but of the class: class_weights for the cross_entropy function)
        """
        super(KGEClassificationModel, self).__init__()

        # Model architecture
        self.pooling = torch.max
        self.linear = torch.nn.Linear(d_in, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

        # Other class-specific parameters
        # Class weights for CE loss
        self.class_weights = torch.tensor(class_weights)  #
        # Learning rate
        self.lr = lr

    def forward(self, x):
        """Perform forward pass consisting of pooling (dimension-wise max), and a linear layer followed by softmax.

        :param x: embedding sequences (random walk embeddings) for the given triples
        :return: class probabilities for the given triples
        """
        h_pooled = self.pooling(x, dim=1).values
        linear_output = self.linear(h_pooled)
        y_pred = self.softmax(linear_output)
        # Note that the forward pass returns class probabilities.
        return y_pred

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        """Perform one training step on one batch using CE loss."""
        train_inputs, train_labels = batch
        train_outputs = self.forward(train_inputs)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="mean", weight=self.class_weights)
        loss = loss_fct(train_outputs, train_labels)
        return loss

    def test_step(self, batch, batch_nb):
        """Perform one test step on a given batch and return the macro-averaged f1 score over all batches."""
        test_inputs, test_labels = batch
        test_class_probs = self.forward(test_inputs)
        # Class probabilities to class labels
        test_predictions = torch.argmax(test_class_probs, dim=1)

        # Get the macro-averaged f1-score
        test_f1 = f1_score(test_labels, test_predictions, average="macro")
        return {'test_f1': torch.tensor(test_f1)}

    def test_epoch_end(self, outputs):
        """Return average and std macro-averaged f1-score over all batches."""
        mean_test_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()
        std_test_f1 = torch.stack([x['test_f1'] for x in outputs]).std()

        return {'mean_test_f1': mean_test_f1, 'std_test_f1': std_test_f1}


class INDRAEntityDataset(torch.utils.data.Dataset):
    """Custom dataset class for INDRA data."""

    def __init__(self, embedding_dict, random_walk_dict, sources, targets, labels, max_len=256):
        """Initialize INDRA Dataset based on random walk embeddings for 2 nodes in each triple."""
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
        item = torch.tensor(self.embeddings[idx, :, :], dtype=torch.float)
        # Get labels for given indices
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return item, labels

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.labels)

    def get_embeddings(self):
        """Get the embedding sequences for each triple in the dataset (node emb from sources + targets random walks).

        :return: embedding sequences for each triple in the dataset
        """
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
                          [target] + random_walk_target.tolist()[:(self.max_length // 2 - 1)]  # noqa: N400
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
    n_splits: int = 5,
) -> List:
    """Return deterministic train/test indices for n_splits based on the fine-tuning dataset that is passed."""
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
    epochs=20,
    train_batch_size=16,
    test_batch_size=64,
    lr=1e-4,
) -> Dict[str, float]:
    """Run KG baseline classification."""
    # Step 1. load the tsv file with the annotation types you want to test and make the splits
    triples_df = pd.read_csv(
        triples_path,
        sep='\t',
        usecols=[
            'source',
            'target',
            'class',
        ],
    )
    # Numerically encode labels
    unique_tags = set(label for label in triples_df["class"])
    tag2id = {label: number for number, label in enumerate(unique_tags)}
    id2tag = {value: key for key, value in tag2id.items()}

    # Get labels
    labels = pd.Series([int(tag2id[label]) for label in triples_df["class"]])

    # Get the train/test split indices
    train_test_splits = get_train_test_splits(triples_df, n_splits=n_splits)

    # Prepare embeddings and random walks
    embeddings_dict = _prepare_df(embedding_path)
    random_walks_dict = _prepare_df(random_walks_path)

    # Initialize f1-scores
    f1_scores = []

    # Initialize INDRA for KG baseline dataset
    kg_embeds = INDRAEntityDataset(
        embeddings_dict,
        random_walks_dict,
        triples_df["source"],
        triples_df["target"],
        labels,
    )

    # Train and test the model in a cv setting
    for indices in train_test_splits:
        # Sample elements randomly from a given list of ids, no replacement
        train_subsampler = torch.utils.data.SubsetRandomSampler(indices["train_idx"])
        test_subsampler = torch.utils.data.SubsetRandomSampler(indices["test_idx"])
        # CE class weights for the model based on training data class distribution,
        # based on the class counts (Inverse Number of Samples, INS)
        weights = [
            1 / len([i
                     for i in triples_df.iloc[indices["train_idx"], :]["class"]  # note that we only employ train idx
                     if i == id2tag[id_num]
                     ])
            for id_num in range(len(unique_tags))
        ]

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            kg_embeds,
            batch_size=train_batch_size,
            sampler=train_subsampler,
        )
        testloader = torch.utils.data.DataLoader(
            kg_embeds,
            batch_size=test_batch_size,
            sampler=test_subsampler,
        )

        model = KGEClassificationModel(
            num_classes=len(triples_df["class"].unique()),
            class_weights=weights,
            lr=lr,
        )

        # Initialize pytorch lighting Trainer for the KG baseline model
        trainer = pl.Trainer(max_epochs=epochs)
        # Fit on training split
        trainer.fit(model, train_dataloader=trainloader)
        # Predict on test split
        test_results = trainer.test(model, test_dataloaders=testloader)

        # Append f1 score per split based on the macro average
        f1_scores.append(test_results[0]["mean_test_f1"])

    # Log mean and std f1-scores from the cross validation procedure (average and std across all splits)
    logger.info(f'Mean f1-score: {np.mean(f1_scores)}')
    logger.info(f'Std f1-score: {np.std(f1_scores)}')

    # Return the final f1 score mean and the std for all CV folds
    return {"f1_score_mean": np.mean(f1_scores), "f1_score_std": float(np.std(f1_scores))}


if __name__ == "__main__":
    run_kg_baseline_classification_cv()
