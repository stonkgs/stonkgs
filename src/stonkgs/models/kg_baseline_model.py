# -*- coding: utf-8 -*-
"""
KG baseline model on the fine-tuning classification task, assuming the model embeddings are pre-trained.

Run with:
python -m src.stonkgs.models.kg_baseline_model
"""

import logging
import os
from collections import Counter
from typing import Dict, List, Optional

import click
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from stonkgs.constants import (
    CELL_LINE_DIR,
    CELL_TYPE_DIR,
    DISEASE_DIR,
    EMBEDDINGS_PATH,
    KG_BL_OUTPUT_DIR,
    LOCATION_DIR,
    MLFLOW_FINETUNING_TRACKING_URI,
    ORGAN_DIR,
    RANDOM_WALKS_PATH,
    RELATION_TYPE_DIR,
    SPECIES_DIR,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# Disable alembic info
logging.getLogger("alembic").setLevel(logging.WARNING)


class KGEClassificationModel(pl.LightningModule):
    """KGE baseline model."""

    def __init__(
        self,
        num_classes,
        class_weights,
        d_in: int = 768,
        lr: float = 1e-3,
    ):
        """Initialize the components of the KGE based classification model.

        :param num_classes: number of classes
        :param class_weights: class weights
        :param d_in: dimensions
        :param lr: learning rate

        The model consists of
        1) "Max-Pooling" (embedding-dimension-wise max)
        2) Dropout
        3) Linear layer (d_in x num_classes)
        4) Softmax
        (Not part of the model, but of the class: class_weights for the cross_entropy function)
        """
        super(KGEClassificationModel, self).__init__()

        # Model architecture
        self.pooling = torch.max
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(d_in, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

        # Other class-specific parameters
        # Class weights for CE loss
        self.class_weights = torch.tensor(class_weights)
        # Learning rate
        self.lr = lr

        # Add an attribute to save the predictions later on
        self.predicted_labels = []

        # Log the additional parameters
        self.log_dict({"num_classes": num_classes, "class_weights": class_weights, "lr": lr})

    def forward(self, x):
        """Perform forward pass consisting of pooling (dimension-wise max), and a linear layer followed by softmax.

        :param x: embedding sequences (random walk embeddings) for the given triples
        :return: class probabilities for the given triples
        """
        h_pooled = self.pooling(x, dim=1).values
        dropout_output = self.dropout(h_pooled)
        linear_output = self.linear(dropout_output)
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

        # Log loss at each training step
        self.log('loss', loss.item(), on_step=True)

        return loss

    def test_step(self, batch, batch_nb):
        """Perform one test step on a given batch and return the weighted-averaged f1 score over all batches."""
        test_inputs, test_labels = batch
        test_class_probs = self.forward(test_inputs)
        # Class probabilities to class labels
        test_predictions = torch.argmax(test_class_probs, dim=1)
        # logger.info(f'Predicted labels: {test_predictions}')

        return {'labels': test_labels, 'predictions': test_predictions}

    def test_epoch_end(self, outputs):
        """Return average and std weighted-averaged f1-score over all batches."""
        # Get the weighted-averaged f1-score
        all_labels = torch.cat([x['labels'] for x in outputs])
        all_predictions = torch.cat([x['predictions'] for x in outputs])
        test_f1 = f1_score(all_labels, all_predictions, average="weighted")

        # Log the final f1 score
        self.log('f1_score_weighted', test_f1)

        # Use this (weird) fix to save the predicted labels
        self.predicted_labels = all_predictions

        return {'test_f1': test_f1}


class INDRAEntityDataset(torch.utils.data.Dataset):
    """Custom dataset class for INDRA data."""

    def __init__(self, embedding_dict, random_walk_dict, sources, targets, labels, max_len=254):
        """Initialize INDRA Dataset based on random walk embeddings for 2 nodes in each triple."""
        self.max_length = max_len
        # Two entities (source, target) of each triple
        self.sources = sources
        self.targets = targets
        # Initialize dictionary of node name -> embedding vector
        self.embedding_dict = embedding_dict
        # Add the null vector to the embedding dict for "out-of-vocabulary" nodes
        self.embedding_dict[-1] = np.zeros(np.shape(next(iter(self.embedding_dict.values()))))
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
            # Get the random walk sequences
            random_walk_source = self.random_walk_dict[source]
            random_walk_target = self.random_walk_dict[target]

            # 2. Concatenate and the random walks
            # The total random walk has the length max_length. Therefore its split half into the random walk of source
            # and half target.
            random_walk = random_walk_source.tolist() + random_walk_target.tolist()  # noqa: N400
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

    # Implement non-stratified train/test splits with no validation split
    # It is shuffled deterministically (determined by random_seed)
    skf = KFold(n_splits=n_splits, random_state=random_seed, shuffle=True)

    # Return a list of dictionaries for train and test indices
    return [{"train_idx": train_idx, "test_idx": test_idx} for train_idx, test_idx in skf.split(data_no_labels, labels)]


def run_kg_baseline_classification_cv(
    triples_path: str,
    embedding_path: str = EMBEDDINGS_PATH,
    random_walks_path: str = RANDOM_WALKS_PATH,
    logging_uri_mlflow: Optional[str] = MLFLOW_FINETUNING_TRACKING_URI,
    n_splits: int = 5,
    epochs: int = 100,
    train_batch_size: int = 8,
    test_batch_size: int = 64,
    lr: float = 1e-3,
    label_column_name: str = 'class',
    log_steps: int = 500,
) -> Dict[str, float]:
    """Run KG baseline classification."""
    # Step 1. load the tsv file with the annotation types you want to test and make the splits
    triples_df = pd.read_csv(
        triples_path,
        sep='\t',
        usecols=[
            'source',
            'target',
            label_column_name,
        ],
    )

    # Prepare embeddings and random walks
    embeddings_dict = _prepare_df(embedding_path)
    random_walks_dict = _prepare_df(random_walks_path)

    # Filter out any triples that contain a node that is not in the embeddings_dict
    original_length = len(triples_df)
    triples_df = triples_df[
        triples_df['source'].isin(embeddings_dict.keys()) & triples_df['target'].isin(embeddings_dict.keys())
    ]
    new_length = len(triples_df)
    logger.info(f'{original_length - new_length} out of {original_length} triples are left out because they contain '
                f'nodes which are not present in the pre-training data')

    # Numerically encode labels
    unique_tags = set(label for label in triples_df[label_column_name])
    tag2id = {label: number for number, label in enumerate(unique_tags)}
    id2tag = {value: key for key, value in tag2id.items()}

    # Get labels
    labels = pd.Series([int(tag2id[label]) for label in triples_df[label_column_name]])

    # Get the train/test split indices
    train_test_splits = get_train_test_splits(triples_df, n_splits=n_splits, label_column_name=label_column_name)

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

    mlflow.set_tracking_uri(logging_uri_mlflow)
    mlflow.set_experiment('KG Baseline for STonKGs')

    mlflow.pytorch.autolog()

    # Initialize a dataframe for all the predicted labels
    result_df = pd.DataFrame()

    # Train and test the model in a cv setting
    for idx, indices in enumerate(train_test_splits):
        # Sample elements randomly from a given list of ids, no replacement
        train_subsampler = torch.utils.data.SubsetRandomSampler(indices["train_idx"])
        test_subsampler = torch.utils.data.SubsetRandomSampler(indices["test_idx"])
        # CE class weights for the model based on training data class distribution,
        # based on the class counts (Inverse Number of Samples, INS)
        weights = [
            1 / len([i
                     for i in triples_df.iloc[indices["train_idx"], :][label_column_name]
                     # note that we only employ train idx
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
            num_classes=len(triples_df[label_column_name].unique()),
            class_weights=weights,
            lr=lr,
        )

        # Initialize pytorch lighting Trainer for the KG baseline model
        trainer = pl.Trainer(
            max_epochs=epochs,
            log_every_n_steps=log_steps,
        )

        # Train and predict in a separate run for each split
        with mlflow.start_run():
            # Fit on training split
            trainer.fit(model, train_dataloader=trainloader)
            # Predict on test split
            test_results = trainer.test(model, test_dataloaders=testloader)

            # Save the predicted + true labels
            partial_result_df = pd.DataFrame(
                {'split': idx,
                 'index': indices["test_idx"].tolist(),
                 'predicted_label': model.predicted_labels.tolist(),
                 'true_label': testloader.dataset.labels.tolist(),
                 }
            )
            result_df = result_df.append(
                partial_result_df,
                ignore_index=True,
            )

            # Log some details about the datasets used in training and testing
            mlflow.log_param('label dict', str(tag2id))
            mlflow.log_param('training dataset size', str(len(trainloader.dataset)))
            mlflow.log_param('training class dist', str(Counter(trainloader.dataset.labels)))
            mlflow.log_param('test dataset size', str(len(testloader.dataset)))
            mlflow.log_param('test class dist', str(Counter(testloader.dataset.labels)))

        # Append f1 score per split based on the weighted average
        f1_scores.append(test_results[0]["test_f1"])

    # Map the labels in the result df back to their original names
    result_df = result_df.replace({'predicted_label': id2tag, 'true_label': id2tag})
    # Save the result_df
    result_df.to_csv(os.path.join(KG_BL_OUTPUT_DIR, 'predicted_labels_kg_df.tsv'), index=False, sep="\t")

    # Save the last model
    trainer.save_checkpoint(os.path.join(KG_BL_OUTPUT_DIR, 'kg_baseline.ckpt'))

    # Log the mean and std f1 score from the cross validation procedure to mlflow
    with mlflow.start_run():
        mlflow.log_metric('f1_score_mean', np.mean(f1_scores))
        mlflow.log_metric('f1_score_std', np.std(f1_scores))

        # Log the task name as well
        mlflow.log_param('task name', str(os.path.split(triples_path)[-1]) + " (" + label_column_name + ")")

        # Also log how many triples were left out
        mlflow.log_param('original no. of triples', original_length)
        mlflow.log_param('no. of left out triples', original_length - new_length)

    # Log mean and std f1-scores from the cross validation procedure (average and std across all splits) to the
    # standard logger
    logger.info(f'Mean f1-score: {np.mean(f1_scores)}')
    logger.info(f'Std f1-score: {np.std(f1_scores)}')

    # Return the final f1 score mean and the std for all CV folds
    return {"f1_score_mean": np.mean(f1_scores), "f1_score_std": float(np.std(f1_scores))}


@click.command()
@click.option('-e', '--epochs', default=5, help='Number of epochs', type=int)
@click.option('--lr', default=1e-3, help='Learning rate', type=float)
@click.option('--logging_dir', default=MLFLOW_FINETUNING_TRACKING_URI, help='Mlflow logging/tracking URI', type=str)
@click.option('--log_steps', default=500, help='Number of steps between each log', type=int)
@click.option('--batch_size', default=8, help='Batch size', type=int)
def run_all_fine_tuning_tasks(
    epochs: int = 5,
    log_steps: int = 500,
    lr: float = 1e-3,
    logging_dir: Optional[str] = MLFLOW_FINETUNING_TRACKING_URI,
    batch_size: int = 8,
):
    """Run all fine-tuning tasks at once."""
    # Run the 6 annotation type tasks
    # Specify all directories and file names
    directories = [
        CELL_LINE_DIR,
        CELL_TYPE_DIR,
        DISEASE_DIR,
        LOCATION_DIR,
        ORGAN_DIR,
        SPECIES_DIR,
        RELATION_TYPE_DIR,
        RELATION_TYPE_DIR,
    ]
    file_names = [
        'cell_line_filtered.tsv',
        'cell_type_filtered.tsv',
        'disease_filtered.tsv',
        'location_filtered.tsv',
        'organ_filtered.tsv',
        'species_filtered.tsv',
        'relation_type.tsv',
        'relation_type.tsv',
    ]
    # Specify the column names of the target variable
    column_names = ['class'] * 6 + ['interaction'] + ['polarity']

    for directory, file, column_name in zip(directories, file_names, column_names):
        # Run each of the eight classification tasks
        run_kg_baseline_classification_cv(
            triples_path=os.path.join(directory, file),
            label_column_name=column_name,
            logging_uri_mlflow=logging_dir,
            epochs=epochs,
            lr=lr,
            log_steps=log_steps,
            train_batch_size=batch_size,
        )
        logger.info(f'Finished the {file} task (with column name {column_name})')


if __name__ == "__main__":
    # Run all fine-tuning classification tasks at once
    run_all_fine_tuning_tasks()
