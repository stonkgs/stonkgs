# -*- coding: utf-8 -*-

"""NLP baseline model on the fine-tuning classification task, assuming the model embeddings are pre-trained."""

import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from ..constants import DUMMY_EXAMPLE_TRIPLES, LOG_DIR, NLP_BL_OUTPUT_DIR, NLP_MODEL_TYPE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class INDRAEvidenceDataset(torch.utils.data.Dataset):
    """Custom Dataset class for INDRA data."""

    def __init__(self, encodings, labels):
        """Initialize INDRA Dataset based on token embeddings for each text evidence."""
        # Assumes that the labels are numerically encoded
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """Return data entries (text evidences) for given indices."""
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.labels)


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

    return [{"train_idx": train_idx, "test_idx": test_idx} for train_idx, test_idx in skf.split(data_no_labels, labels)]


def run_sequence_classification_cv(
    data_path: str = DUMMY_EXAMPLE_TRIPLES,
    sep: str = "\t",
    model_type: str = NLP_MODEL_TYPE,
    logging_dir: str = LOG_DIR,
    output_dir: str = NLP_BL_OUTPUT_DIR,
    label_column_name: str = "class",
    text_data_column_name: str = "evidence",
    epochs: int = 2,
) -> Dict:
    """Run cross-validation for the sequence classification task."""
    # Get data splits
    indra_data = pd.read_csv(data_path, sep=sep)
    train_test_splits = get_train_test_splits(indra_data)

    # Get text evidences and labels
    evidences_text, labels_str = indra_data[text_data_column_name], indra_data[label_column_name]
    # Numerically encode labels
    unique_tags = set(label for label in labels_str)
    tag2id = {label: number for number, label in enumerate(unique_tags)}
    labels = pd.Series([int(tag2id[label]) for label in labels_str])

    # Initialize the f1-score
    f1_scores = []

    for indices in train_test_splits:
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=len(unique_tags))

        # Encode all text evidences, pad and truncate to max_seq_len
        train_evidences = tokenizer(evidences_text[indices["train_idx"]].tolist(), truncation=True, padding=True)
        test_evidences = tokenizer(evidences_text[indices["test_idx"]].tolist(), truncation=True, padding=True)
        train_labels = labels[indices["train_idx"]].tolist()
        test_labels = labels[indices["test_idx"]].tolist()
        train_dataset = INDRAEvidenceDataset(encodings=train_evidences, labels=train_labels)
        test_dataset = INDRAEvidenceDataset(encodings=test_evidences, labels=test_labels)

        # Note that due to the randomization in the batches, the training/evaluation is slightly different every time
        training_args = TrainingArguments(
            # label_names
            output_dir=output_dir,
            num_train_epochs=epochs,  # total number of training epochs
            logging_dir=logging_dir,  # directory for storing logs
            logging_steps=100,
            # TODO: Implement report_to = ["mlflow"] later on
            do_train=True,
            do_predict=True,
        )

        # Initialize Trainer based on the training dataset
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        # Train
        trainer.train()

        # Make predictions for the test dataset
        predictions = trainer.predict(test_dataset=test_dataset).predictions
        predicted_labels = np.argmax(predictions, axis=1)
        # Use macro average for now
        f1_scores.append(f1_score(test_labels, predicted_labels, average="macro"))

    logger.info(f'Mean f1-score: {np.mean(f1_scores)}')
    logger.info(f'Std f1-score: {np.std(f1_scores)}')

    return {"f1_score_mean": np.mean(f1_scores), "f1_score_std": np.std(f1_scores)}


if __name__ == "__main__":
    # Set the huggingface environment variable for tokenizer parallelism to false
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    run_sequence_classification_cv()
