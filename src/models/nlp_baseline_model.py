# -*- coding: utf-8 -*-

"""NLP baseline model on the fine-tuning classification task, assuming the model embeddings are pre-trained."""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import List, Dict
import numpy as np
import torch
from ..constants import NLP_MODEL_TYPE, DUMMY_EXAMPLE_TRIPLES, MODELS_DIR
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, \
    PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from sklearn.metrics import f1_score


class INDRADataset(torch.utils.data.Dataset):
    """Custom Dataset class for INDRA data."""
    def __init__(self, encodings, labels):
        # Assumes that the labels are numerically encoded
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


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
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=False)

    return [{"train_idx": train_idx, "test_idx": test_idx} for train_idx, test_idx in skf.split(data_no_labels, labels)]


def run_sequence_classification_cv(
    data_path: str = DUMMY_EXAMPLE_TRIPLES,
    sep: str = "\t",
    model_type: str = NLP_MODEL_TYPE,
    logging_dir: str = MODELS_DIR,
    label_column_name: str = "class",
    text_data_column_name: str = "evidence",
    epochs: int = 2,
) -> Dict:
    """Runs cross-validation for the sequence classification task."""
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForSequenceClassification.from_pretrained(model_type)

    # Get data splits
    indra_data = pd.read_csv(data_path, sep=sep)
    train_test_splits = get_train_test_splits(indra_data)

    # Get text evidences and labels
    evidences_text, labels_str = indra_data[text_data_column_name], indra_data[label_column_name]
    # Numerically encode labels
    unique_tags = set(tag for doc in labels_str for tag in doc)
    tag2id = {label: number for number, label in enumerate(unique_tags)}
    labels = labels_str.map(tag2id)

    # Encode all text evidences, pad and truncate to max_seq_len
    encoded_text = tokenizer(evidences_text, truncation=True, padding=True)

    f1_scores = []

    for indices in train_test_splits:
        train_evidences, train_labels = encoded_text[indices["train_idx"]], labels[indices["train_idx"]]
        train_dataset = INDRADataset(train_evidences, train_labels)

        test_evidences, test_labels = encoded_text[indices["test_idx"]], labels[indices["test_idx"]]

        # Note that due to the randomization in the batches, the training/evaluation is slightly different every time
        training_args = TrainingArguments(
            num_train_epochs=epochs,  # total number of training epochs
            # TODO: specify better log directory
            logging_dir=logging_dir,  # directory for storing logs
            logging_steps=100,
            # TODO: Implement report_to = ["mlflow"] later on
            do_train=True,
            do_predict=True
        )

        # Initialize Trainer based on the training dataset
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset
        )

        # Make predictions for the test dataset
        predictions = np.argmax(trainer.predict(test_dataset=test_evidences).predictions, axis=1)
        # Use macro average for now
        f1_scores.append(f1_score(test_labels, predictions, average="macro"))

    return {"f1_score_mean": np.mean(f1_scores), "f1_score_std": np.std(f1_scores)}


if __name__ == "__main__":
    run_sequence_classification_cv()
