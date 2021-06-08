# -*- coding: utf-8 -*-

"""NLP baseline model on the fine-tuning classification task, assuming the model embeddings are pre-trained."""

import logging
import os
from collections import Counter
from typing import Dict, List, Optional

import click
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from stonkgs.constants import (
    CELL_LINE_DIR,
    CELL_TYPE_DIR,
    DISEASE_DIR,
    EMBEDDINGS_PATH,
    LOCATION_DIR,
    MLFLOW_FINETUNING_TRACKING_URI,
    NLP_BL_OUTPUT_DIR,
    NLP_MODEL_TYPE,
    ORGAN_DIR,
    RELATION_TYPE_DIR,
    SPECIES_DIR,
    STONKGS_OUTPUT_DIR,
)
from stonkgs.data.indra_for_pretraining import _prepare_df

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# Disable alembic info
logging.getLogger("alembic").setLevel(logging.WARNING)


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

    # Implement non-stratified train/test splits with no validation split
    # It is shuffled deterministically (determined by random_seed)
    skf = KFold(n_splits=n_splits, random_state=random_seed, shuffle=True)

    return [{"train_idx": train_idx, "test_idx": test_idx} for train_idx, test_idx in skf.split(data_no_labels, labels)]


def run_nlp_baseline_classification_cv(
    train_data_path: str,
    sep: Optional[str] = "\t",
    model_type: str = NLP_MODEL_TYPE,
    output_dir: Optional[str] = NLP_BL_OUTPUT_DIR,
    logging_uri_mlflow: Optional[str] = MLFLOW_FINETUNING_TRACKING_URI,
    label_column_name: str = "class",
    text_data_column_name: str = "evidence",
    epochs: Optional[int] = 10,
    log_steps: int = 500,
    lr: float = 5e-5,
    batch_size: int = 16,
    gradient_accumulation: int = 1,
    task_name: str = '',
    embedding_path: str = EMBEDDINGS_PATH,
) -> Dict:
    """Run cross-validation for the sequence classification task."""
    # Get data splits
    indra_data = pd.read_csv(train_data_path, sep=sep)
    # TODO: leave it out later on?
    # Filter out any triples that contain a node that is not in the embeddings_dict
    embeddings_dict = _prepare_df(embedding_path)
    original_length = len(indra_data)
    indra_data = indra_data[
        indra_data['source'].isin(embeddings_dict.keys()) & indra_data['target'].isin(embeddings_dict.keys())
    ].reset_index(drop=True)
    new_length = len(indra_data)
    logger.info(f'{original_length - new_length} out of {original_length} triples are left out because they contain '
                f'nodes which are not present in the pre-training data')

    train_test_splits = get_train_test_splits(indra_data, label_column_name=label_column_name)

    # Get text evidences and labels
    evidences_text, labels_str = indra_data[text_data_column_name], indra_data[label_column_name]
    # Numerically encode labels
    unique_tags = set(label for label in labels_str)
    tag2id = {label: number for number, label in enumerate(unique_tags)}
    id2tag = {value: key for key, value in tag2id.items()}
    labels = pd.Series([int(tag2id[label]) for label in labels_str])

    # Initialize the f1-score
    f1_scores = []

    # End previous run
    mlflow.end_run()
    # Initialize mlflow run, set tracking URI to use the same experiment for all runs,
    # so that one can compare them
    mlflow.set_tracking_uri(logging_uri_mlflow)
    mlflow.set_experiment('NLP Baseline for STonKGs')

    # Start a parent run so that all CV splits are tracked as nested runs
    # mlflow.start_run(run_name='Parent Run')

    # Initialize a dataframe for all the predicted labels
    result_df = pd.DataFrame()

    for idx, indices in enumerate(train_test_splits):
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

        # Note that due to the randomization in the batches, the training/evaluation is slightly
        # different every time
        training_args = TrainingArguments(
            # label_names
            output_dir=output_dir,
            num_train_epochs=epochs,  # total number of training epochs
            logging_steps=log_steps,
            learning_rate=lr,
            report_to=["mlflow"],  # log via mlflow
            do_train=True,
            do_predict=True,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
        )

        # Initialize Trainer based on the training dataset
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        # Train
        trainer.train()

        # Log some details about the datasets used in training and testing
        mlflow.log_param('label dict', str(tag2id))
        mlflow.log_param('training dataset size', str(len(train_labels)))
        mlflow.log_param('training class dist', str(Counter(train_labels)))
        mlflow.log_param('test dataset size', str(len(test_labels)))
        mlflow.log_param('test class dist', str(Counter(test_labels)))

        # Make predictions for the test dataset
        predictions = trainer.predict(test_dataset=test_dataset).predictions
        predicted_labels = np.argmax(predictions, axis=1)
        logger.info(f'Predicted labels: {predicted_labels}')

        # Save the predicted + true labels
        partial_result_df = pd.DataFrame(
            {'split': idx,
             'index': indices["test_idx"].tolist(),
             'predicted_label': predicted_labels.tolist(),
             'true_label': test_labels,
             }
        )
        result_df = result_df.append(
            partial_result_df,
            ignore_index=True,
        )

        # Use weighted average
        f1_sc = f1_score(test_labels, predicted_labels, average="weighted")
        f1_scores.append(f1_sc)

        # Log the final f1 score of the split
        mlflow.log_metric('f1_score_weighted', f1_sc)

    # Log mean and std f1-scores from the cross validation procedure (average and std across all splits) to the
    # standard logger
    logger.info(f'Mean f1-score: {np.mean(f1_scores)}')
    logger.info(f'Std f1-score: {np.std(f1_scores)}')

    # Map the labels in the result df back to their original names
    result_df = result_df.replace({'predicted_label': id2tag, 'true_label': id2tag})
    # Save the result_df
    result_df.to_csv(
        os.path.join(NLP_BL_OUTPUT_DIR, 'predicted_labels_nlp_' + task_name + 'df.tsv'),
        index=False,
        sep="\t",
    )

    # Save the last model
    trainer.save_model(output_dir=NLP_BL_OUTPUT_DIR)

    # End the previous run
    mlflow.end_run()

    # Log the mean and std f1 score from the cross validation procedure to mlflow
    with mlflow.start_run():
        # Log the task name as well
        mlflow.log_param('task name', task_name)
        mlflow.log_metric('f1_score_mean', np.mean(f1_scores))
        mlflow.log_metric('f1_score_std', np.std(f1_scores))

    # End parent run
    # mlflow.end_run()

    return {"f1_score_mean": np.mean(f1_scores), "f1_score_std": np.std(f1_scores)}


@click.command()
@click.option('-e', '--epochs', default=5, help='Number of epochs', type=int)
@click.option('--lr', default=5e-5, help='Learning rate', type=float)
@click.option('--logging_dir', default=MLFLOW_FINETUNING_TRACKING_URI, help='Mlflow logging/tracking URI', type=str)
@click.option('--log_steps', default=500, help='Number of steps between each log', type=int)
@click.option('--output_dir', default=STONKGS_OUTPUT_DIR, help='Output directory', type=str)
@click.option('--batch_size', default=8, help='Batch size used in fine-tuning', type=int)
@click.option('--gradient_accumulation_steps', default=1, help='Gradient accumulation steps', type=int)
def run_all_fine_tuning_tasks(
    epochs: int = 5,
    log_steps: int = 500,
    lr: float = 5e-5,
    output_dir: str = STONKGS_OUTPUT_DIR,
    logging_dir: Optional[str] = MLFLOW_FINETUNING_TRACKING_URI,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
):
    """Run all fine-tuning tasks at once."""
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
    task_names = [
        'cell_line',
        'cell_type',
        'disease',
        'location',
        'organ',
        'species',
        'interaction',
        'polarity'
    ]
    # Specify the column names of the target variable
    column_names = ['class'] * 6 + ['interaction'] + ['polarity']

    for directory, file, column_name, task_name in zip(directories, file_names, column_names, task_names):
        # Run each of the eight classification tasks
        run_nlp_baseline_classification_cv(
            train_data_path=os.path.join(directory, file),
            output_dir=output_dir,
            logging_uri_mlflow=logging_dir,
            epochs=epochs,
            log_steps=log_steps,
            lr=lr,
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation_steps,
            label_column_name=column_name,
            task_name=task_name,
        )
        logger.info(f'Finished the {task_name} task')


if __name__ == "__main__":
    # Set the huggingface environment variable for tokenizer parallelism to false
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Run all classification tasks
    run_all_fine_tuning_tasks()
