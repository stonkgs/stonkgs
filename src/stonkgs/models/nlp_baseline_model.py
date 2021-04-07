# -*- coding: utf-8 -*-

"""NLP baseline model on the fine-tuning classification task, assuming the model embeddings are pre-trained."""

import logging
import os
from typing import Dict, List

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
from transformers.integrations import MLflowCallback

from ..constants import DUMMY_EXAMPLE_TRIPLES, MLFLOW_TRACKING_URI, NLP_BL_OUTPUT_DIR, NLP_MODEL_TYPE

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NestedMLFlowCallback(TrainerCallback):
    """A :class:`~transformers.TrainerCallback` that sends the logs to `MLflow <https://www.mlflow.org/>`__."""

    def __init__(self):
        """Set up some mlflow parameters."""
        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        self._MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH

        self._initialized = False
        self._log_artifacts = False
        self._ml_flow = mlflow

    def setup(self, args, state, model):
        """Set up the optional MLflow integration."""
        log_artifacts = os.getenv("HF_MLFLOW_LOG_ARTIFACTS", "FALSE").upper()
        if log_artifacts in {"TRUE", "1"}:
            self._log_artifacts = True
        if state.is_world_process_zero:
            # Start nested runs here
            self._ml_flow.start_run(nested=True)
            combined_dict = args.to_dict()
            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            # Remove params that are too long for MLflow
            for name, value in list(combined_dict.items()):
                # Internally, all values are converted to str in MLflow
                if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                    logger.warning(
                        "Trainer is attempting to log a value of ",
                        f'"{value}" for key "{name}" as a parameter. '
                        "MLflow's log_param() only accepts values no longer than ",
                        "250 characters so we dropped this attribute.",
                    )
                    del combined_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(combined_dict.items())
            for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
                self._ml_flow.log_params(dict(combined_dict_items[i: i + self._MAX_PARAMS_TAGS_PER_BATCH]))
        self._initialized = True

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Set up the callback when the training procedure begins."""
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        """Perform logging actions."""
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self._ml_flow.log_metric(k, v, step=state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of ",
                        f'"{v}" of type {type(v)} for key "{k}" as a metric. ',
                        "MLflow's log_metric() only accepts float and ",
                        "int types so we dropped this attribute.",
                    )

    def on_train_end(self, args, state, control, **kwargs):
        """Log artifacts after training."""
        if self._initialized and state.is_world_process_zero:
            if self._log_artifacts:
                logger.info("Logging artifacts. This may take time.")
                self._ml_flow.log_artifacts(args.output_dir)

    def __del__(self):
        """Kill run."""
        if self._ml_flow.active_run is not None:
            self._ml_flow.end_run()


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
    output_dir: str = NLP_BL_OUTPUT_DIR,
    logging_uri_mlflow: str = MLFLOW_TRACKING_URI,
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

    # End previous run
    mlflow.end_run()
    # Initialize mlflow run, set tracking URI to use the same experiment for all runs,
    # so that one can compare them
    mlflow.set_tracking_uri(logging_uri_mlflow)
    mlflow.set_experiment('NLP Baseline for STonKGs')

    # Start a parent run so that all CV splits are tracked as nested runs
    mlflow.start_run(run_name='Parent Run')

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

        # Note that due to the randomization in the batches, the training/evaluation is slightly
        # different every time
        training_args = TrainingArguments(
            # label_names
            output_dir=output_dir,
            num_train_epochs=epochs,  # total number of training epochs
            logging_steps=100,
            # report_to=["mlflow"],  # log via mlflow
            do_train=True,
            do_predict=True,
        )

        # Initialize Trainer based on the training dataset
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        # Remove the default MLflowCallback
        trainer.remove_callback(MLflowCallback)
        # Add the custom nested callback
        trainer.add_callback(NestedMLFlowCallback())
        # Train
        trainer.train()

        # Make predictions for the test dataset
        predictions = trainer.predict(test_dataset=test_dataset).predictions
        predicted_labels = np.argmax(predictions, axis=1)
        # Use macro average for now
        f1_scores.append(f1_score(test_labels, predicted_labels, average="macro"))

    logger.info(f'Mean f1-score: {np.mean(f1_scores)}')
    logger.info(f'Std f1-score: {np.std(f1_scores)}')

    # End parent run
    mlflow.end_run()

    return {"f1_score_mean": np.mean(f1_scores), "f1_score_std": np.std(f1_scores)}


if __name__ == "__main__":
    # Set the huggingface environment variable for tokenizer parallelism to false
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    run_sequence_classification_cv()
