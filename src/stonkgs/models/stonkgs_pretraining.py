# -*- coding: utf-8 -*-

"""Script for running the pre-training procedure of STonKGs."""

import logging
import os
from typing import Optional

import click
import mlflow
import pandas as pd
from pyarrow import csv, list_, int16
# import torch.autograd.profiler as profiler
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from stonkgs.constants import (
    EMBEDDINGS_PATH,
    MLFLOW_TRACKING_URI,
    NLP_MODEL_TYPE,
    PRETRAINING_PREPROCESSED_DF_PATH,
    STONKGS_PRETRAINING_DIR,
)
from stonkgs.models.kg_baseline_model import _prepare_df
from stonkgs.models.stonkgs_model import STonKGsForPreTraining

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# Disable alembic info
logging.getLogger("alembic").setLevel(logging.WARNING)


def _load_pre_training_data(
    pretraining_preprocessed_path: str = PRETRAINING_PREPROCESSED_DF_PATH,
    dataset_format: str = 'torch',
) -> Dataset:
    """Create a pytorch dataset based on a preprocessed dataframe for the pretraining dataset."""
    # Load the pickled preprocessed dataframe, only select the relevant columns
    pretraining_preprocessed_df = pd.read_pickle(pretraining_preprocessed_path)[[
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "masked_lm_labels",
        "ent_masked_lm_labels",
        "next_sentence_labels",
    ]]
    logger.info('Finished reading the pickled dataframe')
    pretraining_dataset = Dataset.from_pandas(pretraining_preprocessed_df)
    del pretraining_preprocessed_df

    """"# Try a different way of loading the data
    csv_converter = csv.ConvertOptions(column_types={
        'attention_mask': list_(int16()),
        'token_type_ids': list_(int16()),
        'masked_lm_labels': list_(int16()),
        'ent_masked_lm_labels': list_(int16()),
        'next_sentence_labels': list_(int16()),
    })
    pretraining_dataset = load_dataset(
        'csv',
        data_files=pretraining_preprocessed_path,
        delimiter="\t",
        # convert_options=csv_converter,
    )

    # Correct the list things
    list_mapper = lambda x: [int(y) for y in x.strip('[]').split(', ')]
    pretraining_dataset.map()"""

    # Do not put the dataset on the GPU even if possible, it is only stealing GPU space, use the dataloader instead
    # Putting it on the GPU might only be worth it if 4+ GPUs are used
    pretraining_dataset.set_format(dataset_format)
    logger.info('Finished loading the pretraining dataset')

    return pretraining_dataset


@click.command()
@click.option('-b', '--batch_size', default=8, help='Batch size for training (per device)', type=int)
@click.option('--lr', default=1e-4, help='Learning rate', type=float)
@click.option('--dataloader_num_workers', default=8, help='Number of dataloader workers', type=int)
@click.option('--gradient_accumulation_steps', default=1, help='Number of gradient accumulation steps', type=int)
@click.option('--logging_dir', default=MLFLOW_TRACKING_URI, help='Mlflow logging/tracking URI', type=str)
@click.option('--logging_steps', default=100, help='Logging interval', type=int)
@click.option('-m', '--max_steps', default=200, help='Number of training steps', type=int)
@click.option('--overwrite_output_dir', default=False, help='Whether to override the output dir or not', type=bool)
@click.option('--save_limit', default=5, help='Maximum number of saved models/checkpoints', type=int)
@click.option('--save_steps', default=5000, help='Checkpointing interval', type=int)
@click.option('--training_dir', default=STONKGS_PRETRAINING_DIR, help='Whether to override the output dir', type=str)
def pretrain_stonkgs(
    batch_size: int = 8,
    lr: float = 1e-4,
    dataloader_num_workers: int = 8,  # empirically determined value, I'm open to changing it :)
    gradient_accumulation_steps: int = 1,
    logging_dir: Optional[str] = MLFLOW_TRACKING_URI,
    logging_steps: int = 100,
    max_steps: int = 10000,
    overwrite_output_dir: bool = False,
    save_limit: int = 5,
    save_steps: int = 5000,
    training_dir: str = STONKGS_PRETRAINING_DIR,
):
    """Run the pre-training procedure for the STonKGs model based on the transformers Trainer and TrainingArguments."""
    # Part of this code is taken from
    # https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py

    # Logging with mlflow
    # End previous run
    mlflow.end_run()
    # Initialize mlflow run, set tracking URI to use the same experiment for all runs,
    # so that one can compare them
    mlflow.set_tracking_uri(logging_dir)
    mlflow.set_experiment('STonKGs Pre-Training')

    # Initialize the STonKGs model
    kg_embed_dict = _prepare_df(EMBEDDINGS_PATH)
    stonkgs_model = STonKGsForPreTraining(NLP_MODEL_TYPE, kg_embed_dict)

    # Add the huggingface accelerator
    accelerator = Accelerator()
    # Use the device given by the `accelerator` object and put the model on there
    device = accelerator.device
    stonkgs_model.to(device)

    # Initialize the dataset
    pretraining_data = _load_pre_training_data()

    # Accelerate the model
    stonkgs_model = accelerator.prepare(stonkgs_model)

    # Initialize the TrainingArguments
    training_args = TrainingArguments(
        output_dir=training_dir,
        overwrite_output_dir=overwrite_output_dir,
        do_train=True,
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,  # Use max_steps rather than num_training_epochs
        learning_rate=lr,  # Default is to use that lr with a linear scheduler
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_limit,
        report_to=['mlflow'],
        # Make the dataloader faster by using pinning the memory and using multiple workers
        dataloader_pin_memory=True,
        dataloader_num_workers=dataloader_num_workers,
        # Effectively increase the batch size by gradient acc: batch_size = old_batch_size x grad_acc_steps
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. ",
                "Use --overwrite_output_dir to overcome.",
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change ",
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.",
            )

    # Initialize the Trainer
    trainer = Trainer(
        model=stonkgs_model,
        args=training_args,
        train_dataset=pretraining_data,
    )
    # And train STonKGs to the moon
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    # Log the number of pre-training samples
    metrics["train_samples"] = len(pretraining_data)
    # Log all metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == '__main__':
    # Run the pre-training procedure, overwrite the output dir for now (since we're only working with dummy data)
    # Distinguish between local and cluster execution by using different values (steps, batch size etc.) in the
    # CLI arguments
    pretrain_stonkgs()

    # (Optional) examine the runtime
    # with profiler.profile() as prof:
    #    with profiler.record_function("model_inference"):
    #        pretrain_stonkgs(overwrite_output_dir=True, max_steps=3)

    # logger.info(prof.key_averages().table(sort_by="cpu_time_total"))  # or replace by something like gpu_time_total
