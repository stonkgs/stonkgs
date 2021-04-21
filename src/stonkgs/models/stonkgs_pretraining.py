# -*- coding: utf-8 -*-

"""Script for running the pre-training procedure of STonKGs."""

import logging
import os
from typing import Optional

import mlflow
import pandas as pd
# import torch.autograd.profiler as profiler
from accelerate import Accelerator
from datasets import Dataset
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


def _load_pre_training_data(
    pretraining_preprocessed_path: str = PRETRAINING_PREPROCESSED_DF_PATH,
    dataset_format: str = 'torch',
) -> Dataset:
    """Create a pytorch dataset based on a preprocessed dataframe for the pretraining dataset."""
    # Load the pickled preprocessed dataframe
    pretraining_preprocessed_df = pd.read_pickle(pretraining_preprocessed_path)
    pretraining_dataset = Dataset.from_pandas(pretraining_preprocessed_df)
    # Do not put the dataset on the GPU even if possible, it is only stealing GPU space, use the dataloader instead
    # Putting it on the GPU might only be worth it if 4+ GPUs are used
    pretraining_dataset.set_format(dataset_format)

    return pretraining_dataset


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
    # Effective batch size in this example: 32 x 8 = 256
    pretrain_stonkgs(overwrite_output_dir=True, max_steps=200, batch_size=32, gradient_accumulation_steps=8)

    # (Optional) examine the runtime
    # with profiler.profile() as prof:
    #    with profiler.record_function("model_inference"):
    #        pretrain_stonkgs(overwrite_output_dir=True, max_steps=3)

    # logger.info(prof.key_averages().table(sort_by="cpu_time_total"))  # or replace by something like gpu_time_total
