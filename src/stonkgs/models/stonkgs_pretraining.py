# -*- coding: utf-8 -*-

"""Script for running the pre-training procedure of STonKGs."""

import logging
import os
from typing import Optional

import click
import mlflow
from accelerate import Accelerator
from datasets import Dataset, load_dataset, total_allocated_bytes
from transformers import (
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from stonkgs.constants import (
    DEEPSPEED_CONFIG_PATH,
    MLFLOW_TRACKING_URI,
    NLP_MODEL_TYPE,
    PRETRAINING_PREPROCESSED_POSITIVE_DF_PATH,
    STONKGS_PRETRAINING_NO_NSP_DIR,
)
from stonkgs.models.stonkgs_model import STonKGsForPreTraining

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# Disable alembic info
logging.getLogger("alembic").setLevel(logging.WARNING)


def _load_pre_training_data(
    pretraining_preprocessed_path: str = PRETRAINING_PREPROCESSED_POSITIVE_DF_PATH,
    dataset_format: str = "torch",
) -> Dataset:
    """Create a pytorch dataset based on a preprocessed dataframe for the pretraining dataset."""
    # Load the pickled dataframe using the load_dataset function so that it can be cached
    pretraining_dataset = load_dataset(
        "pandas", data_files=pretraining_preprocessed_path, split="train"
    )

    # Do not put the dataset on the GPU even if possible, it is only stealing GPU space, use the dataloader instead
    # Putting it on the GPU might only be worth it if 4+ GPUs are used
    pretraining_dataset.set_format(dataset_format)
    logger.info("Finished loading the pretraining dataset")

    return pretraining_dataset


@click.command()
@click.option(
    "-b", "--batch_size", default=8, help="Batch size for training (per device)", type=int
)
@click.option("--fp16", default=True, help="Whether to use fp16 precision or not", type=bool)
@click.option("--lr", default=1e-4, help="Learning rate", type=float)
@click.option("--dataloader_num_workers", default=8, help="Number of dataloader workers", type=int)
@click.option("--deepspeed", default=False, help="Whether to use deepspeed or not", type=bool)
@click.option(
    "--gradient_accumulation_steps",
    default=1,
    help="Number of gradient accumulation steps",
    type=int,
)
@click.option(
    "--logging_dir", default=MLFLOW_TRACKING_URI, help="Mlflow logging/tracking URI", type=str
)
@click.option("--logging_steps", default=100, help="Logging interval", type=int)
@click.option("-m", "--max_steps", default=200, help="Number of training steps", type=int)
@click.option(
    "--overwrite_output_dir",
    default=False,
    help="Whether to override the output dir or not",
    type=bool,
)
@click.option(
    "--pretraining_file",
    default=PRETRAINING_PREPROCESSED_POSITIVE_DF_PATH,
    help="File used in pretraining containing the preprocessed training examples",
    type=str,
)
@click.option(
    "--save_limit", default=5, help="Maximum number of saved models/checkpoints", type=int
)
@click.option("--save_steps", default=5000, help="Checkpointing interval", type=int)
@click.option(
    "--training_dir",
    default=STONKGS_PRETRAINING_NO_NSP_DIR,
    help="Whether to override the output dir",
    type=str,
)
def pretrain_stonkgs(
    batch_size: int = 8,
    deepspeed: bool = False,
    fp16: bool = True,
    lr: float = 1e-4,
    dataloader_num_workers: int = 8,  # empirically determined value, I'm open to changing it :)
    gradient_accumulation_steps: int = 1,
    logging_dir: Optional[str] = MLFLOW_TRACKING_URI,
    logging_steps: int = 100,
    max_steps: int = 10000,
    overwrite_output_dir: bool = False,
    pretraining_file: str = PRETRAINING_PREPROCESSED_POSITIVE_DF_PATH,
    save_limit: int = 5,
    save_steps: int = 5000,
    training_dir: str = STONKGS_PRETRAINING_NO_NSP_DIR,
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
    mlflow.set_experiment("STonKGs Pre-Training (No NSP Ablation)")

    # Initialize the STonKGs model
    # config = None just fills up the required argument for automated method calls such as .from_pretrained, it will be
    # overridden anyways in __init__
    stonkgs_model = STonKGsForPreTraining(config=None, nlp_model_type=NLP_MODEL_TYPE)

    # Add the huggingface accelerator
    accelerator = Accelerator()
    # Use the device given by the `accelerator` object and put the model on there
    device = accelerator.device
    stonkgs_model.to(device)

    # Initialize the dataset
    pretraining_data = _load_pre_training_data(pretraining_preprocessed_path=pretraining_file)

    # Print logger statements about the allocated number of bytes of the training dataset
    logger.info(
        f"The number of bytes allocated by the dataset on the drive is {pretraining_data.dataset_size}"
    )
    logger.info(
        f"For comparison, here is the number of bytes allocated in memory by the training dataset"
        f": {total_allocated_bytes()}"
    )

    # Clean up cache files to be sure
    pretraining_data.cleanup_cache_files()

    # Accelerate the model
    stonkgs_model = accelerator.prepare(stonkgs_model)

    # Initialize the TrainingArguments
    training_args = TrainingArguments(
        output_dir=training_dir,
        overwrite_output_dir=overwrite_output_dir,
        # Use deepspeed with a specified config file for speedup
        deepspeed=DEEPSPEED_CONFIG_PATH if deepspeed else None,
        do_train=True,
        # Use fp16 to save space
        fp16=fp16,
        fp16_full_eval=fp16,
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,  # Use max_steps rather than num_training_epochs
        learning_rate=lr,  # Default is to use that lr with a linear scheduler
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_limit,
        report_to=["mlflow"],
        # Make the dataloader faster by using pinning the memory and using multiple workers
        dataloader_pin_memory=True,
        dataloader_num_workers=dataloader_num_workers,
        # Effectively increase the batch size by gradient acc: batch_size = old_batch_size x grad_acc_steps
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Detecting last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
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
    # Delete the pretraining dataset from memory
    del pretraining_data
    # And train STonKGs to the moon
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    # Log all metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    # Run the pre-training procedure, overwrite the output dir for now (since we're only working with dummy data)
    # Distinguish between local and cluster execution by using different values (steps, batch size etc.) in the
    # CLI arguments
    pretrain_stonkgs()

    # (Optional) examine the runtime
    # with profiler.profile() as prof:
    #    with profiler.record_function("model_inference"):
    #        pretrain_stonkgs(overwrite_output_dir=True, max_steps=3)

    # logger.info(prof.key_averages().table(sort_by="cpu_time_total"))  # or replace by something like gpu_time_total
