# -*- coding: utf-8 -*-

"""Runs the ProtSTonKGs model on the fine-tuning classification task, assuming the model embeddings are pre-trained.

Run with:
python -m src.stonkgs.models.protstonkgs_finetuning
"""

import logging
import os
from collections import Counter
from typing import Dict, Optional

import click
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers.models.big_bird import BigBirdModel, BigBirdTokenizer
from transformers.trainer import Trainer, TrainingArguments

from stonkgs.constants import (
    CELL_LINE_DIR,
    CELL_TYPE_DIR,
    CORRECT_DIR,
    DEEPSPEED_CONFIG_PATH,
    DISEASE_DIR,
    LOCATION_DIR,
    MLFLOW_FINETUNING_TRACKING_URI,
    NLP_MODEL_TYPE,
    ORGAN_DIR,
    PRETRAINED_PROTSTONKGS_PATH,
    PROTSTONKGS_MODEL_TYPE,
    PROT_EMBEDDINGS_PATH,
    PROT_RANDOM_WALKS_PATH,
    PROT_SEQ_MODEL_TYPE,
    PROT_STONKGS_OUTPUT_DIR,
    RELATION_TYPE_DIR,
    SPECIES_DIR,
    VOCAB_FILE,
)
from stonkgs.models.kg_baseline_model import prepare_df
from stonkgs.models.protstonkgs_model import ProtSTonKGsForPreTraining
from stonkgs.models.stonkgs_finetuning import INDRADataset, get_train_test_splits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# Disable alembic info
logging.getLogger("alembic").setLevel(logging.WARNING)


def preprocess_fine_tuning_data(
    train_data_path: str,
    class_column_name: str = "class",
    embedding_name_to_vector_path: str = PROT_EMBEDDINGS_PATH,
    embedding_name_to_random_walk_path: str = PROT_RANDOM_WALKS_PATH,
    nlp_model_type: str = NLP_MODEL_TYPE,
    prot_seq_length: int = 3072,
    prot_model_type: str = PROT_SEQ_MODEL_TYPE,
    prot_stonkgs_model_type: str = PROTSTONKGS_MODEL_TYPE,
    text_seq_length: int = 768,
) -> pd.DataFrame:
    """Generate input_ids, attention_mask, token_type_ids etc. based on the source, target, evidence columns."""
    # TODO documentation
    # Load the KG embedding dict to convert the names to numeric indices
    kg_embed_dict = prepare_df(embedding_name_to_vector_path)
    kg_name_to_idx = {key: i for i, key in enumerate(kg_embed_dict.keys())}
    # Load the random walks for each node
    random_walk_dict = prepare_df(embedding_name_to_random_walk_path)
    # Convert random walk sequences to list of numeric indices
    random_walk_idx_dict = {
        k: [kg_name_to_idx[node] for node in v] for k, v in random_walk_dict.items()
    }

    # Load the raw fine-tuning dataset with source, target and evidence
    unprocessed_df = pd.read_csv(
        train_data_path,
        sep="\t",
        usecols=[
            "source",
            "target",
            "evidence",
            "source_description",
            "target_description",
            "source_prot",
            "target_prot",
            class_column_name,
        ],
    )

    # Filter out any triples that contain a node that is not in the embeddings_dict
    original_length = len(unprocessed_df)
    unprocessed_df = unprocessed_df[
        unprocessed_df["source"].isin(kg_embed_dict.keys())
        & unprocessed_df["target"].isin(kg_embed_dict.keys())
    ].reset_index(drop=True)
    new_length = len(unprocessed_df)
    logger.info(
        f"{original_length - new_length} out of {original_length} triples are left out because they contain "
        f"nodes which are not present in the pre-training data"
    )

    # Check how many nodes in the fine-tuning dataset are not covered by the learned KG embeddings
    number_of_pre_training_nodes = len(
        set(unprocessed_df["source"]).union(set(unprocessed_df["target"]))
    )
    if number_of_pre_training_nodes > len(kg_embed_dict):
        logger.warning(
            f"{number_of_pre_training_nodes - len(kg_embed_dict)} out of {number_of_pre_training_nodes}"
            f"nodes are not covered by the embeddings learned in the pretraining dataset"
        )

    # Initialize a FAST LM tokenizer if it's the default one (BioBERT)
    if nlp_model_type == NLP_MODEL_TYPE:
        # Initialize the fast tokenizer for getting the text token ids
        lm_tokenizer = BertTokenizerFast(vocab_file=VOCAB_FILE)
    else:
        # Initialize a slow tokenizer used for getting the text token ids
        lm_tokenizer = BertTokenizer.from_pretrained(nlp_model_type)

    # Initialize the Protein sequence tokenizer
    prot_tokenizer = BertTokenizer.from_pretrained(prot_model_type)

    # Intialize the ProtSTonKGs tokenizer
    protstonkgs_tokenizer = BigBirdTokenizer.from_pretrained(prot_stonkgs_model_type)

    # Initialize the preprocessed data
    fine_tuning_preprocessed = []

    # Log progress with a progress bar
    for _, row in tqdm(
        unprocessed_df.iterrows(),
        total=unprocessed_df.shape[0],
        desc="Preprocessing the protein-specific fine-tuning dataset",
    ):
        # 1. Tokenization for getting the input ids and attention masks for the text
        # Use encode_plus to also get the attention mask ("padding" mask)
        # Manually add all special tokens ([CLS] in the beginning, [SEP] later)
        encoded_evidence = lm_tokenizer.encode_plus(
            row["evidence"],
            padding="max_length",
            truncation=True,
            max_length=text_seq_length // 3 - 2,
        )
        encoded_source_desc = lm_tokenizer.encode_plus(
            row["source_description"],
            padding="max_length",
            truncation=True,
            max_length=text_seq_length // 3 - 1,
            add_special_tokens=False,
        )
        encoded_target_desc = lm_tokenizer.encode_plus(
            row["target_description"],
            padding="max_length",
            truncation=True,
            max_length=text_seq_length // 3 - 1,
            add_special_tokens=False,
        )
        text_token_ids = (
            [lm_tokenizer.cls_token_id]
            + encoded_evidence["input_ids"]
            + [lm_tokenizer.sep_token_id]
            + encoded_source_desc["input_ids"]
            + [lm_tokenizer.sep_token_id]
            + encoded_target_desc["input_ids"]
            + [lm_tokenizer.sep_token_id]
        )
        text_attention_mask = (
            [1]
            + encoded_evidence["attention_mask"]
            + [1]
            + encoded_source_desc["attention_mask"]
            + [1]
            + encoded_target_desc["attention_mask"]
            + [1]
        )

        # 2. Get the random walks sequence/the node indices, add the SEP ID (usually with id=102) from the LM in between
        random_walks = (
            random_walk_idx_dict[row["source"]]
            + [protstonkgs_tokenizer.sep_token_id]
            + random_walk_idx_dict[row["target"]]
            + [protstonkgs_tokenizer.sep_token_id]
        )

        # 3. Get the protein sequence and combine source and target with [SEP]
        prot_sequence_source = prot_tokenizer.encode_plus(
            row["source_prot"],
            padding="max_length",
            truncation=True,
            max_length=prot_seq_length // 2 - 1,
            add_special_tokens=False,
        )
        prot_sequence_target = prot_tokenizer.encode_plus(
            row["target_prot"],
            padding="max_length",
            truncation=True,
            max_length=prot_seq_length // 2 - 1,
            add_special_tokens=False,
        )
        prot_sequence_ids = (
            prot_sequence_source["input_ids"]
            + [prot_tokenizer.sep_token_id]
            + prot_sequence_target["input_ids"]
            + [prot_tokenizer.sep_token_id]
        )
        prot_attention_mask = (
            prot_sequence_source["attention_mask"]
            + [1]
            + prot_sequence_target["attention_mask"]
            + [1]
        )

        # 4. Total attention mask (attention mask is all 1 for the entity sequence)
        attention_mask = text_attention_mask + [1] * len(random_walks) + prot_attention_mask

        # 5. Define the final concatenated input sequence
        input_ids = text_token_ids + random_walks + prot_sequence_ids

        # Add all the features to the preprocessed data
        fine_tuning_preprocessed.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,  # Remove the MLM, ELM and NSP labels since it's not needed anymore
                "labels": row[
                    class_column_name
                ],  # Add the annotation/relation label for fine-tuning instead
            }
        )

    # Put the preprocessed data into a dataframe
    fine_tuning_preprocessed_df = pd.DataFrame(fine_tuning_preprocessed)

    return fine_tuning_preprocessed_df


class ProtSTonKGsForSequenceClassification(ProtSTonKGsForPreTraining):
    """Create the fine-tuning part of the STonKGs model based the pre-trained STonKGs model.

    Note that this class inherits from STonKGsForPreTraining rather than PreTrainedModel, thereby it's deviating from
    the typical huggingface inheritance logic of the fine-tuning classes.
    """

    def __init__(self, config, **kwargs):
        """Initialize the STonKGs sequence classification model based on the pre-trained STonKGs model architecture."""
        super().__init__(config, **kwargs)
        self.num_labels = config.num_labels
        self.config = config

        # Load the pretrained ProtSTonKGs Transformer here
        self.bert = BigBirdModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        # Initialize all the pre-trained as well as new weights (i.e. classifier weights) here
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Perform one forward pass for a given sequence of text_input_ids + ent_input_ids."""
        # No backpropagation is needed for getting the initial embeddings from the backbones
        with torch.no_grad():
            # 1. Use the LM backbone to get the pre-trained token embeddings
            # batch x number_text_tokens x hidden_size
            # The first element of the returned tuple from the LM backbone forward() pass is the sequence of hidden
            # states
            text_embeddings = torch.cat(
                [
                    self.lm_backbone(
                        input_ids[
                            :, i * (self.kg_start_idx // 3) : (i + 1) * (self.kg_start_idx // 3)
                        ]
                    )[0]
                    for i in range(3)
                ],
                dim=1,
            )

            # 2. Use the KG backbone to obtain the pre-trained entity embeddings
            # batch x number_kg_tokens x hidden_size
            ent_embeddings = torch.stack(
                [
                    # for each numeric index in the random walks sequence: get the embedding vector from the KG backbone
                    torch.stack([self.kg_backbone[i.item()] for i in j])
                    # for each example in the batch: get the random walks sequence
                    for j in input_ids[:, self.kg_start_idx : self.prot_start_idx]
                ],
            )
            # 3. Use the Prot backbone to obtain the pre-trained entity embeddings
            # batch x number_prot_tokens x prot_hidden_size (prot_hidden_size != hidden_size)
            prot_embeddings_original_dim = self.prot_backbone(input_ids[:, self.prot_start_idx :])[
                0
            ]

        # Additional layer to project prot_hidden_size onto hidden_size
        prot_embeddings = self.prot_to_lm_hidden_linear(prot_embeddings_original_dim)

        # Concatenate token, KG and prot embeddings obtained from the LM, KG and prot backbones and cast to float
        # batch x seq_len x hidden_size
        inputs_embeds = (
            torch.cat(
                [
                    text_embeddings,
                    ent_embeddings.to(text_embeddings.device),
                    prot_embeddings.to(text_embeddings.device),
                ],
                dim=1,
            )
            .type(torch.FloatTensor)
            .to(self.device)
        )

        # Get the hidden states from the basic ProtSTonKGs Transformer layers
        # batch x seq_len x hidden_size
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=None,
        )
        # Only use the pooled output (of the [CLS] token)
        pooled_output = outputs[1]

        # Apply dropout and the linear layer
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def run_sequence_classification_cv(
    train_data_path: str,
    model_path: str = PRETRAINED_PROTSTONKGS_PATH,
    output_dir: Optional[str] = PROT_STONKGS_OUTPUT_DIR,
    logging_uri_mlflow: Optional[str] = MLFLOW_FINETUNING_TRACKING_URI,
    label_column_name: str = "labels",
    class_column_name: str = "class",
    epochs: Optional[int] = 10,
    log_steps: int = 50,
    lr: float = 5e-5,
    batch_size: int = 8,
    gradient_accumulation: int = 1,
    task_name: str = "",
    deepspeed: bool = True,
    max_dataset_size: int = 100000,
    cv: int = 5,
) -> Dict:
    """Run cross-validation for the sequence classification task(s) using STonKGs."""
    # Get data splits
    fine_tuning_df = preprocess_fine_tuning_data(
        train_data_path=train_data_path,
        class_column_name=class_column_name,
    )

    train_test_splits = get_train_test_splits(
        fine_tuning_df,
        max_dataset_size=max_dataset_size,
        n_splits=cv,
    )

    # Get text evidences and labels
    fine_tuning_data, labels_str = (
        fine_tuning_df.drop(columns=label_column_name),
        fine_tuning_df[label_column_name],
    )
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
    mlflow.set_experiment("ProtSTonKGs Fine-Tuning")

    # Initialize a dataframe for all the predicted labels
    result_df = pd.DataFrame()

    for idx, indices in enumerate(train_test_splits):
        model = ProtSTonKGsForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_path,
            num_labels=len(unique_tags),
        )

        # Based on the preprocessed fine-tuning dataframe: Convert the data into the desired dictionary format
        # for the INDRAEntityEvidenceDataset
        train_data = (
            fine_tuning_data.iloc[indices["train_idx"]]
            .reset_index(drop=True)
            .to_dict(orient="list")
        )
        test_data = (
            fine_tuning_data.iloc[indices["test_idx"]].reset_index(drop=True).to_dict(orient="list")
        )
        train_labels = labels[indices["train_idx"]].tolist()
        test_labels = labels[indices["test_idx"]].tolist()
        train_dataset = INDRADataset(encodings=train_data, labels=train_labels)
        test_dataset = INDRADataset(encodings=test_data, labels=test_labels)

        # Note that due to the randomization in the batches, the training/evaluation is slightly
        # different every time
        # TrainingArgument uses a default batch size of 8
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,  # total number of training epochs
            logging_steps=log_steps,  # reduce the number of logging steps to avoid collisions when writing to the
            # shared database
            # Use deepspeed with a specified config file for speedup
            deepspeed=DEEPSPEED_CONFIG_PATH if deepspeed else None,
            learning_rate=lr,
            report_to=["mlflow"],  # log via mlflow
            do_train=True,
            do_predict=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
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
        mlflow.log_param("label dict", str(tag2id))
        mlflow.log_param("training dataset size", str(len(train_labels)))
        mlflow.log_param("training class dist", str(Counter(train_labels)))
        mlflow.log_param("test dataset size", str(len(test_labels)))
        mlflow.log_param("test class dist", str(Counter(test_labels)))

        # Make predictions for the test dataset
        predictions = trainer.predict(test_dataset=test_dataset).predictions
        predicted_labels = np.argmax(predictions, axis=1)
        logger.info(f"Predicted labels: {predicted_labels}")

        # Save the predicted + true labels
        partial_result_df = pd.DataFrame(
            {
                "split": idx,
                "index": indices["test_idx"].tolist(),
                "predicted_label": predicted_labels.tolist(),
                "true_label": test_labels,
                "evidence": fine_tuning_data.iloc[indices["test_idx"]]["input_ids"].tolist(),
            },
        )
        result_df = result_df.append(
            partial_result_df,
            ignore_index=True,
        )

        # Use weighted average
        f1_sc = f1_score(test_labels, predicted_labels, average="weighted")
        f1_scores.append(f1_sc)

        # Log the final f1_score
        mlflow.log_metric("f1_score_weighted", f1_sc)

    # Log mean and std f1-scores from the cross validation procedure (average and std across all splits) to the
    # standard logger
    logger.info(f"Mean f1-score: {np.mean(f1_scores)}")
    logger.info(f"Std f1-score: {np.std(f1_scores)}")

    # Map the labels in the result df back to their original names
    result_df = result_df.replace({"predicted_label": id2tag, "true_label": id2tag})
    # Save the result_df
    result_df.to_csv(
        os.path.join(
            PROT_STONKGS_OUTPUT_DIR, "predicted_labels_protstonkgs_" + task_name + "df.tsv"
        ),
        index=False,
        sep="\t",
    )

    # Save the last model
    trainer.save_model(output_dir=os.path.join(PROT_STONKGS_OUTPUT_DIR, task_name))

    # End the previous run
    mlflow.end_run()

    # Log the mean and std f1 score from the cross validation procedure to mlflow
    with mlflow.start_run():
        # Log the task name as well
        mlflow.log_param("task name", task_name)
        mlflow.log_metric("f1_score_mean", np.mean(f1_scores))
        mlflow.log_metric("f1_score_std", np.std(f1_scores))

    return {"f1_score_mean": np.mean(f1_scores), "f1_score_std": np.std(f1_scores)}


@click.command()
@click.option("-e", "--epochs", default=5, help="Number of epochs", type=int)
@click.option("--cv", default=5, help="Number of cross validation splits (use 1 to omit cv)", type=int)
@click.option("--lr", default=5e-5, help="Learning rate", type=float)
@click.option(
    "--logging_dir",
    default=MLFLOW_FINETUNING_TRACKING_URI,
    help="Mlflow logging/tracking URI",
    type=str,
)
@click.option("--log_steps", default=500, help="Number of steps between each log", type=int)
@click.option(
    "--model_path",
    default=PRETRAINED_PROTSTONKGS_PATH,
    help="Path of the pretrained model",
    type=str,
)
@click.option("--output_dir", default=PROT_STONKGS_OUTPUT_DIR, help="Output directory", type=str)
@click.option("--batch_size", default=8, help="Batch size used in fine-tuning", type=int)
@click.option(
    "--gradient_accumulation_steps", default=1, help="Gradient accumulation steps", type=int
)
@click.option("--deepspeed", default=True, help="Whether to use deepspeed or not", type=bool)
@click.option(
    "--max_dataset_size",
    default=100000,
    help="Maximum dataset size of the fine-tuning datasets",
    type=int,
)
@click.option("--local_rank", default=-1, help="THIS PARAMETER IS IGNORED", type=int)
def run_all_fine_tuning_tasks(
    epochs: int = 5,
    log_steps: int = 500,
    lr: float = 5e-5,
    model_path: str = PRETRAINED_PROTSTONKGS_PATH,
    output_dir: str = PROT_STONKGS_OUTPUT_DIR,
    logging_dir: Optional[str] = MLFLOW_FINETUNING_TRACKING_URI,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    deepspeed: bool = True,
    max_dataset_size: int = 100000,  # effectively removes the max dataset size restriction
    cv: int = 5,
    local_rank: int = -1,
):
    """Run all fine-tuning tasks at once."""
    # Specify all directories and file names
    directories = [
        CELL_LINE_DIR,
        CELL_TYPE_DIR,
        CORRECT_DIR,
        CORRECT_DIR,
        DISEASE_DIR,
        LOCATION_DIR,
        ORGAN_DIR,
        SPECIES_DIR,
        RELATION_TYPE_DIR,
        RELATION_TYPE_DIR,
    ]
    file_names = [
        "cell_line_ppi_prot.tsv",
        "cell_type_ppi_prot.tsv",
        "correct_incorrect_binary_ppi_prot.tsv",
        "correct_incorrect_multiclass_ppi_prot.tsv",
        "disease_ppi_prot.tsv",
        "location_ppi_prot.tsv",
        "organ_ppi_prot.tsv",
        "species_ppi_prot.tsv",
        "relation_type_ppi_prot.tsv",
        "relation_type_ppi_prot.tsv",
    ]
    task_names = [
        "cell_line",
        "cell_type",
        "correct_binary",
        "correct_multiclass",
        "disease",
        "location",
        "organ",
        "species",
        "interaction",
        "polarity",
    ]
    # Specify the column names of the target variable
    column_names = ["class"] * 8 + ["interaction"] + ["polarity"]

    for directory, file, column_name, task_name in zip(
        directories,
        file_names,
        column_names,
        task_names,
    ):
        run_sequence_classification_cv(
            train_data_path=os.path.join(directory, file),
            model_path=model_path,
            output_dir=output_dir,
            logging_uri_mlflow=logging_dir,
            epochs=epochs,
            log_steps=log_steps,
            lr=lr,
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation_steps,
            class_column_name=column_name,
            task_name=task_name,
            deepspeed=deepspeed,
            max_dataset_size=max_dataset_size,
            cv=cv,
        )
        logger.info(f"Finished the {task_name} task")


if __name__ == "__main__":
    # Set the huggingface environment variable for tokenizer parallelism to false
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Run all CV fine-tuning tasks
    run_all_fine_tuning_tasks()
