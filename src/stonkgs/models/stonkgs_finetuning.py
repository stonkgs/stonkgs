# -*- coding: utf-8 -*-

"""Runs the STonKGs model on the fine-tuning classification task, assuming the model embeddings are pre-trained.

Run with:
python -m src.stonkgs.models.stonkgs_finetuning
"""

import logging
import os
from typing import Dict, List, Optional

import click
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert import BertModel, BertTokenizer, BertTokenizerFast
from transformers.trainer import Trainer, TrainingArguments

from stonkgs.constants import (
    CELL_LINE_DIR,
    CELL_TYPE_DIR,
    DISEASE_DIR,
    EMBEDDINGS_PATH,
    LOCATION_DIR,
    MLFLOW_FINETUNING_TRACKING_URI,
    NLP_MODEL_TYPE,
    ORGAN_DIR,
    PRETRAINED_STONKGS_DUMMY_PATH,
    RANDOM_WALKS_PATH,
    RELATION_TYPE_DIR,
    SPECIES_DIR,
    STONKGS_OUTPUT_DIR,
    VOCAB_FILE,
)
from stonkgs.models.kg_baseline_model import _prepare_df
from stonkgs.models.stonkgs_model import STonKGsForPreTraining

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# Disable alembic info
logging.getLogger("alembic").setLevel(logging.WARNING)


def get_train_test_splits(
    train_data: pd.DataFrame,
    type_column_name: str = "labels",
    random_seed: int = 42,
    n_splits: int = 5,
) -> List:
    """Return train/test indices for n_splits many splits based on the fine-tuning dataset that is passed."""
    # Leave out the label in the dataset
    data = train_data.drop(type_column_name, axis=1)
    labels = train_data[type_column_name]

    # Implement stratified train/test splits
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=True)

    return [{"train_idx": train_idx, "test_idx": test_idx} for train_idx, test_idx in skf.split(data, labels)]


def preprocess_fine_tuning_data(
    train_data_path: str,
    class_column_name: str = "class",
    embedding_name_to_vector_path: str = EMBEDDINGS_PATH,
    embedding_name_to_random_walk_path: str = RANDOM_WALKS_PATH,
    nlp_model_type: str = NLP_MODEL_TYPE,
    sep_id: int = 102,
    unk_id: int = 100,
) -> pd.DataFrame:
    """Generate input_ids, attention_mask, token_type_ids etc. based on the source, target, evidence columns."""
    # Load the KG embedding dict to convert the names to numeric indices
    kg_embed_dict = _prepare_df(embedding_name_to_vector_path)
    kg_name_to_idx = {key: i for i, key in enumerate(kg_embed_dict.keys())}
    # Load the random walks for each node
    random_walk_dict = _prepare_df(embedding_name_to_random_walk_path)
    # Convert random walk sequences to list of numeric indices
    random_walk_idx_dict = {k: [kg_name_to_idx[node] for node in v] for k, v in random_walk_dict.items()}

    # Load the raw fine-tuning dataset with source, target and evidence
    unprocessed_df = pd.read_csv(train_data_path, sep='\t', usecols=["source", "target", "evidence", class_column_name])

    # Check how many nodes in the fine-tuning dataset are not covered by the learned KG embeddings
    number_of_pre_training_nodes = len(set(unprocessed_df["source"]).union(set(unprocessed_df["target"])))
    if number_of_pre_training_nodes > len(kg_embed_dict):
        logger.warning(f'{number_of_pre_training_nodes - len(kg_embed_dict)} out of {number_of_pre_training_nodes}'
                       f'nodes are not covered by the embeddings learned in the pretraining dataset')

    # Get the length of the text or entity embedding sequences (2 random walks + 2 = entity embedding sequence length)
    random_walk_length = len(next(iter(random_walk_idx_dict.values())))
    half_length = random_walk_length * 2 + 2

    # Initialize a FAST tokenizer if it's the default one (BioBERT)
    if nlp_model_type == NLP_MODEL_TYPE:
        # Initialize the fast tokenizer for getting the text token ids
        tokenizer = BertTokenizerFast(vocab_file=VOCAB_FILE)
    else:
        # Initialize a slow tokenizer used for getting the text token ids
        tokenizer = BertTokenizer.from_pretrained(nlp_model_type)

    # Initialize the preprocessed data
    fine_tuning_preprocessed = []

    # Log progress with a progress bar
    for _, row in tqdm(
        unprocessed_df.iterrows(),
        total=unprocessed_df.shape[0],
        desc='Preprocessing the fine-tuning dataset',
    ):
        # 1. "Token type IDs": 0 for text tokens, 1 for entity tokens
        token_type_ids = [0] * half_length + [1] * half_length

        # 2. Tokenization for getting the input ids and attention masks for the text
        # Use encode_plus to also get the attention mask ("padding" mask)
        encoded_text = tokenizer.encode_plus(
            row['evidence'],
            padding='max_length',
            truncation=True,
            max_length=half_length,
        )
        text_token_ids = encoded_text['input_ids']
        text_attention_mask = encoded_text['attention_mask']

        # 3. Get the random walks sequence and the node indices, add the SEP (usually with id=102) in between
        # Use a sequence of UNK tokens if the node is not contained in the dictionary of the nodes from pre-training
        random_w_source = random_walk_idx_dict[
            row['source']
        ] if row['source'] in random_walk_idx_dict.keys() else [unk_id] * random_walk_length
        random_w_target = random_walk_idx_dict[
            row['target']
        ] if row['target'] in random_walk_idx_dict.keys() else [unk_id] * random_walk_length
        random_w_ids = random_w_source + [sep_id] + random_w_target + [sep_id]

        # 4. Total attention mask (attention mask is all 1 for the entity sequence)
        attention_mask = text_attention_mask + [1] * half_length

        # 5. Total input_ids = half text ids + half entity ids
        input_ids = text_token_ids + random_w_ids

        # Add all the features to the preprocessed data
        fine_tuning_preprocessed.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,  # Remove the MLM, ELM and NSP labels since it's not needed anymore
            'labels': row[class_column_name],  # Add the annotation/relation label for fine-tuning instead
        })

    # Put the preprocessed data into a dataframe
    fine_tuning_preprocessed_df = pd.DataFrame(fine_tuning_preprocessed)

    return fine_tuning_preprocessed_df


class INDRAEntityEvidenceDataset(torch.utils.data.Dataset):
    """Custom Dataset class for INDRA data containing the combination of text and KG triple data."""

    def __init__(
        self,
        encodings,
        labels,
    ):
        """Initialize INDRA Dataset based on the combined input sequence consisting of text and triple data."""
        self.encodings = encodings
        # Assumes that the labels are numerically encoded
        self.labels = labels

    def __getitem__(self, idx):
        """Return data entries for given indices."""
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.labels)


class STonKGsForSequenceClassification(STonKGsForPreTraining):
    """Create the fine-tuning part of the STonKGs model based the pre-trained STonKGs model.

    Note that this class inherits from STonKGsForPreTraining rather than PreTrainedModel, thereby it's deviating from
    the typical huggingface inheritance logic of the fine-tuning classes.
    """

    def __init__(self, config):
        """Initialize the STonKGs sequence classification model based on the pre-trained STonKGs model architecture."""
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Load the pretrained STonKGs Transformer here
        self.bert = BertModel(config)
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
        # Use the LM backbone to get the pre-trained token embeddings
        # batch x half_length x hidden_size
        # The first element of the returned tuple from the LM backbone forward() pass is the sequence of hidden states
        token_embeddings = self.lm_backbone(input_ids[:, :self.cls.predictions.half_length])[0]

        # Use the KG backbone to obtain the pre-trained entity embeddings
        # batch x half_length x hidden_size
        ent_embeddings = torch.stack([
            # for each numeric index in the random walks sequence: get the embedding vector from the KG backbone
            torch.stack([self.kg_backbone[i.item()] for i in j])
            # for each example in the batch: get the random walks sequence
            for j in input_ids[:, self.cls.predictions.half_length:]],
        )

        # Concatenate token and entity embeddings obtained from the LM and KG backbones and cast to float
        # batch x seq_len x hidden_size
        inputs_embeds = torch.cat(
            [token_embeddings, ent_embeddings.to(token_embeddings.device)],
            dim=1,
        ).type(torch.FloatTensor).to(self.device)

        # Get the hidden states from the pretrained STonKGs Transformer layers
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
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
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
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
    model_path: str = PRETRAINED_STONKGS_DUMMY_PATH,
    output_dir: Optional[str] = STONKGS_OUTPUT_DIR,
    logging_uri_mlflow: Optional[str] = MLFLOW_FINETUNING_TRACKING_URI,
    label_column_name: str = "labels",
    class_column_name: str = "class",
    epochs: Optional[int] = 10,
    log_steps: int = 500,
    lr: float = 5e-5,
) -> Dict:
    """Run cross-validation for the sequence classification task(s) using STonKGs."""
    # Get data splits
    fine_tuning_df = preprocess_fine_tuning_data(
        train_data_path=train_data_path,
        class_column_name=class_column_name,
    )
    train_test_splits = get_train_test_splits(fine_tuning_df)

    # Get text evidences and labels
    fine_tuning_data, labels_str = fine_tuning_df.drop(columns=label_column_name), fine_tuning_df[label_column_name]
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
    mlflow.set_experiment('STonKGs Fine-Tuning')

    for indices in train_test_splits:
        model = STonKGsForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_path,
            num_labels=len(unique_tags),
        )

        # Based on the preprocessed fine-tuning dataframe: Convert the data into the desired dictionary format
        # for the INDRAEntityEvidenceDataset
        train_data = fine_tuning_data.iloc[indices["train_idx"]].reset_index(drop=True).to_dict(orient='list')
        test_data = fine_tuning_data.iloc[indices["test_idx"]].reset_index(drop=True).to_dict(orient='list')
        train_labels = labels[indices["train_idx"]].tolist()
        test_labels = labels[indices["test_idx"]].tolist()
        train_dataset = INDRAEntityEvidenceDataset(encodings=train_data, labels=train_labels)
        test_dataset = INDRAEntityEvidenceDataset(encodings=test_data, labels=test_labels)

        # Note that due to the randomization in the batches, the training/evaluation is slightly
        # different every time
        # TrainingArgument uses a default batch size of 8
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,  # total number of training epochs
            logging_steps=log_steps,  # reduce the number of logging steps to avoid collisions when writing to the
            # shared database
            learning_rate=lr,
            report_to=["mlflow"],  # log via mlflow
            lr_scheduler_type='constant',
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
        logger.info(f'Predicted labels: {predicted_labels}')

        # Use macro average for now
        f1_sc = f1_score(test_labels, predicted_labels, average="macro")
        f1_scores.append(f1_sc)

        # Log the final f1_score
        mlflow.log_metric('f1_score_macro', f1_sc)

    # Log mean and std f1-scores from the cross validation procedure (average and std across all splits) to the
    # standard logger
    logger.info(f'Mean f1-score: {np.mean(f1_scores)}')
    logger.info(f'Std f1-score: {np.std(f1_scores)}')

    # End the previous run
    mlflow.end_run()

    # Log the mean and std f1 score from the cross validation procedure to mlflow
    with mlflow.start_run():
        mlflow.log_metric('f1_score_mean', np.mean(f1_scores))
        mlflow.log_metric('f1_score_std', np.std(f1_scores))

    return {"f1_score_mean": np.mean(f1_scores), "f1_score_std": np.std(f1_scores)}


@click.command()
@click.option('-e', '--epochs', default=3, help='Number of epochs', type=int)
@click.option('--lr', default=5e-5, help='Learning rate', type=float)
@click.option('--logging_dir', default=MLFLOW_FINETUNING_TRACKING_URI, help='Mlflow logging/tracking URI', type=str)
@click.option('--log_steps', default=500, help='Number of steps between each log', type=int)
@click.option('--model_path', default=PRETRAINED_STONKGS_DUMMY_PATH, help='Path of the pretrained model', type=str)
@click.option('--output_dir', default=STONKGS_OUTPUT_DIR, help='Output directory', type=str)
def run_all_fine_tuning_tasks(
    epochs: int = 3,
    log_steps: int = 500,
    lr: float = 5e-5,
    model_path: str = PRETRAINED_STONKGS_DUMMY_PATH,
    output_dir: str = STONKGS_OUTPUT_DIR,
    logging_dir: str = MLFLOW_FINETUNING_TRACKING_URI,
):
    # Run the 6 annotation type tasks
    # 1. Cell line
    run_sequence_classification_cv(
        train_data_path=os.path.join(CELL_LINE_DIR, 'cell_line_filtered.tsv'),
        model_path=model_path,
        output_dir=output_dir,
        logging_uri_mlflow=logging_dir,
        epochs=epochs,
        log_steps=log_steps,
        lr=lr,
    )
    logger.info('Finished the cell line task')

    # 2. Cell type
    run_sequence_classification_cv(
        train_data_path=os.path.join(CELL_TYPE_DIR, 'cell_type_filtered.tsv'),
        model_path=model_path,
        output_dir=output_dir,
        logging_uri_mlflow=logging_dir,
        epochs=epochs,
        log_steps=log_steps,
        lr=lr,
    )
    logger.info('Finished the cell type task')

    # 3. Disease
    run_sequence_classification_cv(
        train_data_path=os.path.join(DISEASE_DIR, 'disease_filtered.tsv'),
        model_path=model_path,
        output_dir=output_dir,
        logging_uri_mlflow=logging_dir,
        epochs=epochs,
        log_steps=log_steps,
        lr=lr,
    )
    logger.info('Finished the disease task')

    # 4. Location
    run_sequence_classification_cv(
        train_data_path=os.path.join(LOCATION_DIR, 'location_filtered.tsv'),
        model_path=model_path,
        output_dir=output_dir,
        logging_uri_mlflow=logging_dir,
        epochs=epochs,
        log_steps=log_steps,
        lr=lr,
    )
    logger.info('Finished the location task')

    # 5. Organ
    run_sequence_classification_cv(
        train_data_path=os.path.join(ORGAN_DIR, 'organ_filtered.tsv'),
        model_path=model_path,
        output_dir=output_dir,
        logging_uri_mlflow=logging_dir,
        epochs=epochs,
        log_steps=log_steps,
        lr=lr,
    )
    logger.info('Finished the organ task')

    # 6. Species
    run_sequence_classification_cv(
        train_data_path=os.path.join(SPECIES_DIR, 'species_filtered.tsv'),
        model_path=model_path,
        output_dir=output_dir,
        logging_uri_mlflow=logging_dir,
        epochs=epochs,
        log_steps=log_steps,
        lr=lr,
    )
    logger.info('Finished the species task')

    # Run the two relation type classification tasks
    # 7. Interaction type
    run_sequence_classification_cv(
        train_data_path=os.path.join(RELATION_TYPE_DIR, 'relation_type.tsv'),
        class_column_name='interaction',
        model_path=model_path,
        output_dir=output_dir,
        logging_uri_mlflow=logging_dir,
        epochs=epochs,
        log_steps=log_steps,
        lr=lr,
    )
    logger.info('Finished the interaction type task')

    # 8. Polarity
    run_sequence_classification_cv(
        train_data_path=os.path.join(RELATION_TYPE_DIR, 'relation_type.tsv'),
        class_column_name='polarity',
        model_path=model_path,
        output_dir=output_dir,
        logging_uri_mlflow=logging_dir,
        epochs=epochs,
        log_steps=log_steps,
        lr=lr,
    )
    logger.info('Finished the polarity task')


if __name__ == "__main__":
    # Set the huggingface environment variable for tokenizer parallelism to false
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Run all CV fine-tuning tasks
    run_all_fine_tuning_tasks()
