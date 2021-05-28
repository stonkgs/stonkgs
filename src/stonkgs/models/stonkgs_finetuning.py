# -*- coding: utf-8 -*-

"""STonKGs model on the fine-tuning classification task, assuming the model embeddings are pre-trained."""

import logging
from typing import List

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert import BertModel

from stonkgs.constants import PRETRAINED_STONKGS_DUMMY_PATH
from stonkgs.models.stonkgs_model import STonKGsForPreTraining

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_train_test_splits(
    train_data: pd.DataFrame,
    type_column_name: str = "class",
    random_seed: int = 42,
    n_splits: int = 5,
) -> List:
    """Return train/test indices for n_splits many splits based on the fine-tuning dataset that is passed."""
    # Leave out the label in the dataset
    data = train_data.drop(type_column_name, axis=1)
    labels = train_data[type_column_name]

    # Implement stratified train/test splits
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=False)

    return [[train_idx, test_idx] for train_idx, test_idx in skf.split(data, labels)]


def preprocess_fine_tuning_data(
    train_data_path: str,
) -> pd.DataFrame:
    """Generates input_ids, attention_mask, token_type_ids etc. based on the source, target, evidence columns."""
    unprocessed_df = pd.read_csv(train_data_path, sep='\t')


# TODO: fine-tuning class
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

        # TODO put from pretrained here? or is this happening automatically?
        # Load the pretrained STonKGs Transformer here
        self.bert = BertModel(config)
        # TODO: implement dropout in the KG baseline model, too?
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

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


if __name__ == "__main__":
    # Add a test for loading the model
    unique_tags = 10
    # config = PretrainedConfig.from_pretrained(PRETRAINED_STONKGS_DUMMY_PATH)
    model = STonKGsForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=PRETRAINED_STONKGS_DUMMY_PATH,
        num_labels=unique_tags)
    # TODO: set up fine tuning

