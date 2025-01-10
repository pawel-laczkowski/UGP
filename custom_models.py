import logging

LOGGER = logging.getLogger(__name__)

import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaModel, GPT2ForSequenceClassification, GPT2Model


# Custom RoBERTa model
class CustomRobertaClassificationHead(nn.Module):
    """
    Multi-layer perceptron head with dropout and ReLU + GELU activation function for RoBERTa model.
    """

    def __init__(self, hidden_size, num_classes, dropout_prob=0.1):
        super(CustomRobertaClassificationHead, self).__init__()

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.activation2 = nn.GELU()
        self.fc3 = nn.Linear(hidden_size // 4, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class CustomRoBERTaForSequenceClassification(RobertaForSequenceClassification):
    """
    Custom RoBERTa model for sequence classification with a custom head and using the last 3 hidden states modifications.
    """

    def __init__(self, config):
        super(CustomRoBERTaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = CustomRobertaClassificationHead(
            hidden_size=config.hidden_size * 3,
            num_classes=config.num_labels,
            dropout_prob=config.hidden_dropout_prob,
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            **kwargs,
        )

        hidden_states = outputs.hidden_states
        last_3_hidden_states = torch.cat(hidden_states[-3:], dim=-1)

        cls_representation = last_3_hidden_states[:, 0, :]
        logits = self.classifier(cls_representation)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return (loss, logits) if loss is not None else logits


# Custom GPT2 model
class CustomGPT2ClassificationHead(nn.Module):
    """
    Custom classification head based on ResNet-like MLP with BatchNorm and GELU.
    """

    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.1):
        super(CustomGPT2ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.activation1 = nn.GELU()

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.activation2 = nn.GELU()

        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, pooled_output):
        x = self.fc1(pooled_output)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout(x)

        x_res = x

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.dropout(x)

        x += x_res

        return self.fc3(x)


class CustomGPT2ForSequenceClassification(GPT2ForSequenceClassification):
    """
    Custom GPT-2 model for sequence classification with a custom head and freezing the first 3 transformer blocks.
    """

    def __init__(self, config):
        super(CustomGPT2ForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.transformer = GPT2Model(config)
        self.score = CustomGPT2ClassificationHead(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size * 2,
            num_classes=config.num_labels,
            dropout_prob=config.resid_pdrop,
        )

        # Freezing the first 3 transformer blocks
        LOGGER.info("Freezing the first 3 GPT2 transformer blocks.")
        for idx, block in enumerate(self.transformer.h):
            if idx < 3:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return (loss, logits) if loss is not None else logits


# Custom T5 model
pass
