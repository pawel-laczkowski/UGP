import logging

LOGGER = logging.getLogger(__name__)

# Custom RoBERTa model
import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaModel


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
pass

# Custom T5 model
pass
