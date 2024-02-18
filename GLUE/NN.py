import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BertModel

class My_BertForSequenceClassification(nn.Module):
    def __init__(self, hidden_size, num_labels, args):
        super().__init__()
        self.num_labels = num_labels
        self.args = args
        self.hidden_size = hidden_size

        self.bert = BertModel.from_pretrained(self.args.model_name)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        self.weight_init()

    
        