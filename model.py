import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Network(nn.Module):
    """
    A Transformer-based network for EEG signal processing with masked autoencoding.
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.input_layer = nn.Linear(29, hidden_dim)

        self.projection_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        emb = self.input_layer(data)
        features = self.projection_layer(emb)
        output = self.output_layer(features)

        return output
