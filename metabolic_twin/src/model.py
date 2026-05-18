"""Recurrent metabolic twin model."""

from __future__ import annotations

import torch
from torch import nn


class MetabolicTwin(nn.Module):
    """Sequence model for multihorizon glucose prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_horizons: int, rnn_type: str = "gru"):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        if self.rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.encoder = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, hidden = self.encoder(x)
        if self.rnn_type == "lstm":
            z_t = hidden[0][-1]
        else:
            z_t = hidden[-1]
        return self.head(z_t)
