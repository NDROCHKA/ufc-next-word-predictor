"""Bidirectional vanilla-RNN next-word classifier."""

import torch
from torch import nn


class BiRNNNextWord(nn.Module):
    """Read the supplied prefix in both directions and classify its next word.

    Bidirectionality applies only to the words already typed by the user. The
    target next word is never placed in the input, so there is no label leakage.
    """

    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        _, hidden = self.rnn(embedded)
        # The last two states are the final forward and reverse states from the
        # top recurrent layer. Concatenating them uses both directions.
        combined = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(combined))

