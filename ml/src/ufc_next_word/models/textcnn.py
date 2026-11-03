"""TextCNN next-word classifier using several local context widths."""

import torch
from torch import nn
from torch.nn import functional as F


class TextCNNNextWord(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        embedding_dim: int,
        channels: int,
        kernel_sizes: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=channels,
                kernel_size=kernel_size,
            )
            for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * len(kernel_sizes), vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Conv1d expects (batch, channels, sequence), while Embedding returns
        # (batch, sequence, embedding). Transpose the last two dimensions.
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled_features = []
        for convolution in self.convolutions:
            activated = F.relu(convolution(embedded))
            pooled_features.append(F.max_pool1d(activated, activated.size(2)).squeeze(2))
        combined = torch.cat(pooled_features, dim=1)
        return self.classifier(self.dropout(combined))

