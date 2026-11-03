"""Single model construction path used by training and inference."""

from torch import nn

from ufc_next_word.config import ModelConfig

from .birnn import BiRNNNextWord
from .textcnn import TextCNNNextWord


def create_model(
    model_name: str,
    vocab_size: int,
    padding_idx: int,
    config: ModelConfig,
) -> nn.Module:
    normalized = model_name.lower()
    if normalized == "birnn":
        return BiRNNNextWord(
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    if normalized == "textcnn":
        return TextCNNNextWord(
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            embedding_dim=config.embedding_dim,
            channels=config.cnn_channels,
            kernel_sizes=config.cnn_kernel_sizes,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported model name: {model_name}")

