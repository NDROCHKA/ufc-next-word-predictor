"""Central paths and defaults shared by scripts, training, and inference."""

from dataclasses import asdict, dataclass
from pathlib import Path


# This file lives at ml/src/ufc_next_word/config.py, so parents[3] is the repo.
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "ufcstats_2013_2025.csv"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


@dataclass(frozen=True)
class ModelConfig:
    """Architecture values saved beside the trained weights."""

    embedding_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.25
    cnn_channels: int = 64
    cnn_kernel_sizes: tuple[int, ...] = (2, 3, 4)
    max_context_length: int = 12

    def to_dict(self) -> dict:
        data = asdict(self)
        # JSON has arrays rather than tuples. Converting explicitly makes the
        # artifact stable and easy to inspect outside Python.
        data["cnn_kernel_sizes"] = list(self.cnn_kernel_sizes)
        return data


@dataclass(frozen=True)
class TrainingConfig:
    """Conservative CPU-friendly training defaults."""

    seed: int = 433
    batch_size: int = 64
    learning_rate: float = 0.001
    max_epochs: int = 20
    patience: int = 4
    min_word_frequency: int = 2

