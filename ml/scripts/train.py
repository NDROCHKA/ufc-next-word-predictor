"""CLI entry point that trains and evaluates both required models."""

import argparse
import json

from ufc_next_word.config import (
    ARTIFACTS_DIR,
    PROCESSED_DATA_DIR,
    ModelConfig,
    TrainingConfig,
)
from ufc_next_word.training.trainer import train_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TrainingConfig.max_epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--patience", type=int, default=TrainingConfig.patience)
    args = parser.parse_args()

    training_config = TrainingConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    report = train_all(
        processed_dir=PROCESSED_DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        model_config=ModelConfig(),
        training_config=training_config,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

