"""Reproducible training orchestration for both required models."""

from copy import deepcopy
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from ufc_next_word.config import ModelConfig, TrainingConfig
from ufc_next_word.data.dataset import PrefixDataset, load_tokenized_sentences
from ufc_next_word.data.vocabulary import Vocabulary, build_vocabulary
from ufc_next_word.models import create_model

from .metrics import evaluate


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    dataset: PrefixDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,  # Works consistently on Windows and in classroom laptops.
    )


def train_all(
    processed_dir: Path,
    artifacts_dir: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> dict:
    _seed_everything(training_config.seed)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_sentences = load_tokenized_sentences(processed_dir / "train.jsonl")
    validation_sentences = load_tokenized_sentences(processed_dir / "validation.jsonl")
    test_sentences = load_tokenized_sentences(processed_dir / "test.jsonl")
    if not train_sentences or not validation_sentences or not test_sentences:
        raise ValueError("All three processed splits must contain sentences.")

    vocabulary = build_vocabulary(
        train_sentences, min_frequency=training_config.min_word_frequency
    )
    vocabulary.save(artifacts_dir / "vocabulary.json")
    (artifacts_dir / "model_config.json").write_text(
        json.dumps(model_config.to_dict(), indent=2), encoding="utf-8"
    )

    datasets = {
        "train": PrefixDataset(
            train_sentences, vocabulary, model_config.max_context_length
        ),
        "validation": PrefixDataset(
            validation_sentences, vocabulary, model_config.max_context_length
        ),
        "test": PrefixDataset(
            test_sentences, vocabulary, model_config.max_context_length
        ),
    }
    loaders = {
        "train": _loader(
            datasets["train"], training_config.batch_size, True, training_config.seed
        ),
        "validation": _loader(
            datasets["validation"], training_config.batch_size, False, training_config.seed
        ),
        "test": _loader(
            datasets["test"], training_config.batch_size, False, training_config.seed
        ),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    complete_results: dict[str, dict] = {
        "device": str(device),
        "vocabulary_size": len(vocabulary.token_to_id),
        "examples": {name: len(dataset) for name, dataset in datasets.items()},
        "models": {},
    }

    for model_name in ("birnn", "textcnn"):
        model = create_model(
            model_name,
            vocab_size=len(vocabulary.token_to_id),
            padding_idx=vocabulary.pad_id,
            config=model_config,
        ).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.unk_id)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        history = []
        started_at = time.perf_counter()

        for epoch in range(1, training_config.max_epochs + 1):
            model.train()
            running_loss = 0.0
            trained_examples = 0
            for contexts, targets in loaders["train"]:
                contexts = contexts.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                logits = model(contexts)
                loss = criterion(logits, targets)
                loss.backward()
                # Gradient clipping reduces unstable RNN updates.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size
                trained_examples += batch_size

            validation = evaluate(
                model, loaders["validation"], criterion, device
            )
            epoch_record = {
                "epoch": epoch,
                "train_loss": running_loss / max(trained_examples, 1),
                "validation": validation.to_dict(),
            }
            history.append(epoch_record)
            print(
                f"{model_name} epoch {epoch:02d} | "
                f"train loss {epoch_record['train_loss']:.4f} | "
                f"val loss {validation.loss:.4f} | "
                f"val top-1 {validation.top1_accuracy:.3f} | "
                f"val top-3 {validation.top3_accuracy:.3f}",
                flush=True,
            )

            if validation.loss < best_loss - 1e-4:
                best_loss = validation.loss
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= training_config.patience:
                    break

        if best_state is None:
            raise RuntimeError(f"{model_name} did not produce a checkpoint")

        model.load_state_dict(best_state)
        torch.save(best_state, artifacts_dir / f"{model_name}.pt")
        test_result = evaluate(model, loaders["test"], criterion, device)
        complete_results["models"][model_name] = {
            "best_validation_loss": best_loss,
            "test": test_result.to_dict(),
            "training_seconds": round(time.perf_counter() - started_at, 2),
            "history": history,
        }

    (artifacts_dir / "training_report.json").write_text(
        json.dumps(complete_results, indent=2), encoding="utf-8"
    )
    return complete_results

