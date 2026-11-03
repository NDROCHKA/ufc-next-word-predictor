"""Load both checkpoints once and expose a model-independent prediction API."""

import json
from pathlib import Path

import torch

from ufc_next_word.config import ModelConfig
from ufc_next_word.data.text import tokenize
from ufc_next_word.data.vocabulary import PAD_TOKEN, UNK_TOKEN, Vocabulary
from ufc_next_word.models import create_model


class NextWordPredictor:
    def __init__(self, artifacts_dir: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocabulary = Vocabulary.load(artifacts_dir / "vocabulary.json")
        raw_config = json.loads(
            (artifacts_dir / "model_config.json").read_text(encoding="utf-8")
        )
        raw_config["cnn_kernel_sizes"] = tuple(raw_config["cnn_kernel_sizes"])
        self.config = ModelConfig(**raw_config)
        self.models = {}

        for model_name in ("birnn", "textcnn"):
            model = create_model(
                model_name,
                vocab_size=len(self.vocabulary.token_to_id),
                padding_idx=self.vocabulary.pad_id,
                config=self.config,
            )
            # weights_only prevents arbitrary pickled objects from being loaded.
            state = torch.load(
                artifacts_dir / f"{model_name}.pt",
                map_location=self.device,
                weights_only=True,
            )
            model.load_state_dict(state)
            model.to(self.device).eval()
            self.models[model_name] = model

    def _encode_context(self, text: str) -> torch.Tensor:
        tokens = tokenize(text)
        if not tokens:
            raise ValueError("Enter at least one word before requesting a prediction.")
        recent = tokens[-self.config.max_context_length :]
        encoded = self.vocabulary.encode(recent)
        padded = [self.vocabulary.pad_id] * (
            self.config.max_context_length - len(encoded)
        ) + encoded
        return torch.tensor([padded], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def predict(self, text: str, alternatives: int = 3) -> dict:
        context = self._encode_context(text)
        id_to_token = self.vocabulary.id_to_token
        results = {}
        blocked = {
            self.vocabulary.token_to_id[PAD_TOKEN],
            self.vocabulary.token_to_id[UNK_TOKEN],
        }

        for model_name, model in self.models.items():
            probabilities = torch.softmax(model(context), dim=1).squeeze(0)
            # Ask for a few extras so special tokens can be filtered safely.
            count = min(alternatives + len(blocked), probabilities.numel())
            values, indices = probabilities.topk(count)
            choices = [
                {"word": id_to_token[token_id], "confidence": round(float(value), 6)}
                for value, token_id in zip(values.tolist(), indices.tolist())
                if token_id not in blocked
            ][:alternatives]
            results[model_name] = {
                "word": choices[0]["word"],
                "confidence": choices[0]["confidence"],
                "alternatives": choices,
            }

        return {"input": " ".join(tokenize(text)), "predictions": results}

