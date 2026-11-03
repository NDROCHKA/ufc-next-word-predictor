"""PyTorch dataset for prefix-to-next-word learning examples."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .text import tokenize
from .vocabulary import Vocabulary


def load_tokenized_sentences(path: Path) -> list[list[str]]:
    sentences: list[list[str]] = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            tokens = tokenize(record["sentence"])
            if len(tokens) >= 2:
                sentences.append(tokens)
    return sentences


class PrefixDataset(Dataset):
    """Create every valid sentence prefix and its following target word."""

    def __init__(
        self,
        tokenized_sentences: list[list[str]],
        vocabulary: Vocabulary,
        max_context_length: int,
    ) -> None:
        self.examples: list[tuple[list[int], int]] = []
        self.pad_id = vocabulary.pad_id
        self.max_context_length = max_context_length

        for tokens in tokenized_sentences:
            encoded = vocabulary.encode(tokens)
            for target_position in range(1, len(encoded)):
                context = encoded[max(0, target_position - max_context_length) : target_position]
                target = encoded[target_position]
                # A word excluded by the training vocabulary cannot be learned
                # as a class. Excluding that example keeps loss and accuracy
                # denominators meaningful across every split.
                if target == vocabulary.unk_id:
                    continue
                self.examples.append((context, target))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        context, target = self.examples[index]
        # Left padding keeps the newest words at the right edge. Both models see
        # a consistent fixed length while padding stays distinguishable.
        padded = [self.pad_id] * (self.max_context_length - len(context)) + context
        return torch.tensor(padded, dtype=torch.long), torch.tensor(target, dtype=torch.long)

