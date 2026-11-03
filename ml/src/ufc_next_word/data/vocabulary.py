"""Vocabulary creation and serialization."""

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class Vocabulary:
    token_to_id: dict[str, int]

    @property
    def id_to_token(self) -> list[str]:
        result = [""] * len(self.token_to_id)
        for token, token_id in self.token_to_id.items():
            result[token_id] = token
        return result

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"token_to_id": self.token_to_id}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(token_to_id=data["token_to_id"])


def build_vocabulary(
    tokenized_sentences: list[list[str]], min_frequency: int
) -> Vocabulary:
    """Build IDs from training text only to avoid validation/test leakage."""

    counts = Counter(token for sentence in tokenized_sentences for token in sentence)
    # Frequency first, then alphabetical order makes IDs deterministic.
    retained = sorted(
        (token for token, count in counts.items() if count >= min_frequency),
        key=lambda token: (-counts[token], token),
    )
    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    token_to_id.update({token: index + 2 for index, token in enumerate(retained)})
    return Vocabulary(token_to_id)

