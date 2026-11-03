"""Text normalization used identically during training and inference."""

import re


# Names such as O'Malley stay one token. Hyphenated words are deliberately
# separated so input typed with or without a hyphen behaves consistently.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into the project's word-level tokens."""

    return TOKEN_PATTERN.findall(text.lower().replace("’", "'"))


def normalize_sentence(text: str) -> str:
    """Return the canonical space-separated representation stored on disk."""

    return " ".join(tokenize(text))

