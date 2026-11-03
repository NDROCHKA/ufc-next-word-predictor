from ufc_next_word.data.text import normalize_sentence, tokenize


def test_tokenize_preserves_apostrophe_in_fighter_name() -> None:
    assert tokenize("Sean O’Malley fought in the UFC!") == [
        "sean",
        "o'malley",
        "fought",
        "in",
        "the",
        "ufc",
    ]


def test_normalize_sentence_is_deterministic() -> None:
    assert normalize_sentence("  Khabib vs. McGregor ") == "khabib vs mcgregor"

