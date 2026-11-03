from ufc_next_word.data.corpus_builder import _stable_split, _winner_and_loser


def test_one_bout_always_has_one_split() -> None:
    assert _stable_split("abc123") == _stable_split("abc123")


def test_winner_is_read_from_outcome_columns() -> None:
    row = {
        "header_fighter_a_name": "Fighter One",
        "header_fighter_b_name": "Fighter Two",
        "header_fighter_a_outcome": "L",
        "header_fighter_b_outcome": "W",
    }
    assert _winner_and_loser(row) == ("Fighter Two", "Fighter One")

