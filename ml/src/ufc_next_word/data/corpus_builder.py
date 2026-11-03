"""Turn verified UFCStats-derived bout rows into next-word sentences.

The raw source supplies facts. Templates only provide grammatical wording; they
never invent a matchup, winner, division, round, referee, or event.
"""

from collections import Counter
import csv
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .text import normalize_sentence


START_DATE = date(2018, 1, 1)
END_DATE = date(2025, 12, 31)

# These names are retained only when they occur in the source CSV. Misspelled or
# absent entries are reported, never silently turned into fictional records.
FEATURED_FIGHTERS = {
    "Khabib Nurmagomedov",
    "Conor McGregor",
    "Islam Makhachev",
    "Jon Jones",
    "Alex Pereira",
    "Israel Adesanya",
    "Max Holloway",
    "Dustin Poirier",
    "Justin Gaethje",
    "Charles Oliveira",
    "Alexander Volkanovski",
    "Sean O'Malley",
    "Francis Ngannou",
    "Stipe Miocic",
    "Kamaru Usman",
    "Leon Edwards",
    "Amanda Nunes",
    "Valentina Shevchenko",
    "Rose Namajunas",
    "Zhang Weili",
}

GENERAL_TRAINING_SENTENCES = [
    "the ufc bout takes place inside the octagon",
    "a ufc fight can end by knockout submission or decision",
    "the referee can stop the fight",
    "the judges score each completed round",
    "a title fight is normally scheduled for five rounds",
    "a non title ufc bout is normally scheduled for three rounds",
    "fighters compete in a specific weight division",
    "a fighter can attack with punches kicks knees and elbows",
    "a takedown moves the fight to the ground",
    "a submission can force an opponent to tap",
    "significant strikes can target the head body or legs",
    "control time measures how long a fighter controls the opponent",
]


def _parse_date(value: str) -> date:
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def _stable_split(source_id: str) -> str:
    """Assign every variant of one bout to the same deterministic split."""

    bucket = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _winner_and_loser(row: dict[str, str]) -> tuple[str, str] | None:
    a_name = row["header_fighter_a_name"].strip()
    b_name = row["header_fighter_b_name"].strip()
    a_outcome = row["header_fighter_a_outcome"].strip().upper()
    b_outcome = row["header_fighter_b_outcome"].strip().upper()
    if a_outcome == "W" and b_outcome == "L":
        return a_name, b_name
    if b_outcome == "W" and a_outcome == "L":
        return b_name, a_name
    return None


def _sentences_for_bout(row: dict[str, str]) -> Iterable[tuple[str, str]]:
    """Yield (template_id, sentence) pairs using fields from one source row."""

    fighter_a = row["header_fighter_a_name"].strip()
    fighter_b = row["header_fighter_b_name"].strip()
    division = row["weight_class"].strip()
    event = row["event_name"].strip()
    referee = row["header_referee_detailed"].strip()
    round_number = row["header_round_detailed"].strip()

    yield "matchup_faced", f"{fighter_a} faced {fighter_b} in the ufc"
    yield "matchup_reverse", f"{fighter_b} faced {fighter_a} in the ufc"
    yield "matchup_versus", f"{fighter_a} versus {fighter_b}"
    yield "event_featured", f"{event} featured {fighter_a} versus {fighter_b}"
    yield "division_a", f"{fighter_a} competed in a {division} bout"
    yield "division_b", f"{fighter_b} competed in a {division} bout"

    winner_loser = _winner_and_loser(row)
    if winner_loser:
        winner, loser = winner_loser
        yield "winner_defeated", f"{winner} defeated {loser} in the ufc"
        yield "winner_against", f"{winner} won against {loser}"
        yield "loser_lost", f"{loser} lost to {winner}"

    if referee:
        yield "referee", f"the referee for {fighter_a} versus {fighter_b} was {referee}"
    if round_number.isdigit():
        yield "round", f"{fighter_a} versus {fighter_b} ended in round {round_number}"


def build_corpus(
    raw_csv_path: Path,
    output_dir: Path,
    focus_fighter_count: int = 80,
) -> dict:
    """Filter, validate, generate, split, and write the modern UFC corpus."""

    with raw_csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))

    modern_rows = [
        row
        for row in rows
        if row.get("event_date_iso")
        and START_DATE <= _parse_date(row["event_date_iso"]) <= END_DATE
        and row.get("FightID_generated")
        and row.get("header_fighter_a_name")
        and row.get("header_fighter_b_name")
    ]

    appearances = Counter(
        name
        for row in modern_rows
        for name in (
            row["header_fighter_a_name"].strip(),
            row["header_fighter_b_name"].strip(),
        )
    )
    top_fighters = {name for name, _ in appearances.most_common(focus_fighter_count)}
    present_featured = FEATURED_FIGHTERS.intersection(appearances)
    focus_fighters = top_fighters.union(present_featured)

    focused_rows = [
        row
        for row in modern_rows
        if row["header_fighter_a_name"].strip() in focus_fighters
        or row["header_fighter_b_name"].strip() in focus_fighters
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "validation", "test")
    }
    counts = Counter()
    template_counts = Counter()
    try:
        for row in focused_rows:
            source_id = row["FightID_generated"].strip()
            split = _stable_split(source_id)
            seen_for_bout: set[str] = set()
            for template_id, sentence in _sentences_for_bout(row):
                normalized = normalize_sentence(sentence)
                if len(normalized.split()) < 2 or normalized in seen_for_bout:
                    continue
                seen_for_bout.add(normalized)
                record = {
                    "source_id": source_id,
                    "source_url": row["fight_stats_page_url"].strip(),
                    "template_id": template_id,
                    "sentence": normalized,
                }
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
                template_counts[template_id] += 1

        # General domain language is authored background material, so it is
        # training-only and cannot inflate validation or test performance.
        for index, sentence in enumerate(GENERAL_TRAINING_SENTENCES):
            record = {
                "source_id": f"authored-domain-{index:02d}",
                "source_url": None,
                "template_id": "domain_background",
                "sentence": normalize_sentence(sentence),
            }
            handles["train"].write(json.dumps(record) + "\n")
            counts["train"] += 1
            template_counts["domain_background"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    report = {
        "source_file": str(raw_csv_path),
        "source_dataset": "01x3ATM3/UFC_FIGHT_DATA_2013_TO_2025",
        "upstream_source": "http://ufcstats.com",
        "date_range": {"start": START_DATE.isoformat(), "end": END_DATE.isoformat()},
        "raw_rows": len(rows),
        "modern_rows": len(modern_rows),
        "focused_bouts": len(focused_rows),
        "focus_fighter_count": len(focus_fighters),
        "focus_fighters": sorted(focus_fighters),
        "featured_fighters_missing_from_source": sorted(FEATURED_FIGHTERS - appearances.keys()),
        "sentences_by_split": dict(counts),
        "sentences_by_template": dict(template_counts),
    }
    (output_dir / "corpus_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report

