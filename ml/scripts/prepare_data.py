"""CLI entry point for creating leak-resistant train/validation/test corpora."""

import argparse
import json

from ufc_next_word.config import PROCESSED_DATA_DIR, RAW_DATA_PATH
from ufc_next_word.data.corpus_builder import build_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default=str(RAW_DATA_PATH))
    parser.add_argument("--output", type=str, default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--focus-fighters", type=int, default=80)
    args = parser.parse_args()

    from pathlib import Path

    report = build_corpus(
        raw_csv_path=Path(args.raw),
        output_dir=Path(args.output),
        focus_fighter_count=args.focus_fighters,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

