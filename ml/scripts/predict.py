"""Small terminal smoke test for saved model artifacts."""

import argparse
import json

from ufc_next_word.config import ARTIFACTS_DIR
from ufc_next_word.inference.predictor import NextWordPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="+", help="Incomplete UFC sentence")
    args = parser.parse_args()
    predictor = NextWordPredictor(ARTIFACTS_DIR)
    print(json.dumps(predictor.predict(" ".join(args.text)), indent=2))


if __name__ == "__main__":
    main()

