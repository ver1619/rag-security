import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BEIR_DIR = PROJECT_ROOT / "data/raw/beir_nq"
NQ_OPEN_FILE = PROJECT_ROOT / "data/raw/nq_open/validation.jsonl"


def inspect_beir():
    print("\n--- BEIR NQ ---")

    for filename in ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]:
        path = BEIR_DIR / filename
        print(f"{filename}: {'FOUND' if path.exists() else 'MISSING'}")

    # Inspect only one record; do not load the full corpus.
    corpus_file = BEIR_DIR / "corpus.jsonl"

    with corpus_file.open("r", encoding="utf-8") as file:
        first_record = json.loads(next(file))

    print("\nSample corpus record:")
    print(first_record)


def inspect_nq_open():
    print("\n--- NQ OPEN ---")

    with NQ_OPEN_FILE.open("r", encoding="utf-8") as file:
        first_record = json.loads(next(file))

    print("Sample question-answer record:")
    print(first_record)


if __name__ == "__main__":
    inspect_beir()
    inspect_nq_open()