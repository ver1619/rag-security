
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BEIR_DIR = PROJECT_ROOT / "data/raw/beir_nq"
NQ_OPEN_FILE = PROJECT_ROOT / "data/raw/nq_open/validation.jsonl"


def count_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def inspect_qrels(path):
    with path.open("r", encoding="utf-8") as file:
        header = next(file).strip()
        first_record = next(file).strip()

    print("\nQrels header:", header)
    print("Sample qrels row:", first_record)


def main():
    corpus = BEIR_DIR / "corpus.jsonl"
    queries = BEIR_DIR / "queries.jsonl"
    qrels = BEIR_DIR / "qrels" / "test.tsv"

    print("BEIR NQ statistics")
    print("------------------")
    print("Corpus passages:", count_jsonl(corpus))
    print("Queries:", count_jsonl(queries))

    inspect_qrels(qrels)

    print("\nNQ Open statistics")
    print("------------------")
    print("Question-answer examples:", count_jsonl(NQ_OPEN_FILE))

    with NQ_OPEN_FILE.open("r", encoding="utf-8") as file:
        sample = json.loads(next(file))

    print("Question field:", type(sample["question"]).__name__)
    print("Answer field:", type(sample["answer"]).__name__)
    print("Number of sample answer variants:", len(sample["answer"]))


if __name__ == "__main__":
    main()
