
import json
import random
import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = (
    PROJECT_ROOT / "data/processed/filter_v1/results/answer_supported_questions.json"
)

CORPUS_FILE = (
    PROJECT_ROOT / "data/raw/beir_nq/corpus.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "data/processed/exploration"


def load_questions():
    with DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_passages(passages_needed):
    """
    Read only the passages referenced by selected questions.
    Never load the entire 2.68M corpus into RAM.
    """

    passages = {}

    with CORPUS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            if record["_id"] in passages_needed:
                passages[record["_id"]] = record

                if len(passages) == len(passages_needed):
                    break

    return passages


def show_statistics(records):

    print("\n========== DATASET STATISTICS ==========")

    total = len(records)

    answer_counts = [
        len(record["answers"]) for record in records
    ]

    gold_counts = [
        len(record["relevant_passage_ids"]) for record in records
    ]

    supported_counts = [
        len(record["answer_supported_passage_ids"])
        for record in records
    ]

    print("Total questions:", total)
    print("Average answer variants:", sum(answer_counts) / total)
    print("Average relevant passages:", sum(gold_counts) / total)
    print("Average answer-supported passages:", sum(supported_counts) / total)

    print("\nAnswer count distribution:")
    print(Counter(answer_counts))

    print("\nRelevant passage count distribution:")
    print(Counter(gold_counts))


def inspect_random_samples(records, sample_size=5):

    rng = random.Random(42)

    samples = rng.sample(
        records,
        min(sample_size, len(records))
    )

    passage_ids = {
        passage_id
        for record in samples
        for passage_id in record["answer_supported_passage_ids"]
    }

    passages = load_passages(passage_ids)

    print("\n========== RANDOM SAMPLES ==========")

    for index, record in enumerate(samples, start=1):

        print(f"\n--- SAMPLE {index} ---")
        print("Query ID:", record["beir_query_id"])
        print("Question:", record["question"])
        print("Answers:", record["answers"])

        for passage_id in record["answer_supported_passage_ids"]:

            passage = passages.get(passage_id)

            if not passage:
                continue

            print("\nPassage ID:", passage_id)
            print("Title:", passage.get("title", ""))
            print("Text:", passage.get("text", ""))


def export_to_csv(records):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "dataset_overview.csv"

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "beir_query_id",
                "question",
                "answers",
                "relevant_passage_ids",
                "answer_supported_passage_ids"
            ]
        )

        writer.writeheader()

        for record in records:

            writer.writerow({
                "beir_query_id": record["beir_query_id"],
                "question": record["question"],
                "answers": " | ".join(record["answers"]),
                "relevant_passage_ids": " | ".join(
                    record["relevant_passage_ids"]
                ),
                "answer_supported_passage_ids": " | ".join(
                    record["answer_supported_passage_ids"]
                )
            })

    print("\nCSV exported to:", output_file)


def main():

    records = load_questions()

    show_statistics(records)

    inspect_random_samples(records, sample_size=5)

    export_to_csv(records)


if __name__ == "__main__":
    main()
