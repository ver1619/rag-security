
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MATCHED_FILE = (
    PROJECT_ROOT / "data/processed/shared/matched_questions.json"
)

QRELS_FILE = (
    PROJECT_ROOT / "data/raw/beir_nq/qrels/test.tsv"
)

CORPUS_FILE = (
    PROJECT_ROOT / "data/raw/beir_nq/corpus.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "data/processed/filter_v1/results"

OUTPUT_FILE = (
    OUTPUT_DIR / "answer_supported_questions.json"
)

STATS_FILE = (
    OUTPUT_DIR / "answer_support_statistics.json"
)


def normalize_text(text: str) -> str:
    """Normalize text for answer-string comparison."""

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def contains_answer(passage_text, answers):
    """
    Return the answer variants found in a passage.

    Whitespace boundaries avoid matching a short answer
    such as '23' inside a longer number such as '123'.
    """

    normalized_passage = normalize_text(passage_text)

    padded_passage = f" {normalized_passage} "

    found_answers = []

    for answer in answers:
        normalized_answer = normalize_text(answer)

        if not normalized_answer:
            continue

        padded_answer = f" {normalized_answer} "

        if padded_answer in padded_passage:
            found_answers.append(answer)

    return found_answers


def load_matched_questions():
    with MATCHED_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    return records


def load_relevant_passages(matched_query_ids):
    """
    Build:
        query_id -> all relevant corpus passage IDs

    Only positive relevance judgments are retained.
    """

    query_to_passages = defaultdict(list)

    with QRELS_FILE.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:

            query_id = row["query-id"]
            passage_id = row["corpus-id"]
            score = int(row["score"])

            if query_id not in matched_query_ids:
                continue

            if score > 0:
                query_to_passages[query_id].append(passage_id)

    return query_to_passages


def inspect_corpus(matched_records, query_to_passages):
    """
    Stream the corpus rather than loading all 2.68M
    passages into RAM.

    Only relevant passage IDs belonging to matched
    questions are inspected.
    """

    passage_to_queries = defaultdict(list)

    for record in matched_records:

        query_id = record["beir_query_id"]

        for passage_id in query_to_passages.get(query_id, []):
            passage_to_queries[passage_id].append(query_id)

    remaining_ids = set(passage_to_queries)

    answer_supported = defaultdict(list)

    print(f"Relevant passage IDs to inspect: {len(remaining_ids)}")

    with CORPUS_FILE.open("r", encoding="utf-8") as file:

        for line in file:

            if not remaining_ids:
                break

            passage = json.loads(line)

            passage_id = passage["_id"]

            if passage_id not in remaining_ids:
                continue

            passage_text = passage.get("text", "")

            for query_id in passage_to_queries[passage_id]:

                record = next(
                    item for item in matched_records
                    if item["beir_query_id"] == query_id
                )

                found = contains_answer(
                    passage_text,
                    record["answers"]
                )

                if found:
                    answer_supported[query_id].append({
                        "passage_id": passage_id,
                        "matched_answers": found
                    })

            remaining_ids.remove(passage_id)

    if remaining_ids:
        print(
            f"Warning: {len(remaining_ids)} relevant passage IDs "
            "were not found in the corpus."
        )

    return answer_supported, len(remaining_ids)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading matched questions...")

    matched_records = load_matched_questions()

    matched_query_ids = {
        record["beir_query_id"]
        for record in matched_records
    }

    print(f"Matched questions: {len(matched_records)}")

    print("\nLoading BEIR relevance judgments...")

    query_to_passages = load_relevant_passages(
        matched_query_ids
    )

    print(f"Questions with qrels: {len(query_to_passages)}")

    print("\nInspecting relevant corpus passages...")

    answer_supported, missing_passages = inspect_corpus(
        matched_records,
        query_to_passages
    )

    supported_records = []

    for record in matched_records:

        query_id = record["beir_query_id"]

        relevant_ids = query_to_passages.get(query_id, [])

        supported_passages = answer_supported.get(
            query_id, []
        )

        if not supported_passages:
            continue

        updated_record = {
            **record,
            "relevant_passage_ids": relevant_ids,
            "answer_supported_passages": supported_passages,
            "answer_supported_passage_ids": [
                item["passage_id"]
                for item in supported_passages
            ]
        }

        supported_records.append(updated_record)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:

        json.dump(
            supported_records,
            file,
            indent=2,
            ensure_ascii=False
        )

    statistics = {
        "matched_questions": len(matched_records),
        "questions_with_qrels": len(query_to_passages),
        "answer_supported_questions": len(supported_records),
        "questions_without_answer_support": (
            len(matched_records) - len(supported_records)
        ),
        "missing_corpus_passages": missing_passages,
        "matching_method": "normalized_exact_answer_containment"
    }

    with STATS_FILE.open("w", encoding="utf-8") as file:

        json.dump(statistics, file, indent=2)

    print("\nAnswer-support validation completed.")
    print("-------------------------------------")

    for key, value in statistics.items():
        print(f"{key}: {value}")

    print(f"\nSaved questions: {OUTPUT_FILE}")
    print(f"Saved statistics: {STATS_FILE}")


if __name__ == "__main__":
    main()
