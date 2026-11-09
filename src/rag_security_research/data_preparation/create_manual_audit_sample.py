
import csv
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
RAW_DIR = PROJECT_ROOT / "data/raw"

ACCEPTED_FILE = (
    PROCESSED_DIR / "filter_v1/results/answer_supported_questions.json"
)

MATCHED_FILE = (
    PROCESSED_DIR / "shared/matched_questions.json"
)

QRELS_FILE = (
    RAW_DIR / "beir_nq/qrels/test.tsv"
)

CORPUS_FILE = (
    RAW_DIR / "beir_nq/corpus.jsonl"
)

OUTPUT_DIR = PROCESSED_DIR / "exploration" / "manual_audit"

SEED = 42

SAMPLE_SIZES = {
    "random_accepted": 30,
    "short_answer": 10,
    "multiple_answers": 10,
    "rejected": 20,
}


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_qrels(query_ids):
    """Load all positive qrels for the selected query IDs."""

    query_to_passages = defaultdict(list)

    with QRELS_FILE.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            query_id = row["query-id"]

            if query_id not in query_ids:
                continue

            if int(row["score"]) > 0:
                query_to_passages[query_id].append(
                    row["corpus-id"]
                )

    return query_to_passages


# --------------------------------------------------
# 3. NORMALIZATION
# --------------------------------------------------

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def has_short_answer(record):
    """
    A short answer is defined here as an answer
    containing 1-3 normalized characters.

    This is an operational sampling rule, not a
    universal definition of a short answer.
    """

    return any(
        0 < len(normalize_text(answer)) <= 3
        for answer in record["answers"]
    )


# --------------------------------------------------
# 4. SELECT REPRODUCIBLE SAMPLES
# --------------------------------------------------

def select_samples(accepted, rejected):

    rng = random.Random(SEED)

    selected = {}
    categories = defaultdict(list)

    # Group 1: random accepted records
    random_accepted = rng.sample(
        accepted,
        min(SAMPLE_SIZES["random_accepted"], len(accepted))
    )

    for record in random_accepted:
        categories[record["beir_query_id"]].append(
            "random_accepted"
        )

    # Group 2: accepted records with short answers
    short_records = [
        record for record in accepted
        if has_short_answer(record)
    ]

    short_sample = rng.sample(
        short_records,
        min(SAMPLE_SIZES["short_answer"], len(short_records))
    )

    for record in short_sample:
        categories[record["beir_query_id"]].append(
            "short_answer"
        )

    # Group 3: accepted records with multiple answers
    multiple_records = [
        record for record in accepted
        if len(record["answers"]) > 1
    ]

    multiple_sample = rng.sample(
        multiple_records,
        min(
            SAMPLE_SIZES["multiple_answers"],
            len(multiple_records)
        )
    )

    for record in multiple_sample:
        categories[record["beir_query_id"]].append(
            "multiple_answers"
        )

    # Group 4: rejected records
    rejected_sample = rng.sample(
        rejected,
        min(SAMPLE_SIZES["rejected"], len(rejected))
    )

    for record in rejected_sample:
        categories[record["beir_query_id"]].append(
            "rejected"
        )

    # Combine records by unique question ID.
    record_lookup = {
        record["beir_query_id"]: record
        for record in accepted + rejected
    }

    for query_id, group_names in categories.items():

        selected[query_id] = {
            "record": record_lookup[query_id],
            "sampling_groups": group_names
        }

    return selected


# --------------------------------------------------
# 5. LOAD RELEVANT PASSAGE TEXT
# --------------------------------------------------

def load_selected_passages(required_ids):

    passages = {}

    print(f"Unique passages required: {len(required_ids)}")
    print("Scanning BEIR corpus...")

    with CORPUS_FILE.open("r", encoding="utf-8") as file:

        for line in file:

            record = json.loads(line)
            passage_id = record["_id"]

            if passage_id not in required_ids:
                continue

            passages[passage_id] = {
                "passage_id": passage_id,
                "title": record.get("title", ""),
                "text": record.get("text", "")
            }

            if len(passages) == len(required_ids):
                break

    missing = required_ids - set(passages)

    if missing:
        print(f"WARNING: {len(missing)} passage IDs were not found.")

    return passages


# --------------------------------------------------
# 6. BUILD AUDIT RECORDS
# --------------------------------------------------

def build_audit_records(selected, query_to_passages, passages):

    audit_records = []

    for query_id, item in selected.items():

        record = item["record"]

        relevant_ids = query_to_passages.get(query_id, [])

        evidence = [
            passages[passage_id]
            for passage_id in relevant_ids
            if passage_id in passages
        ]

        is_accepted = bool(
            record.get("answer_supported_passage_ids")
        )

        audit_records.append({

            "beir_query_id": query_id,

            "question": record["question"],

            "answers": record["answers"],

            "sampling_groups": item["sampling_groups"],

            "automated_filter_result": (
                "accepted" if is_accepted else "rejected"
            ),

            "relevant_passage_ids": relevant_ids,

            "evidence_passages": evidence,

            # Fill these fields during manual inspection.
            "manual_judgment": "",

            "audit_notes": ""
        })

    audit_records.sort(
        key=lambda record: record["beir_query_id"]
    )

    return audit_records


# --------------------------------------------------
# 7. EXPORT JSON AND CSV
# --------------------------------------------------

def export_audit(audit_records):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "audit_samples.json"

    csv_path = OUTPUT_DIR / "audit_samples.csv"

    with json_path.open("w", encoding="utf-8") as file:

        json.dump(
            audit_records,
            file,
            indent=2,
            ensure_ascii=False
        )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        fieldnames = [
            "beir_query_id",
            "question",
            "answers",
            "sampling_groups",
            "automated_filter_result",
            "relevant_passage_ids",
            "evidence_passages",
            "manual_judgment",
            "audit_notes"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for record in audit_records:

            evidence_text = "\n\n".join(
                (
                    f"Passage ID: {passage['passage_id']}\n"
                    f"Title: {passage['title']}\n"
                    f"Text: {passage['text']}"
                )
                for passage in record["evidence_passages"]
            )

            writer.writerow({
                "beir_query_id": record["beir_query_id"],
                "question": record["question"],
                "answers": " | ".join(record["answers"]),
                "sampling_groups": " | ".join(
                    record["sampling_groups"]
                ),
                "automated_filter_result": (
                    record["automated_filter_result"]
                ),
                "relevant_passage_ids": " | ".join(
                    record["relevant_passage_ids"]
                ),
                "evidence_passages": evidence_text,
                "manual_judgment": "",
                "audit_notes": ""
            })

    print("\nAudit files created:")
    print("JSON:", json_path)
    print("CSV:", csv_path)


# --------------------------------------------------
# 8. MAIN
# --------------------------------------------------

def main():

    print("Loading accepted and rejected questions...")

    accepted = load_json(ACCEPTED_FILE)
    matched = load_json(MATCHED_FILE)

    accepted_ids = {
        record["beir_query_id"]
        for record in accepted
    }

    rejected = [
        record for record in matched
        if record["beir_query_id"] not in accepted_ids
    ]

    print("Accepted questions:", len(accepted))
    print("Rejected questions:", len(rejected))

    print("\nSelecting audit samples...")

    selected = select_samples(accepted, rejected)

    print("Unique selected questions:", len(selected))

    for category in SAMPLE_SIZES:
        count = sum(
            category in item["sampling_groups"]
            for item in selected.values()
        )

        print(f"{category}: {count}")

    selected_ids = set(selected)

    print("\nLoading qrels...")

    query_to_passages = load_qrels(selected_ids)

    required_passage_ids = {
        passage_id
        for passage_list in query_to_passages.values()
        for passage_id in passage_list
    }

    passages = load_selected_passages(
        required_passage_ids
    )

    print("\nBuilding audit records...")

    audit_records = build_audit_records(
        selected,
        query_to_passages,
        passages
    )

    export_audit(audit_records)

    print("\nAudit sample preparation completed.")


if __name__ == "__main__":
    main()
