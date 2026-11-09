
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BEIR_QUERIES = (
    PROJECT_ROOT / "data/raw/beir_nq/queries.jsonl"
)

NQ_OPEN_FILE = (
    PROJECT_ROOT / "data/raw/nq_open/validation.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "data/processed/shared"

OUTPUT_FILE = OUTPUT_DIR / "matched_questions.json"
STATS_FILE = OUTPUT_DIR / "matching_statistics.json"


# --------------------------------------------------
# 2. QUESTION NORMALIZATION
# --------------------------------------------------

def normalize_question(question: str) -> str:
    """
    Normalize question text for matching.

    Operations:
    - Unicode normalization
    - Lowercase conversion
    - Punctuation removal
    - Whitespace normalization
    """

    question = unicodedata.normalize("NFKC", question)

    question = question.lower()

    # Replace punctuation and symbols with spaces.
    question = re.sub(r"[^\w\s]", " ", question)

    # Remove extra whitespace.
    question = " ".join(question.split())

    return question


# --------------------------------------------------
# 3. LOAD NQ OPEN ANSWERS
# --------------------------------------------------

def load_nq_open():
    """
    Build a mapping:
        normalized_question -> answer variants

    If duplicate normalized questions exist,
    merge their unique answer variants.
    """

    question_to_answers = defaultdict(list)

    total_examples = 0
    duplicate_questions = 0

    with NQ_OPEN_FILE.open("r", encoding="utf-8") as file:

        for line in file:

            record = json.loads(line)

            question = record["question"]
            answers = record["answer"]

            normalized = normalize_question(question)

            if normalized in question_to_answers:
                duplicate_questions += 1

            for answer in answers:

                if answer not in question_to_answers[normalized]:
                    question_to_answers[normalized].append(answer)

            total_examples += 1

    print(f"NQ Open examples: {total_examples}")
    print(f"Unique normalized questions: {len(question_to_answers)}")
    print(f"Duplicate normalized questions: {duplicate_questions}")

    return question_to_answers, total_examples, duplicate_questions


# --------------------------------------------------
# 4. MATCH BEIR QUESTIONS
# --------------------------------------------------

def match_beir_questions(question_to_answers):

    matched_records = []

    unmatched_count = 0

    total_beir_queries = 0

    with BEIR_QUERIES.open("r", encoding="utf-8") as file:

        for line in file:

            record = json.loads(line)

            beir_query_id = record["_id"]
            question = record["text"]

            normalized = normalize_question(question)

            total_beir_queries += 1

            answers = question_to_answers.get(normalized)

            if not answers:
                unmatched_count += 1
                continue

            matched_records.append({
                "beir_query_id": beir_query_id,
                "question": question,
                "normalized_question": normalized,
                "answers": answers
            })

    return matched_records, total_beir_queries, unmatched_count


# --------------------------------------------------
# 5. SAVE MATCHED QUESTIONS AND STATISTICS
# --------------------------------------------------

def save_results(
    matched_records,
    total_beir_queries,
    total_nq_open,
    unmatched_count,
    duplicate_questions
):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:

        json.dump(
            matched_records,
            file,
            indent=2,
            ensure_ascii=False
        )

    statistics = {
        "beir_queries": total_beir_queries,
        "nq_open_examples": total_nq_open,
        "matched_questions": len(matched_records),
        "unmatched_beir_questions": unmatched_count,
        "duplicate_normalized_nq_open_questions": duplicate_questions,
        "matching_method": "normalized_exact_question_match"
    }

    with STATS_FILE.open("w", encoding="utf-8") as file:

        json.dump(
            statistics,
            file,
            indent=2
        )

    print("\nMatching completed.")
    print("-------------------")

    for key, value in statistics.items():
        print(f"{key}: {value}")

    print(f"\nMatched questions saved to: {OUTPUT_FILE}")
    print(f"Statistics saved to: {STATS_FILE}")


# --------------------------------------------------
# 6. MAIN
# --------------------------------------------------

def main():

    print("Loading NQ Open answers...")

    (
        question_to_answers,
        total_nq_open,
        duplicate_questions
    ) = load_nq_open()

    print("\nMatching BEIR questions...")

    (
        matched_records,
        total_beir_queries,
        unmatched_count
    ) = match_beir_questions(question_to_answers)

    save_results(
        matched_records=matched_records,
        total_beir_queries=total_beir_queries,
        total_nq_open=total_nq_open,
        unmatched_count=unmatched_count,
        duplicate_questions=duplicate_questions
    )


if __name__ == "__main__":
    main()