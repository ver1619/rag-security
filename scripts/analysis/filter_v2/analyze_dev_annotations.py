
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

# Project root:
# rag-security-research/
#   scripts/analysis/filter_v2/analyze_dev_annotations.py

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ANNOTATIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "filter_v2"
    / "golden_dataset"
    / "annotations"
)

DEFAULT_INPUT = ANNOTATIONS_DIR / "dev_annotations.jsonl"

DEFAULT_OUTPUT = ANNOTATIONS_DIR / "dev_annotation_statistics.json"


# ---------------------------------------------------------
# Annotation schema
# ---------------------------------------------------------

ALLOWED_LABELS = {
    "direct_support",
    "semantic_support",
    "partial_support",
    "incidental_match",
    "contradictory",
    "no_support",
    "ambiguous",
}

ALLOWED_CONFIDENCE_LEVELS = {
    "high",
    "medium",
    "low",
}

REQUIRED_FIELDS = {
    "pair_id",
    "beir_query_id",
    "question",
    "answer_variant",
    "passage_id",
    "passage_text",
    "classification",
    "rationale",
    "final_label",
}


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records without modifying the source file."""

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at line {line_number}"
                )

            records.append(record)

    return records


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def count_values(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    """Count values in a given annotation field."""

    counts = Counter(
        str(record.get(field))
        if record.get(field) is not None
        else "null"
        for record in records
    )

    return dict(sorted(counts.items()))


def calculate_percentage(
    numerator: int,
    denominator: int,
) -> float | None:
    """Calculate percentage safely."""

    if denominator == 0:
        return None

    return round((numerator / denominator) * 100, 2)


def find_duplicates(values: list[str]) -> list[str]:
    """Return duplicate values."""

    return sorted(
        value
        for value, count in Counter(values).items()
        if count > 1
    )


# ---------------------------------------------------------
# Annotation analysis
# ---------------------------------------------------------

def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:

    total_pairs = len(records)

    # -----------------------------------------------------
    # 1. Unique IDs and duplicate detection
    # -----------------------------------------------------

    pair_ids = [
        record["pair_id"]
        for record in records
        if isinstance(record.get("pair_id"), str)
        and record["pair_id"].strip()
    ]

    query_ids = {
        record["beir_query_id"]
        for record in records
        if isinstance(record.get("beir_query_id"), str)
        and record["beir_query_id"].strip()
    }

    duplicate_pair_ids = find_duplicates(pair_ids)

    # -----------------------------------------------------
    # 2. Required field validation
    # -----------------------------------------------------

    missing_fields = {
        field: sum(
            1
            for record in records
            if record.get(field) is None
            or (
                isinstance(record.get(field), str)
                and not record[field].strip()
            )
        )
        for field in sorted(REQUIRED_FIELDS)
    }

    records_with_missing_fields = sum(
        any(
            record.get(field) is None
            or (
                isinstance(record.get(field), str)
                and not record[field].strip()
            )
            for field in REQUIRED_FIELDS
        )
        for record in records
    )

    # -----------------------------------------------------
    # 3. LLM and human classification distributions
    # -----------------------------------------------------

    llm_distribution = count_values(
        records,
        "classification",
    )

    human_distribution = count_values(
        records,
        "final_label",
    )

    invalid_llm_labels = sorted({
        str(record.get("classification"))
        for record in records
        if record.get("classification") not in ALLOWED_LABELS
    })

    invalid_human_labels = sorted({
        str(record.get("final_label"))
        for record in records
        if record.get("final_label") not in ALLOWED_LABELS
    })

    # -----------------------------------------------------
    # 4. LLM-human agreement
    # -----------------------------------------------------

    # Derive agreement from the actual labels rather than
    # relying exclusively on the stored agreement field.

    comparable_records = [
        record
        for record in records
        if record.get("classification") in ALLOWED_LABELS
        and record.get("final_label") in ALLOWED_LABELS
    ]

    agreement_count = sum(
        record["classification"] == record["final_label"]
        for record in comparable_records
    )

    disagreement_count = (
        len(comparable_records) - agreement_count
    )

    agreement_percentage = calculate_percentage(
        agreement_count,
        len(comparable_records),
    )

    # Compare the stored agreement field with the derived value.
    inconsistent_agreement_flags = []

    for record in comparable_records:
        expected_agreement = (
            record["classification"] == record["final_label"]
        )

        stored_agreement = record.get("agreement")

        if (
            isinstance(stored_agreement, bool)
            and stored_agreement != expected_agreement
        ):
            inconsistent_agreement_flags.append(
                record.get("pair_id")
            )

    missing_agreement_flags = sum(
        not isinstance(record.get("agreement"), bool)
        for record in comparable_records
    )

    # -----------------------------------------------------
    # 5. Confidence distribution
    # -----------------------------------------------------

    confidence_distribution = count_values(
        records,
        "confidence",
    )

    invalid_confidence_values = sorted({
        str(record.get("confidence"))
        for record in records
        if record.get("confidence") is not None
        and record.get("confidence") not in ALLOWED_CONFIDENCE_LEVELS
    })

    # -----------------------------------------------------
    # 6. Evidence quote validation
    # -----------------------------------------------------

    valid_quotes = 0
    invalid_quotes = 0
    empty_quotes = 0
    invalid_quote_pair_ids = []

    for record in records:
        quote = record.get("evidence_quote")
        passage = record.get("passage_text")

        if quote is None or (
            isinstance(quote, str) and not quote.strip()
        ):
            empty_quotes += 1

        elif (
            isinstance(quote, str)
            and isinstance(passage, str)
            and quote in passage
        ):
            valid_quotes += 1

        else:
            invalid_quotes += 1
            invalid_quote_pair_ids.append(
                record.get("pair_id")
            )

    quote_validation_total = (
        valid_quotes + invalid_quotes + empty_quotes
    )

    # -----------------------------------------------------
    # 7. Question-level pair distribution
    # -----------------------------------------------------

    pairs_per_question = Counter(
        record["beir_query_id"]
        for record in records
        if isinstance(record.get("beir_query_id"), str)
        and record["beir_query_id"].strip()
    )

    pair_distribution = Counter(
        pairs_per_question.values()
    )

    # -----------------------------------------------------
    # 8. Question-level label diversity
    # -----------------------------------------------------

    labels_per_question = defaultdict(set)

    for record in records:
        query_id = record.get("beir_query_id")
        label = record.get("final_label")

        if query_id and label:
            labels_per_question[query_id].add(label)

    questions_with_multiple_final_labels = sum(
        len(labels) > 1
        for labels in labels_per_question.values()
    )

    # -----------------------------------------------------
    # 9. Assemble statistics
    # -----------------------------------------------------

    statistics = {
        "dataset": "Filter v2 DEV annotations",

        "dataset_overview": {
            "total_pairs": total_pairs,
            "unique_questions": len(query_ids),
            "unique_pair_ids": len(set(pair_ids)),
            "duplicate_pair_ids": duplicate_pair_ids,
        },

        "llm_classification": {
            "distribution": llm_distribution,
            "invalid_labels": invalid_llm_labels,
        },

        "human_final_classification": {
            "distribution": human_distribution,
            "invalid_labels": invalid_human_labels,
        },

        "llm_human_agreement": {
            "agreements": agreement_count,
            "disagreements": disagreement_count,
            "comparable_records": len(comparable_records),
            "agreement_percentage": agreement_percentage,
            "missing_stored_flags": missing_agreement_flags,
            "inconsistent_stored_flags": (
                inconsistent_agreement_flags
            ),
        },

        "confidence": {
            "distribution": confidence_distribution,
            "invalid_values": invalid_confidence_values,
        },

        "evidence_quote_validation": {
            "valid_exact_quotes": valid_quotes,
            "invalid_quotes": invalid_quotes,
            "empty_or_null_quotes": empty_quotes,
            "total_quotes_checked": quote_validation_total,
            "invalid_quote_pair_ids": invalid_quote_pair_ids,
        },

        "data_integrity": {
            "records_with_missing_required_fields": (
                records_with_missing_fields
            ),
            "missing_field_counts": missing_fields,
            "duplicate_pair_ids": duplicate_pair_ids,
        },

        "question_level_analysis": {
            "pairs_per_question_distribution": {
                str(pair_count): question_count
                for pair_count, question_count
                in sorted(pair_distribution.items())
            },
            "questions_with_multiple_final_labels": (
                questions_with_multiple_final_labels
            ),
        },
    }

    return statistics


# ---------------------------------------------------------
# Console report
# ---------------------------------------------------------

def print_distribution(
    title: str,
    distribution: dict[str, int],
) -> None:

    print(f"\n{title}")

    for label, count in distribution.items():
        print(f"  {label}: {count}")


def print_report(statistics: dict[str, Any]) -> None:

    overview = statistics["dataset_overview"]
    agreement = statistics["llm_human_agreement"]
    evidence = statistics["evidence_quote_validation"]
    integrity = statistics["data_integrity"]

    print("\n" + "=" * 55)
    print("FILTER V2 - DEV ANNOTATION STATISTICS")
    print("=" * 55)

    print("\nDATASET OVERVIEW")
    print(f"Total pairs: {overview['total_pairs']}")
    print(f"Unique questions: {overview['unique_questions']}")
    print(f"Unique pair IDs: {overview['unique_pair_ids']}")

    print_distribution(
        "LLM CLASSIFICATION DISTRIBUTION",
        statistics["llm_classification"]["distribution"],
    )

    print_distribution(
        "HUMAN FINAL LABEL DISTRIBUTION",
        statistics["human_final_classification"]["distribution"],
    )

    print_distribution(
        "CONFIDENCE DISTRIBUTION",
        statistics["confidence"]["distribution"],
    )

    print("\nLLM-HUMAN AGREEMENT")
    print(f"Agreements: {agreement['agreements']}")
    print(f"Disagreements: {agreement['disagreements']}")
    print(f"Comparable records: {agreement['comparable_records']}")
    print(f"Raw agreement: {agreement['agreement_percentage']}%")

    print("\nEVIDENCE QUOTE VALIDATION")
    print(f"Valid exact quotes: {evidence['valid_exact_quotes']}")
    print(f"Invalid quotes: {evidence['invalid_quotes']}")
    print(f"Empty/null quotes: {evidence['empty_or_null_quotes']}")

    print("\nDATA INTEGRITY")
    print(
        "Records with missing required fields:",
        integrity["records_with_missing_required_fields"],
    )
    print("Duplicate pair IDs:", overview["duplicate_pair_ids"])

    print(
        "Invalid LLM labels:",
        statistics["llm_classification"]["invalid_labels"],
    )
    print(
        "Invalid human labels:",
        statistics["human_final_classification"]["invalid_labels"],
    )
    print(
        "Inconsistent agreement flags:",
        agreement["inconsistent_stored_flags"],
    )
    print(
        "Missing agreement flags:",
        agreement["missing_stored_flags"],
    )

    print("\n" + "=" * 55)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate statistics and integrity checks "
            "for Filter v2 DEV annotations."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to dev_annotations.jsonl",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the statistics JSON output",
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {input_path}"
        )

    # Prevent accidentally overwriting the source annotations.
    if input_path == output_path:
        raise ValueError(
            "Input and output paths must be different. "
            "The annotation file must remain unchanged."
        )

    records = load_jsonl(input_path)

    if not records:
        raise ValueError("The annotation file contains no records.")

    statistics = analyze(records)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            statistics,
            file,
            indent=4,
            ensure_ascii=False,
        )
        file.write("\n")

    print_report(statistics)

    print("\nStatistics saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
