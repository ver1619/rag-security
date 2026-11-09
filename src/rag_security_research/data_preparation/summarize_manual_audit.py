
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "exploration"
    / "manual_audit"
)

INPUT_FILE = AUDIT_DIR / "audit_samples_updated.json"

OUTPUT_JSON = AUDIT_DIR / "audit_statistics.json"
OUTPUT_CSV = AUDIT_DIR / "audit_classification_counts.csv"


# --------------------------------------------------
# LOAD AUDIT DATA
# --------------------------------------------------

def load_audit_data():
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Updated JSON format:
    # {
    #   "audit_summary": {...},
    #   "records": [...]
    # }

    if isinstance(data, dict):
        records = data.get("records", [])
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Unsupported audit JSON structure.")

    if not records:
        raise ValueError("No audit records found.")

    return records


# --------------------------------------------------
# BASIC STATISTICS
# --------------------------------------------------

def calculate_basic_statistics(records):
    total = len(records)

    accepted = [
        r for r in records
        if r.get("automated_filter_result") == "accepted"
    ]

    rejected = [
        r for r in records
        if r.get("automated_filter_result") == "rejected"
    ]

    return {
        "total_records": total,
        "automated_accepted": len(accepted),
        "automated_rejected": len(rejected),
        "accepted_percentage": round(
            len(accepted) / total * 100, 2
        ),
        "rejected_percentage": round(
            len(rejected) / total * 100, 2
        ),
    }


# --------------------------------------------------
# MANUAL CLASSIFICATION STATISTICS
# --------------------------------------------------

def calculate_classification_statistics(records):
    counts = Counter(
        r.get("manual_judgment", "unclassified")
        for r in records
    )

    total = len(records)

    result = {}

    for label, count in sorted(counts.items()):
        result[label] = {
            "count": count,
            "percentage": round(count / total * 100, 2)
        }

    return result


# --------------------------------------------------
# SAMPLING GROUP STATISTICS
# --------------------------------------------------

def calculate_sampling_group_statistics(records):
    group_counts = defaultdict(Counter)

    for record in records:
        groups = record.get("sampling_groups", [])

        for group in groups:
            judgment = record.get(
                "manual_judgment", "unclassified"
            )

            group_counts[group][judgment] += 1

    result = {}

    for group, counts in sorted(group_counts.items()):
        result[group] = {
            "total_records": sum(counts.values()),
            "classification_counts": dict(counts)
        }

    return result


# --------------------------------------------------
# AUTOMATED FILTER VS MANUAL JUDGMENT
# --------------------------------------------------

def calculate_filter_comparison(records):
    comparison = defaultdict(Counter)

    for record in records:
        automated = record.get(
            "automated_filter_result", "unknown"
        )

        manual = record.get(
            "manual_judgment", "unclassified"
        )

        comparison[automated][manual] += 1

    return {
        automated: dict(classifications)
        for automated, classifications
        in sorted(comparison.items())
    }


# --------------------------------------------------
# ANSWER AND PASSAGE STATISTICS
# --------------------------------------------------

def calculate_content_statistics(records):
    answer_counts = []
    passage_counts = []
    supported_passage_counts = []

    for record in records:
        answers = record.get("answers", [])
        passages = record.get("evidence_passages", [])

        answer_counts.append(len(answers))
        passage_counts.append(len(passages))

        supported_ids = record.get(
            "answer_supported_passage_ids", []
        )

        supported_passage_counts.append(len(supported_ids))

    return {
        "total_answer_variants": sum(answer_counts),
        "average_answer_variants": round(
            mean(answer_counts), 2
        ),
        "average_evidence_passages": round(
            mean(passage_counts), 2
        ),
        "average_answer_supported_passages": (
            round(mean(supported_passage_counts), 2)
            if any(supported_passage_counts)
            else None
        ),
        "maximum_answer_variants": max(answer_counts),
        "maximum_evidence_passages": max(passage_counts),
    }


# --------------------------------------------------
# EXPORT CLASSIFICATION CSV
# --------------------------------------------------

def export_classification_csv(records):
    counts = Counter(
        (
            r.get("automated_filter_result", "unknown"),
            r.get("manual_judgment", "unclassified")
        )
        for r in records
    )

    with OUTPUT_CSV.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "automated_filter_result",
            "manual_judgment",
            "record_count"
        ])

        for (automated, manual), count in sorted(counts.items()):
            writer.writerow([automated, manual, count])


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    records = load_audit_data()

    statistics = {
        "basic_statistics": calculate_basic_statistics(records),
        "manual_classifications": (
            calculate_classification_statistics(records)
        ),
        "sampling_group_statistics": (
            calculate_sampling_group_statistics(records)
        ),
        "automated_vs_manual": (
            calculate_filter_comparison(records)
        ),
        "content_statistics": (
            calculate_content_statistics(records)
        ),
        "methodological_note": (
            "These statistics describe the supplied stratified "
            "audit sample only. They must not be interpreted as "
            "population-wide precision, recall, or error rates."
        )
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(statistics, file, indent=4, ensure_ascii=False)

    export_classification_csv(records)

    print("\n========== AUDIT STATISTICS ==========\n")

    print("Total records:",
          statistics["basic_statistics"]["total_records"])

    print("\nAutomated filter:")
    print(statistics["basic_statistics"])

    print("\nManual classifications:")
    for label, values in statistics[
        "manual_classifications"
    ].items():
        print(f"{label}: {values}")

    print("\nSampling groups:")
    for group, values in statistics[
        "sampling_group_statistics"
    ].items():
        print(f"{group}: {values}")

    print("\nAutomated vs Manual:")
    print(json.dumps(
        statistics["automated_vs_manual"],
        indent=4
    ))

    print("\nContent statistics:")
    print(statistics["content_statistics"])

    print("\nGenerated files:")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
