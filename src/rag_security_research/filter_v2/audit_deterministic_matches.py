
"""Create a reproducible manual audit sample of deterministic matches."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


SEED = 42

SAMPLE_SIZES = {
    "random_matched": 20,
    "title_only": 10,
    "unmatched": 20,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records."""
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def select_sample(
    records: list[dict[str, Any]],
    group: str,
    sample_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Select a random sample from a group."""
    if len(records) < sample_size:
        raise ValueError(
            f"Insufficient records for {group}: "
            f"available={len(records)}, required={sample_size}"
        )

    return rng.sample(records, sample_size)


def build_audit_sample(
    pairs: list[dict[str, Any]],
    match_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join pair data with matching results and select audit records."""

    pair_by_id = {
        pair["pair_id"]: pair
        for pair in pairs
    }

    result_by_id = {
        result["pair_id"]: result
        for result in match_results
    }

    if len(pair_by_id) != len(pairs):
        raise ValueError("Duplicate pair IDs found in pair dataset.")

    if len(result_by_id) != len(match_results):
        raise ValueError("Duplicate pair IDs found in match results.")

    if set(pair_by_id) != set(result_by_id):
        raise ValueError(
            "Pair IDs do not match between pair dataset and match results."
        )

    combined = []

    for pair_id, pair in pair_by_id.items():
        result = result_by_id[pair_id]

        combined.append({
            **pair,
            "match_found": result["match_found"],
            "matching_method": result["matching_method"],
            "title_match_count": result["title_match_count"],
            "body_match_count": result["body_match_count"],
            "occurrences": result["occurrences"],
        })

    groups = {
        "random_matched": [
            record for record in combined
            if record["matching_method"] in {
                "body_only",
                "title_and_body",
            }
        ],
        "title_only": [
            record for record in combined
            if record["matching_method"] == "title_only"
        ],
        "unmatched": [
            record for record in combined
            if record["matching_method"] == "no_exact_match"
        ],
    }

    rng = random.Random(SEED)
    audit_records = []

    for group_name, sample_size in SAMPLE_SIZES.items():
        selected = select_sample(
            records=groups[group_name],
            group=group_name,
            sample_size=sample_size,
            rng=rng,
        )

        for record in selected:
            audit_records.append({
                "audit_group": group_name,
                "audit_seed": SEED,
                **record,
                "human_match_judgment": "",
                "human_notes": "",
            })

    # Shuffle the final audit order so the groups are not presented
    # as obvious consecutive blocks.
    rng.shuffle(audit_records)

    return audit_records


def save_json(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_csv(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        return

    fieldnames = [
        "audit_group",
        "audit_seed",
        "pair_id",
        "beir_query_id",
        "question",
        "answer_variant_index",
        "answer_variant",
        "passage_id",
        "passage_title",
        "passage_text",
        "match_found",
        "matching_method",
        "title_match_count",
        "body_match_count",
        "occurrences",
        "human_match_judgment",
        "human_notes",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for record in records:
            row = record.copy()

            row["occurrences"] = json.dumps(
                row["occurrences"],
                ensure_ascii=False,
            )

            writer.writerow(row)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    dataset_dir = project_root / "data/processed/filter_v2/dataset"
    matching_dir = project_root / "data/processed/filter_v2/matching"
    output_dir = project_root / "data/processed/filter_v2/manual_audit"

    pairs = load_jsonl(
        dataset_dir / "filter_v2_pairs.jsonl"
    )

    match_results = load_jsonl(
        matching_dir / "results/deterministic_match_results.jsonl"
    )

    audit_records = build_audit_sample(
        pairs=pairs,
        match_results=match_results,
    )

    save_json(
        output_dir / "audit_samples.json",
        audit_records,
    )

    save_csv(
        output_dir / "audit_samples.csv",
        audit_records,
    )

    group_counts = Counter(
        record["audit_group"]
        for record in audit_records
    )

    statistics = {
        "seed": SEED,
        "total_audit_records": len(audit_records),
        "sample_sizes": dict(group_counts),
        "unique_pair_ids": len({
            record["pair_id"]
            for record in audit_records
        }),
    }

    save_json(
        output_dir / "audit_statistics.json",
        statistics,
    )

    print("\nDeterministic matching audit created.")
    print(json.dumps(statistics, indent=2))
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
