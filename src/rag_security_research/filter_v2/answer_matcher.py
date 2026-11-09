
"""Deterministic answer matching for Filter v2.

Searches passage bodies only.
This module identifies candidate occurrences; it does not
determine whether an occurrence actually supports an answer.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


def normalize_with_offsets(
    text: str,
) -> tuple[str, list[tuple[int, int]]]:
    """Normalize text while preserving original character offsets.

    Handles a specific mojibake artefact: U+00C2 (Â)
    immediately followed by U+00A0 (non-breaking space).

    Original source text is never modified.
    """
    normalized_chars: list[str] = []
    offsets: list[tuple[int, int]] = []

    for original_index, original_char in enumerate(text):

        # Targeted repair:
        # Skip Â only when immediately followed by NBSP.
        # The following NBSP will become a normal space.
        if (
            original_char == "\u00c2"
            and original_index + 1 < len(text)
            and text[original_index + 1] == "\u00a0"
        ):
            continue

        transformed = unicodedata.normalize(
            "NFKC", original_char
        ).casefold()

        for char in transformed:
            if (
                char.isspace()
                or unicodedata.category(char).startswith("P")
            ):
                char = " "

            normalized_chars.append(char)
            offsets.append(
                (original_index, original_index + 1)
            )

    # Collapse whitespace while preserving original offsets.
    collapsed_chars: list[str] = []
    collapsed_offsets: list[tuple[int, int]] = []

    for char, offset in zip(normalized_chars, offsets):
        if char == " ":
            if not collapsed_chars or collapsed_chars[-1] == " ":
                if collapsed_chars:
                    previous_start = collapsed_offsets[-1][0]
                    collapsed_offsets[-1] = (
                        previous_start,
                        offset[1],
                    )
                continue

        collapsed_chars.append(char)
        collapsed_offsets.append(offset)

    # Remove leading whitespace.
    while collapsed_chars and collapsed_chars[0] == " ":
        collapsed_chars.pop(0)
        collapsed_offsets.pop(0)

    # Remove trailing whitespace.
    while collapsed_chars and collapsed_chars[-1] == " ":
        collapsed_chars.pop()
        collapsed_offsets.pop()

    return "".join(collapsed_chars), collapsed_offsets


def find_answer_occurrences(
    answer: str,
    source_text: str,
) -> list[dict[str, Any]]:
    """Find exact normalized occurrences in the original text."""
    normalized_answer, _ = normalize_with_offsets(answer)
    normalized_text, offsets = normalize_with_offsets(source_text)

    if not normalized_answer or not normalized_text:
        return []

    matches = []
    search_from = 0

    while True:
        start = normalized_text.find(
            normalized_answer,
            search_from,
        )

        if start == -1:
            break

        end = start + len(normalized_answer)

        # Require token boundaries.
        left_valid = (
            start == 0 or normalized_text[start - 1] == " "
        )
        right_valid = (
            end == len(normalized_text)
            or normalized_text[end] == " "
        )

        if left_valid and right_valid:
            original_start = offsets[start][0]
            original_end = offsets[end - 1][1]

            matches.append({
                "matched_text": source_text[
                    original_start:original_end
                ],
                "normalized_match": normalized_answer,
                "normalized_start": start,
                "normalized_end": end,
                "original_start": original_start,
                "original_end": original_end,
            })

        # Advance by one to retain overlapping occurrences.
        search_from = start + 1

    return matches


def match_pair(pair: dict[str, Any]) -> dict[str, Any]:
    """Search a pair's answer in passage_text only.

    passage_title is deliberately excluded from matching.
    """
    answer = pair["answer_variant"]
    passage_text = pair.get("passage_text", "")

    body_matches = find_answer_occurrences(
        answer,
        passage_text,
    )

    occurrences = [
        {
            **match,
            "match_source": "body",
        }
        for match in body_matches
    ]

    matching_method = (
        "body_only" if body_matches else "no_exact_match"
    )

    return {
        "pair_id": pair["pair_id"],
        "beir_query_id": pair["beir_query_id"],
        "answer_variant_index": pair["answer_variant_index"],
        "answer_variant": answer,
        "passage_id": pair["passage_id"],
        "match_found": bool(body_matches),
        "matching_method": matching_method,
        "title_match_count": 0,
        "body_match_count": len(body_matches),
        "occurrences": occurrences,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON Lines records."""
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def save_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Save JSON Lines records."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def run_matching(
    pairs_path: Path,
    output_path: Path,
    statistics_path: Path,
) -> dict[str, Any]:
    """Run deterministic body-only matching on all evaluation pairs."""
    pairs = load_jsonl(pairs_path)

    results = [
        match_pair(pair)
        for pair in pairs
    ]

    method_counts = Counter(
        result["matching_method"]
        for result in results
    )

    statistics = {
        "total_pairs": len(results),
        "matched_pairs": sum(
            result["match_found"] for result in results
        ),
        "unmatched_pairs": sum(
            not result["match_found"] for result in results
        ),
        "body_only_pairs": method_counts["body_only"],
        "no_exact_match_pairs": method_counts["no_exact_match"],
        "total_body_occurrences": sum(
            result["body_match_count"] for result in results
        ),
        "matching_method_counts": dict(method_counts),
    }

    save_jsonl(output_path, results)

    statistics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistics_path.write_text(
        json.dumps(statistics, indent=2),
        encoding="utf-8",
    )

    return statistics


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    dataset_dir = project_root / "data/processed/filter_v2/dataset"
    matching_dir = project_root / "data/processed/filter_v2/matching"

    statistics = run_matching(
        pairs_path=dataset_dir / "filter_v2_pairs.jsonl",
        output_path=(
            matching_dir / "results/deterministic_match_results_body_only.jsonl"
        ),
        statistics_path=(
            matching_dir / "statistics/deterministic_match_statistics_body_only.json"
        ),
    )

    print("\nDeterministic body-only matching complete.")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
