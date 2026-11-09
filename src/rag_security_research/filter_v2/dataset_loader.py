
"""Reconstruct the complete Filter v2 question-passage dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_matched_questions(path: Path) -> list[dict[str, Any]]:
    """Load all matched BEIR/NQ Open questions."""
    with path.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("Matched questions must be a JSON list.")

    required_fields = {"beir_query_id", "question", "answers"}

    for index, record in enumerate(questions):
        missing = required_fields - record.keys()
        if missing:
            raise ValueError(
                f"Question at index {index} is missing fields: {missing}"
            )

    return questions


def load_positive_qrels(
    path: Path,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Load positive qrels passages, preserving their original order."""
    passages_by_question: dict[str, list[str]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()

    statistics = {
        "total_qrels_rows": 0,
        "positive_qrels_rows": 0,
        "duplicate_qrels_pairs": 0,
    }

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")

        required = {"query-id", "corpus-id", "score"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Unexpected qrels columns: {reader.fieldnames}"
            )

        for row in reader:
            statistics["total_qrels_rows"] += 1

            query_id = row["query-id"].strip()
            corpus_id = row["corpus-id"].strip()
            score = int(row["score"])

            if score <= 0:
                continue

            statistics["positive_qrels_rows"] += 1
            pair = (query_id, corpus_id)

            if pair in seen_pairs:
                statistics["duplicate_qrels_pairs"] += 1
                continue

            seen_pairs.add(pair)
            passages_by_question[query_id].append(corpus_id)

    return dict(passages_by_question), statistics


def load_required_passages(
    corpus_path: Path,
    required_passage_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Stream the corpus and retain only passages referenced by qrels."""
    passages: dict[str, dict[str, Any]] = {}

    statistics = {
        "corpus_records_read": 0,
        "duplicate_corpus_ids": 0,
    }

    with corpus_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            statistics["corpus_records_read"] += 1
            record = json.loads(line)

            passage_id = record["_id"]

            if passage_id not in required_passage_ids:
                continue

            if passage_id in passages:
                statistics["duplicate_corpus_ids"] += 1
                continue

            passages[passage_id] = {
                "passage_id": passage_id,
                "title": record.get("title", ""),
                "text": record.get("text", ""),
                "metadata": record.get("metadata", {}),
            }

    return passages, statistics


def make_pair_id(
    question_id: str,
    answer_index: int,
    passage_id: str,
) -> str:
    """Create a stable pair ID from its source identifiers."""
    raw = f"{question_id}|{answer_index}|{passage_id}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"pair_{digest}"


def build_filter_v2_dataset(
    matched_questions_path: Path,
    qrels_path: Path,
    corpus_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build question records and answer-variant/passage evaluation pairs."""
    questions = load_matched_questions(matched_questions_path)

    question_ids = [q["beir_query_id"] for q in questions]
    duplicate_question_ids = [
        question_id
        for question_id, count in Counter(question_ids).items()
        if count > 1
    ]

    if duplicate_question_ids:
        raise ValueError(
            f"Duplicate question IDs found: {duplicate_question_ids[:10]}"
        )

    qrels_by_question, qrels_stats = load_positive_qrels(qrels_path)

    matched_id_set = set(question_ids)

    relevant_ids = {
        passage_id
        for question_id in matched_id_set
        for passage_id in qrels_by_question.get(question_id, [])
    }

    passages, corpus_stats = load_required_passages(
        corpus_path,
        relevant_ids,
    )

    missing_question_ids = sorted(
        matched_id_set - set(qrels_by_question)
    )

    missing_passage_ids = sorted(
        relevant_ids - set(passages)
    )

    question_records = []
    pairs = []

    for question in questions:
        question_id = question["beir_query_id"]
        answers = question["answers"]

        relevant_passage_ids = qrels_by_question.get(question_id, [])

        question_pairs = []

        for answer_index, answer in enumerate(answers):
            for passage_id in relevant_passage_ids:
                passage = passages.get(passage_id)

                if passage is None:
                    continue

                pair = {
                    "pair_id": make_pair_id(
                        question_id,
                        answer_index,
                        passage_id,
                    ),
                    "beir_query_id": question_id,
                    "question": question["question"],
                    "answer_variant_index": answer_index,
                    "answer_variant": answer,
                    "passage_id": passage_id,
                    "passage_title": passage["title"],
                    "passage_text": passage["text"],
                    "passage_metadata": passage["metadata"],
                }

                pairs.append(pair)
                question_pairs.append(pair["pair_id"])

        question_records.append({
            **question,
            "relevant_passage_ids": relevant_passage_ids,
            "available_passage_ids": [
                passage_id
                for passage_id in relevant_passage_ids
                if passage_id in passages
            ],
            "pair_ids": question_pairs,
        })

    pair_ids = [pair["pair_id"] for pair in pairs]

    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Duplicate pair IDs generated.")

    statistics = {
        "matched_question_count": len(questions),
        "questions_with_positive_qrels": len(
            matched_id_set & set(qrels_by_question)
        ),
        "questions_without_positive_qrels": len(missing_question_ids),
        "unique_relevant_passage_ids": len(relevant_ids),
        "available_passage_count": len(passages),
        "missing_question_ids": missing_question_ids,
        "missing_passage_ids": missing_passage_ids,
        "answer_variant_count": sum(
            len(q["answers"]) for q in questions
        ),
        "evaluation_pair_count": len(pairs),
        "duplicate_question_ids": duplicate_question_ids,
        "qrels": qrels_stats,
        "corpus": corpus_stats,
    }

    return question_records, {
        "statistics": statistics,
        "pairs": pairs,
    }


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as UTF-8 JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def save_dataset(
    question_records: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    statistics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save reconstructed questions, pairs and loading statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(
        output_dir / "filter_v2_questions.jsonl",
        question_records,
    )

    save_jsonl(
        output_dir / "filter_v2_pairs.jsonl",
        pairs,
    )

    stats_path = output_dir / "dataset_statistics.json"
    stats_path.write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    question_records, result = build_filter_v2_dataset(
        matched_questions_path=(
            project_root / "data/processed/shared/matched_questions.json"
        ),
        qrels_path=(
            project_root / "data/raw/beir_nq/qrels/test.tsv"
        ),
        corpus_path=(
            project_root / "data/raw/beir_nq/corpus.jsonl"
        ),
    )

    save_dataset(
        question_records=question_records,
        pairs=result["pairs"],
        statistics=result["statistics"],
        output_dir=(
            project_root / "data/processed/filter_v2/dataset"
        ),
    )

    print("\nFilter v2 dataset reconstruction complete.")
    print(json.dumps(result["statistics"], indent=2))


if __name__ == "__main__":
    main()
