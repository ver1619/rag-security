
import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_QUESTIONS = (
    PROJECT_ROOT / "data/processed/filter_v2/dataset/filter_v2_questions.jsonl"
)
DEFAULT_PAIRS = (
    PROJECT_ROOT / "data/processed/filter_v2/dataset/filter_v2_pairs.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/processed/filter_v2/golden_dataset"
)


def read_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

    return records


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate_dataset(questions, pairs):
    question_map = {}
    pair_map = {}
    pairs_by_question = defaultdict(list)

    # Validate question records.
    for question in questions:
        question_id = question.get("beir_query_id")

        if not question_id:
            raise ValueError("A question has no beir_query_id.")

        if question_id in question_map:
            raise ValueError(
                f"Duplicate question ID: {question_id}"
            )

        question_map[question_id] = question

    # Validate pair records.
    for pair in pairs:
        pair_id = pair.get("pair_id")
        question_id = pair.get("beir_query_id")

        if not pair_id:
            raise ValueError("An evaluation pair has no pair_id.")

        if pair_id in pair_map:
            raise ValueError(f"Duplicate pair ID: {pair_id}")

        if question_id not in question_map:
            raise ValueError(
                f"Pair {pair_id} references unknown question "
                f"{question_id}."
            )

        if pair.get("question") != question_map[question_id].get("question"):
            raise ValueError(
                f"Question text mismatch for pair {pair_id}."
            )

        pair_map[pair_id] = pair
        pairs_by_question[question_id].append(pair)

    # Verify that every question's declared pair IDs are complete.
    for question_id, question in question_map.items():
        declared_ids = question.get("pair_ids", [])
        actual_ids = [
            pair["pair_id"]
            for pair in pairs_by_question[question_id]
        ]

        if len(declared_ids) != len(set(declared_ids)):
            raise ValueError(
                f"Duplicate pair IDs in question {question_id}."
            )

        if set(declared_ids) != set(actual_ids):
            missing = set(declared_ids) - set(actual_ids)
            unexpected = set(actual_ids) - set(declared_ids)

            raise ValueError(
                f"Pair membership mismatch for {question_id}. "
                f"Missing: {sorted(missing)}; "
                f"Unexpected: {sorted(unexpected)}"
            )

    return question_map, pairs_by_question


def main():
    parser = argparse.ArgumentParser(
        description="Create reproducible question-level DEV/TEST splits."
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=DEFAULT_PAIRS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--dev-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files.",
    )

    args = parser.parse_args()

    if args.dev_size <= 0 or args.test_size <= 0:
        raise ValueError("DEV and TEST sizes must be positive.")

    output_files = [
        "dev_questions.jsonl",
        "test_questions.jsonl",
        "dev_pairs.jsonl",
        "test_pairs.jsonl",
        "sampling_metadata.json",
    ]

    args.output.mkdir(parents=True, exist_ok=True)

    existing = [
        args.output / name
        for name in output_files
        if (args.output / name).exists()
    ]

    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist. Use --overwrite only "
            "if you intentionally want to replace them:\n"
            + "\n".join(str(path) for path in existing)
        )

    print("Loading source datasets...")

    questions = read_jsonl(args.questions)
    pairs = read_jsonl(args.pairs)

    print(f"Questions loaded: {len(questions)}")
    print(f"Evaluation pairs loaded: {len(pairs)}")

    question_map, pairs_by_question = validate_dataset(
        questions, pairs
    )

    total_required = args.dev_size + args.test_size

    if total_required > len(question_map):
        raise ValueError(
            f"Requested {total_required} questions, "
            f"but only {len(question_map)} are available."
        )

    print("Dataset integrity checks passed.")

    # Sort before shuffling to make the split independent
    # of the original question-file row order.
    question_ids = sorted(question_map.keys())

    rng = random.Random(args.seed)
    rng.shuffle(question_ids)

    dev_ids = set(question_ids[:args.dev_size])
    test_ids = set(
        question_ids[args.dev_size:total_required]
    )

    if dev_ids & test_ids:
        raise AssertionError("DEV and TEST question overlap detected.")

    # Preserve original file order in the output records.
    dev_questions = [
        q for q in questions
        if q["beir_query_id"] in dev_ids
    ]
    test_questions = [
        q for q in questions
        if q["beir_query_id"] in test_ids
    ]

    dev_pairs = [
        p for p in pairs
        if p["beir_query_id"] in dev_ids
    ]
    test_pairs = [
        p for p in pairs
        if p["beir_query_id"] in test_ids
    ]

    # Verify that every selected question retains all its pairs.
    expected_dev_pairs = sum(
        len(pairs_by_question[qid]) for qid in dev_ids
    )
    expected_test_pairs = sum(
        len(pairs_by_question[qid]) for qid in test_ids
    )

    if len(dev_pairs) != expected_dev_pairs:
        raise AssertionError("DEV pair count mismatch.")

    if len(test_pairs) != expected_test_pairs:
        raise AssertionError("TEST pair count mismatch.")

    # Verify that no pair appears in both splits.
    dev_pair_ids = {p["pair_id"] for p in dev_pairs}
    test_pair_ids = {p["pair_id"] for p in test_pairs}

    if dev_pair_ids & test_pair_ids:
        raise AssertionError("Pair overlap detected.")

    # Save split files.
    output_paths = {
        "dev_questions": args.output / "dev_questions.jsonl",
        "test_questions": args.output / "test_questions.jsonl",
        "dev_pairs": args.output / "dev_pairs.jsonl",
        "test_pairs": args.output / "test_pairs.jsonl",
    }

    write_jsonl(output_paths["dev_questions"], dev_questions)
    write_jsonl(output_paths["test_questions"], test_questions)
    write_jsonl(output_paths["dev_pairs"], dev_pairs)
    write_jsonl(output_paths["test_pairs"], test_pairs)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_method": (
            "Sort unique question IDs, shuffle with a fixed seed, "
            "then select DEV followed by TEST."
        ),
        "seed": args.seed,
        "dev_question_count": len(dev_questions),
        "test_question_count": len(test_questions),
        "dev_pair_count": len(dev_pairs),
        "test_pair_count": len(test_pairs),
        "total_source_question_count": len(questions),
        "total_source_pair_count": len(pairs),
        "unused_question_count": (
            len(questions) - len(dev_questions) - len(test_questions)
        ),
        "unused_pair_count": (
            len(pairs) - len(dev_pairs) - len(test_pairs)
        ),
        "dev_question_ids": sorted(dev_ids),
        "test_question_ids": sorted(test_ids),
        "dev_test_question_overlap": 0,
        "dev_test_pair_overlap": 0,
        "input_files": {
            "questions": {
                "path": str(args.questions),
                "sha256": sha256_file(args.questions),
            },
            "pairs": {
                "path": str(args.pairs),
                "sha256": sha256_file(args.pairs),
            },
        },
        "output_sha256": {
            name: sha256_file(path)
            for name, path in output_paths.items()
        },
    }

    metadata_path = args.output / "sampling_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print("\nSampling completed successfully.")
    print(f"DEV:  {len(dev_questions)} questions, {len(dev_pairs)} pairs")
    print(f"TEST: {len(test_questions)} questions, {len(test_pairs)} pairs")
    print(f"Unused questions: {metadata['unused_question_count']}")
    print(f"Unused pairs: {metadata['unused_pair_count']}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
