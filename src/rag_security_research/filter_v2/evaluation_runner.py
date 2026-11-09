"""Run Filter v2 contextual evaluation over prepared question-passage pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_loader import get_judge_config, load_config
from .evaluator import ContextualEvaluator
from .judge.base import JudgeRequest
from .judge.provider import create_contextual_judge


COMPLETED_STATUSES = {"success", "review_required"}
MANIFEST_VERSION = 1


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records and reject malformed or duplicate pair IDs."""
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}, line {line_number}"
                )
            pair_id = record.get("pair_id")
            if not isinstance(pair_id, str) or not pair_id:
                raise ValueError(
                    f"Missing or invalid pair_id in {path}, line {line_number}"
                )
            if pair_id in seen_ids:
                raise ValueError(f"Duplicate pair_id '{pair_id}' in {path}")
            seen_ids.add(pair_id)
            records.append(record)
    return records


def load_completed_ids(output_path: Path) -> set[str]:
    """Return IDs whose latest saved status is success or review_required."""
    if not output_path.exists():
        return set()
    latest_status_by_id: dict[str, str] = {}
    with output_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in existing output, line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in existing output, line {line_number}"
                )
            pair_id = record.get("pair_id")
            if not isinstance(pair_id, str) or not pair_id:
                continue
            if isinstance(record.get("llm_evaluation"), dict):
                status = record["llm_evaluation"].get("status")
            else:
                status = record.get("status", "unknown")
            latest_status_by_id[pair_id] = (
                status if isinstance(status, str) else "unknown"
            )
    return {
        pair_id for pair_id, status in latest_status_by_id.items()
        if status in COMPLETED_STATUSES
    }


def build_judge_request(
    pair: dict[str, Any], match: dict[str, Any]
) -> JudgeRequest:
    """Combine one evaluation pair and its deterministic match record."""
    if pair["pair_id"] != match["pair_id"]:
        raise ValueError("Pair ID mismatch while constructing judge request.")
    required_pair_fields = (
        "question", "answer_variant", "passage_title", "passage_text"
    )
    for field in required_pair_fields:
        if not isinstance(pair.get(field), str):
            raise ValueError(
                f"Invalid or missing '{field}' for pair {pair['pair_id']}"
            )
    occurrences = match.get("occurrences", [])
    if not isinstance(occurrences, list):
        raise ValueError(f"Invalid occurrences for pair {pair['pair_id']}")
    matching_method = match.get("matching_method")
    if not isinstance(matching_method, str):
        raise ValueError(f"Invalid matching_method for pair {pair['pair_id']}")
    return JudgeRequest(
        question=pair["question"],
        answer_variant=pair["answer_variant"],
        passage_title=pair["passage_title"],
        passage_text=pair["passage_text"],
        matching_method=matching_method,
        occurrences=occurrences,
    )


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path_for(output_path: Path) -> Path:
    """Return the sidecar manifest path for a JSONL result file."""
    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def build_run_manifest(
    *,
    config_path: Path,
    pairs_path: Path,
    matches_path: Path,
    output_path: Path,
    judge_settings: dict[str, Any],
) -> dict[str, Any]:
    """Build the immutable identity of one evaluation experiment."""
    # Never store API keys or environment variable names in the manifest.
    settings = {
        "provider": judge_settings["provider"],
        "model": judge_settings["model"],
        "base_url": judge_settings["base_url"],
        "temperature": judge_settings["temperature"],
        "timeout": judge_settings["timeout"],
        "validation_max_retries": judge_settings["max_retries"],
        "api_retry": judge_settings["api_retry"],
        "request_pacing": judge_settings["request_pacing"],
        "response_format": judge_settings["response_format"],
        "prompt_version": judge_settings["prompt_version"],
    }
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "inputs": {
            "pairs": {
                "path": str(pairs_path.resolve()),
                "sha256": sha256_file(pairs_path),
            },
            "matches": {
                "path": str(matches_path.resolve()),
                "sha256": sha256_file(matches_path),
            },
        },
        "output_path": str(output_path.resolve()),
        "experiment": settings,
    }


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write manifest atomically so an interrupted write cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def validate_resume_manifest(
    manifest_path: Path, expected_manifest: dict[str, Any]
) -> None:
    """Reject resume unless experiment settings and input identities match."""
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume safely: run manifest is missing: {manifest_path}. "
            "Start a new run with a new --output path."
        )
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read run manifest {manifest_path}: {exc}") from exc
    if not isinstance(existing, dict):
        raise ValueError(f"Invalid run manifest: {manifest_path}")
    if existing.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("Incompatible run manifest version.")
    # created_at_utc is intentionally ignored; --limit is not stored.
    keys = ("config_path", "inputs", "output_path", "experiment")
    mismatches = [
        key for key in keys if existing.get(key) != expected_manifest.get(key)
    ]
    if mismatches:
        raise ValueError(
            "Resume rejected: experiment manifest mismatch in "
            + ", ".join(mismatches)
            + ". Use the original inputs/configuration or choose a new --output."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Filter v2 contextual evaluation."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--pairs",
        default="data/processed/filter_v2/dataset/filter_v2_pairs.jsonl",
    )
    parser.add_argument(
        "--matches",
        default=(
            "data/processed/filter_v2/matching/results/"
            "deterministic_match_results_body_only.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "data/processed/filter_v2/evaluation/dev/groq/"
            "llm_evaluation_results_body_only.jsonl"
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=5,
        help="Maximum number of new pairs to evaluate. Default: 5.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume only when the sidecar manifest matches this experiment.",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1.")
    return args


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = config_path.parent

    def project_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    pairs_path = project_path(args.pairs).resolve()
    matches_path = project_path(args.matches).resolve()
    output_path = project_path(args.output).resolve()
    manifest_path = manifest_path_for(output_path)

    if output_path.exists() and not args.resume:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use --resume with its matching manifest, or provide a new --output."
        )
    if args.resume and not output_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume: output file does not exist: {output_path}"
        )

    print("Loading input datasets...")
    pairs = read_jsonl(pairs_path)
    matches = read_jsonl(matches_path)
    print(f"Evaluation pairs: {len(pairs)}")
    print(f"Deterministic match records: {len(matches)}")

    match_by_id = {record["pair_id"]: record for record in matches}
    missing_matches = [
        pair["pair_id"] for pair in pairs
        if pair["pair_id"] not in match_by_id
    ]
    if missing_matches:
        raise ValueError(
            f"{len(missing_matches)} evaluation pairs have no "
            f"deterministic match record. First missing ID: {missing_matches[0]}"
        )
    print("All evaluation pairs have matching records.")

    config = load_config(config_path)
    judge_settings = get_judge_config(config)
    expected_manifest = build_run_manifest(
        config_path=config_path,
        pairs_path=pairs_path,
        matches_path=matches_path,
        output_path=output_path,
        judge_settings=judge_settings,
    )

    if args.resume:
        validate_resume_manifest(manifest_path, expected_manifest)
        print(f"Validated run manifest: {manifest_path}")
    else:
        write_run_manifest(manifest_path, expected_manifest)
        print(f"Created run manifest: {manifest_path}")

    print(f"Provider: {judge_settings['provider']}")
    print(f"Model: {judge_settings['model']}")
    print(f"Maximum validation retries: {judge_settings['max_retries']}")
    print(f"Maximum API retries: {judge_settings['api_retry']['max_retries']}")
    print(f"Request pacing enabled: {judge_settings['request_pacing']['enabled']}")
    print(
        "Request pacing interval: "
        f"{judge_settings['request_pacing']['min_interval_seconds']} seconds"
    )

    judge = create_contextual_judge(config_path=config_path)
    evaluator = ContextualEvaluator(
        judge=judge,
        max_retries=judge_settings["max_retries"],
        api_retry=judge_settings["api_retry"],
        request_pacing=judge_settings["request_pacing"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set[str] = set()
    if args.resume:
        completed_ids = load_completed_ids(output_path)
        print(f"Previously completed pairs: {len(completed_ids)}")

    pending_pairs = [
        pair for pair in pairs if pair["pair_id"] not in completed_ids
    ]
    selected_pairs = pending_pairs[:args.limit]
    if not selected_pairs:
        print("No new pairs to evaluate.")
        return

    print(f"Pairs selected for this run: {len(selected_pairs)}")
    print(f"Output file: {output_path}")
    print("\nStarting contextual evaluation...\n")

    output_mode = "a" if args.resume else "w"
    counts = {
        "success": 0, "review_required": 0,
        "api_error": 0, "runner_error": 0,
    }
    with output_path.open(
        output_mode, encoding="utf-8", buffering=1
    ) as output_file:
        for index, pair in enumerate(selected_pairs, start=1):
            pair_id = pair["pair_id"]
            match = match_by_id[pair_id]
            print(f"[{index}/{len(selected_pairs)}] Evaluating {pair_id}...")
            try:
                request = build_judge_request(pair, match)
                evaluation = evaluator.evaluate_pair(request)
                result = {
                    "pair_id": pair_id,
                    "beir_query_id": pair.get("beir_query_id"),
                    "question": pair.get("question"),
                    "answer_variant_index": pair.get("answer_variant_index"),
                    "answer_variant": pair.get("answer_variant"),
                    "passage_id": pair.get("passage_id"),
                    "passage_title": pair.get("passage_title"),
                    "deterministic_match": {
                        "match_found": match.get("match_found"),
                        "matching_method": match.get("matching_method"),
                        "title_match_count": match.get("title_match_count", 0),
                        "body_match_count": match.get("body_match_count", 0),
                        "occurrences": match.get("occurrences", []),
                    },
                    "experiment_config": {
                        "provider": judge_settings["provider"],
                        "model": judge_settings["model"],
                        "temperature": judge_settings["temperature"],
                        "prompt_version": judge_settings["prompt_version"],
                    },
                    "llm_evaluation": evaluation,
                }
                status = evaluation["status"]
            except Exception as exc:
                result = {
                    "pair_id": pair_id,
                    "beir_query_id": pair.get("beir_query_id"),
                    "question": pair.get("question"),
                    "answer_variant": pair.get("answer_variant"),
                    "passage_id": pair.get("passage_id"),
                    "experiment_config": {
                        "provider": judge_settings["provider"],
                        "model": judge_settings["model"],
                        "temperature": judge_settings["temperature"],
                        "prompt_version": judge_settings["prompt_version"],
                    },
                    "status": "runner_error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                status = "runner_error"
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            counts[status] += 1
            print(f"  Result: {status}\n")

    print("Evaluation run completed.")
    print(f"Results saved to: {output_path}")
    print(f"Run manifest: {manifest_path}")
    print(f"Successful evaluations: {counts['success']}")
    print(f"Require review: {counts['review_required']}")
    print(f"API errors: {counts['api_error']}")
    print(f"Runner errors: {counts['runner_error']}")


if __name__ == "__main__":
    main()
