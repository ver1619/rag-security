import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_security_research.filter_v2.evaluation_runner import (
    build_judge_request,
    build_run_manifest,
    load_completed_ids,
    manifest_path_for,
    read_jsonl,
    sha256_file,
    validate_resume_manifest,
    write_run_manifest,
)


class TestEvaluationRunner(unittest.TestCase):
    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def sample_settings(self):
        return {
            "provider": "groq",
            "model": "openai/gpt-oss-20b",
            "base_url": "https://api.groq.com/openai/v1",
            "temperature": 0.0,
            "timeout": 60.0,
            "max_retries": 1,
            "api_retry": {
                "max_retries": 3,
                "initial_delay_seconds": 5.0,
                "max_delay_seconds": 60.0,
                "backoff_multiplier": 2.0,
            },
            "request_pacing": {
                "enabled": True,
                "min_interval_seconds": 2.0,
            },
            "response_format": "json",
            "prompt_version": "filter_v2_judge_v2",
            "api_key": "SECRET_SHOULD_NOT_BE_STORED",
            "api_key_env": "GROQ_API_KEY",
        }

    def make_manifest(self, root: Path):
        pairs = root / "pairs.jsonl"
        matches = root / "matches.jsonl"
        config = root / "config.yaml"
        output = root / "results.jsonl"
        pairs.write_text('{"pair_id":"p1"}\n', encoding="utf-8")
        matches.write_text('{"pair_id":"p1"}\n', encoding="utf-8")
        config.write_text("filter_v2: {}\n", encoding="utf-8")
        manifest = build_run_manifest(
            config_path=config,
            pairs_path=pairs,
            matches_path=matches,
            output_path=output,
            judge_settings=self.sample_settings(),
        )
        return manifest, pairs, matches, config, output

    def test_read_jsonl_loads_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "records.jsonl"
            self.write_jsonl(path, [{"pair_id": "p1"}, {"pair_id": "p2"}])
            self.assertEqual(len(read_jsonl(path)), 2)

    def test_read_jsonl_rejects_duplicate_pair_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "records.jsonl"
            self.write_jsonl(path, [{"pair_id": "p1"}, {"pair_id": "p1"}])
            with self.assertRaisesRegex(ValueError, "Duplicate pair_id"):
                read_jsonl(path)

    def test_read_jsonl_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "records.jsonl"
            path.write_text("{invalid}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid JSON"):
                read_jsonl(path)

    def test_read_jsonl_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                read_jsonl(Path(temp) / "missing.jsonl")

    def test_load_completed_ids_uses_latest_status(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.jsonl"
            self.write_jsonl(path, [
                {"pair_id": "p1", "llm_evaluation": {"status": "success"}},
                {"pair_id": "p2", "llm_evaluation": {"status": "success"}},
                {"pair_id": "p2", "llm_evaluation": {"status": "api_error"}},
            ])
            self.assertEqual(load_completed_ids(path), {"p1"})

    def test_load_completed_ids_handles_missing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(load_completed_ids(Path(temp) / "missing"), set())

    def test_build_judge_request(self):
        pair = {
            "pair_id": "p1", "question": "Q", "answer_variant": "A",
            "passage_title": "T", "passage_text": "Body",
        }
        match = {
            "pair_id": "p1", "matching_method": "body_only",
            "occurrences": [],
        }
        request = build_judge_request(pair, match)
        self.assertEqual(request.passage_text, "Body")

    def test_build_judge_request_rejects_pair_id_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Pair ID mismatch"):
            build_judge_request(
                {"pair_id": "p1"}, {"pair_id": "p2"}
            )

    def test_build_judge_request_rejects_missing_passage_text(self):
        pair = {
            "pair_id": "p1", "question": "Q", "answer_variant": "A",
            "passage_title": "T",
        }
        match = {"pair_id": "p1", "matching_method": "body_only"}
        with self.assertRaisesRegex(ValueError, "passage_text"):
            build_judge_request(pair, match)

    def test_manifest_created_and_excludes_secrets(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, *_ = self.make_manifest(Path(temp))
            serialized = json.dumps(manifest)
            self.assertNotIn("SECRET", serialized)
            self.assertNotIn("api_key_env", serialized)
            self.assertIn("sha256", manifest["inputs"]["pairs"])

    def test_manifest_path_uses_jsonl_sidecar(self):
        self.assertEqual(
            manifest_path_for(Path("results.jsonl")).name,
            "results.jsonl.manifest.json",
        )

    def test_resume_manifest_accepts_matching_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, *_ = self.make_manifest(root)
            sidecar = root / "manifest.json"
            write_run_manifest(sidecar, manifest)
            expected = dict(manifest)
            expected["created_at_utc"] = "different timestamp"
            validate_resume_manifest(sidecar, expected)

    def test_resume_rejects_missing_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "manifest is missing"):
                validate_resume_manifest(
                    Path(temp) / "missing.json", {}
                )

    def test_resume_rejects_changed_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, *_ = self.make_manifest(root)
            sidecar = root / "manifest.json"
            write_run_manifest(sidecar, manifest)
            expected = json.loads(json.dumps(manifest))
            expected["experiment"]["model"] = "different-model"
            with self.assertRaisesRegex(ValueError, "experiment"):
                validate_resume_manifest(sidecar, expected)

    def test_resume_rejects_changed_input_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, pairs, *_ = self.make_manifest(root)
            sidecar = root / "manifest.json"
            write_run_manifest(sidecar, manifest)
            pairs.write_text('{"pair_id":"p1"}\n{"pair_id":"p2"}\n', encoding="utf-8")
            expected = json.loads(json.dumps(manifest))
            expected["inputs"]["pairs"]["sha256"] = sha256_file(pairs)
            with self.assertRaisesRegex(ValueError, "inputs"):
                validate_resume_manifest(sidecar, expected)

    def test_manifest_write_and_read_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, *_ = self.make_manifest(root)
            sidecar = root / "run.manifest.json"
            write_run_manifest(sidecar, manifest)
            self.assertEqual(json.loads(sidecar.read_text()), manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
