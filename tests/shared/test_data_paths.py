import json
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_security_research.filter_v2.config_loader import (
    load_config,
    resolve_project_path,
)


class TestDataDirectoryPaths(unittest.TestCase):
    """Verify that all data paths match the new authoritative directory structure and exist on disk."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = PROJECT_ROOT
        cls.config_path = cls.project_root / "config/filter_v2/config.yaml"
        cls.config_gemini_path = cls.project_root / "config/filter_v2/config.gemini.yaml"
        cls.config = load_config(cls.config_path)

    def test_raw_dataset_paths_exist(self):
        raw_paths = [
            self.project_root / "data/raw/beir_nq/corpus.jsonl",
            self.project_root / "data/raw/beir_nq/queries.jsonl",
            self.project_root / "data/raw/beir_nq/qrels/test.tsv",
            self.project_root / "data/raw/nq_open/validation.jsonl",
        ]
        for path in raw_paths:
            with self.subTest(path=str(path.relative_to(self.project_root))):
                self.assertTrue(path.is_file(), f"Missing raw file: {path}")

    def test_shared_dataset_paths_exist(self):
        shared_paths = [
            self.project_root / "data/processed/shared/matched_questions.json",
            self.project_root / "data/processed/shared/matching_statistics.json",
        ]
        for path in shared_paths:
            with self.subTest(path=str(path.relative_to(self.project_root))):
                self.assertTrue(path.is_file(), f"Missing shared file: {path}")

    def test_filter_v1_paths_exist(self):
        v1_paths = [
            self.project_root / "data/processed/filter_v1/results/answer_supported_questions.json",
            self.project_root / "data/processed/filter_v1/results/answer_support_statistics.json",
        ]
        for path in v1_paths:
            with self.subTest(path=str(path.relative_to(self.project_root))):
                self.assertTrue(path.is_file(), f"Missing filter_v1 file: {path}")

    def test_filter_v2_dataset_paths_exist(self):
        dataset_paths = [
            self.project_root / "data/processed/filter_v2/dataset/filter_v2_questions.jsonl",
            self.project_root / "data/processed/filter_v2/dataset/filter_v2_pairs.jsonl",
            self.project_root / "data/processed/filter_v2/dataset/dataset_statistics.json",
        ]
        for path in dataset_paths:
            with self.subTest(path=str(path.relative_to(self.project_root))):
                self.assertTrue(path.is_file(), f"Missing filter_v2 dataset file: {path}")

    def test_filter_v2_matching_paths_exist(self):
        matching_paths = [
            self.project_root / "data/processed/filter_v2/matching/results/deterministic_match_results.jsonl",
            self.project_root / "data/processed/filter_v2/matching/results/deterministic_match_results_body_only.jsonl",
            self.project_root / "data/processed/filter_v2/matching/statistics/deterministic_match_statistics.json",
            self.project_root / "data/processed/filter_v2/matching/statistics/deterministic_match_statistics_body_only.json",
        ]
        for path in matching_paths:
            with self.subTest(path=str(path.relative_to(self.project_root))):
                self.assertTrue(path.is_file(), f"Missing matching file: {path}")

    def test_golden_dataset_paths_exist(self):
        golden_paths = [
            self.project_root / "data/processed/filter_v2/golden_dataset/dev_pairs.jsonl",
            self.project_root / "data/processed/filter_v2/golden_dataset/test_pairs.jsonl",
            self.project_root / "data/processed/filter_v2/golden_dataset/sampling_metadata.json",
            self.project_root / "data/processed/filter_v2/golden_dataset/annotations/dev_annotations.jsonl",
            self.project_root / "data/processed/filter_v2/golden_dataset/annotations/dev_annotation_statistics.json",
            self.project_root / "data/processed/filter_v2/golden_dataset/annotations/test_annotations.jsonl",
        ]
        for path in golden_paths:
            with self.subTest(path=str(path.relative_to(self.project_root))):
                self.assertTrue(path.is_file(), f"Missing golden dataset file: {path}")

    def test_shared_benchmark_paths_exist(self):
        benchmark_paths = [
            self.project_root / "data/processed/shared/benchmark/dev_questions.jsonl",
            self.project_root / "data/processed/shared/benchmark/test_questions.jsonl",
            self.project_root / "data/processed/shared/benchmark/split_metadata.json",
        ]
        for path in benchmark_paths:
            with self.subTest(path=str(path.relative_to(self.project_root))):
                self.assertTrue(path.is_file(), f"Missing shared benchmark file: {path}")

    def test_evaluation_manifests_remain_paired_with_results(self):
        eval_dir = self.project_root / "data/processed/filter_v2/evaluation"
        manifest_files = list(eval_dir.rglob("*.manifest.json"))
        self.assertGreater(len(manifest_files), 0, "No evaluation manifests found.")

        for manifest_path in manifest_files:
            # Expected pairing: <name>.jsonl.manifest.json -> <name>.jsonl
            expected_jsonl_name = manifest_path.name.replace(".manifest.json", "")
            expected_jsonl_path = manifest_path.with_name(expected_jsonl_name)
            with self.subTest(manifest=str(manifest_path.relative_to(self.project_root))):
                self.assertTrue(
                    expected_jsonl_path.is_file(),
                    f"Manifest {manifest_path.name} is missing its paired result file: {expected_jsonl_path}",
                )

    def test_diagnostics_and_audits_exist(self):
        diag_and_audit_paths = [
            self.project_root / "data/processed/filter_v2/diagnostics/diagnostic_results.jsonl",
            self.project_root / "data/processed/filter_v2/diagnostics/diagnostic_retry_results.jsonl",
            self.project_root / "data/processed/filter_v2/diagnostics/wake_island_prompt_v2_results.jsonl",
            self.project_root / "data/processed/filter_v2/manual_audit/audit_samples.json",
            self.project_root / "data/processed/filter_v2/manual_audit/audit_statistics.json",
            self.project_root / "data/processed/exploration/manual_audit/audit_samples.json",
            self.project_root / "data/processed/exploration/manual_audit/audit_statistics.json",
        ]
        for path in diag_and_audit_paths:
            with self.subTest(path=str(path.relative_to(self.project_root))):
                self.assertTrue(path.is_file(), f"Missing diagnostic or audit file: {path}")

    def test_config_yaml_paths_resolve_to_existing_files(self):
        for cfg_file in [self.config_path, self.config_gemini_path]:
            cfg = load_config(cfg_file)
            test_keys = [
                ("datasets.beir_nq.corpus", True),
                ("datasets.beir_nq.queries", True),
                ("datasets.beir_nq.qrels", True),
                ("datasets.nq_open.validation", True),
                ("filter_v2.questions", True),
                ("filter_v2.pairs", True),
                ("filter_v2.dataset_statistics", True),
                ("filter_v2.deterministic_results", True),
                ("filter_v2.deterministic_statistics", True),
                ("filter_v2.manual_audit", False),
                ("shared.matched_questions", True),
                ("shared.matching_statistics", True),
                ("shared.benchmark.dev_questions", True),
                ("shared.benchmark.test_questions", True),
                ("shared.benchmark.split_metadata", True),
                ("filter_v1.answer_supported_questions", True),
                ("filter_v1.answer_support_statistics", True),
                ("baseline.dataset", False),
                ("baseline.knowledge_base", False),
                ("baseline.indexes", False),
                ("baseline.results.dev", False),
                ("baseline.results.test", False),
                ("baseline.results.summaries", False),
            ]
            for key, expect_file in test_keys:
                resolved = resolve_project_path(cfg, key, self.project_root / "config.yaml")
                with self.subTest(config=cfg_file.name, key=key):
                    if expect_file:
                        self.assertTrue(resolved.is_file(), f"{key} in {cfg_file.name} did not resolve to a file: {resolved}")
                    else:
                        self.assertTrue(resolved.is_dir(), f"{key} in {cfg_file.name} did not resolve to a directory: {resolved}")

    def test_working_directory_independent_resolution(self):
        # Resolve config from a different directory (e.g. tests/)
        tests_dir = self.project_root / "tests"
        old_cwd = Path.cwd()
        try:
            os.chdir(tests_dir)
            resolved = resolve_project_path(
                self.config,
                "filter_v2.questions",
                self.project_root / "config.yaml",
            )
            self.assertEqual(
                resolved,
                self.project_root / "data/processed/filter_v2/dataset/filter_v2_questions.jsonl",
            )
            self.assertTrue(resolved.is_file())
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main(verbosity=2)

