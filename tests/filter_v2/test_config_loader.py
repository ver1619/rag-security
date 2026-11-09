import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_security_research.filter_v2.config_loader import (
    ConfigError,
    get_judge_config,
    load_config,
    resolve_project_path,
)


class TestConfigLoader(unittest.TestCase):

    def setUp(self):
        self.original_groq_key = os.environ.get("GROQ_API_KEY")
        self.original_ollama_key = os.environ.get("OLLAMA_API_KEY")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)

        self.config_file = self.project_root / "config.yaml"
        self.env_file = self.project_root / ".env"

        self.write_config()

        self.env_file.write_text(
            "GROQ_API_KEY=test-key-for-unit-test\n",
            encoding="utf-8",
        )

        os.environ.pop("GROQ_API_KEY", None)
        os.environ.pop("OLLAMA_API_KEY", None)

    def write_config(self, provider="groq", extra=""):
        base_url = (
            "https://api.groq.com/openai/v1"
            if provider == "groq"
            else "http://localhost:11434/v1"
        )

        api_key_env = (
            "GROQ_API_KEY" if provider == "groq" else "null"
        )

        self.config_file.write_text(
            f"""
paths:
  results: results

filter_v2:
  judge:
    enabled: true
    provider: {provider}
    model: test-model
    api_key_env: {api_key_env}
    temperature: 0
    timeout: 60
    max_retries: 1
    base_url: {base_url}
    prompt_version: filter_v2_judge_v2
    api_retry:
      max_retries: 3
      initial_delay_seconds: 5
      max_delay_seconds: 60
      backoff_multiplier: 2
    request_pacing:
      enabled: true
      min_interval_seconds: 2
{extra}
""",
            encoding="utf-8",
        )

    def tearDown(self):
        if self.original_groq_key is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = self.original_groq_key

        if self.original_ollama_key is None:
            os.environ.pop("OLLAMA_API_KEY", None)
        else:
            os.environ["OLLAMA_API_KEY"] = self.original_ollama_key

        self.temp_dir.cleanup()

    def test_load_config(self):
        config = load_config(self.config_file)

        self.assertIsInstance(config, dict)
        self.assertEqual(
            config["filter_v2"]["judge"]["provider"],
            "groq",
        )

    def test_get_groq_judge_config(self):
        config = load_config(self.config_file)
        judge = get_judge_config(config)

        self.assertEqual(judge["provider"], "groq")
        self.assertEqual(judge["model"], "test-model")
        self.assertEqual(judge["temperature"], 0)
        self.assertEqual(judge["timeout"], 60)
        self.assertEqual(judge["max_retries"], 1)
        self.assertEqual(
            judge["base_url"],
            "https://api.groq.com/openai/v1",
        )
        self.assertEqual(
            judge["api_key"],
            "test-key-for-unit-test",
        )

    def test_get_ollama_judge_config_without_api_key(self):
        self.write_config(provider="ollama")

        config = load_config(self.config_file)
        judge = get_judge_config(config)

        self.assertEqual(judge["provider"], "ollama")
        self.assertEqual(judge["model"], "test-model")
        self.assertEqual(
            judge["base_url"],
            "http://localhost:11434/v1",
        )
        self.assertEqual(judge["api_key"], "ollama")

    def test_api_retry_settings_are_loaded(self):
        config = load_config(self.config_file)
        judge = get_judge_config(config)

        self.assertEqual(
            judge["api_retry"],
            {
                "max_retries": 3,
                "initial_delay_seconds": 5.0,
                "max_delay_seconds": 60.0,
                "backoff_multiplier": 2.0,
            },
        )

    def test_request_pacing_settings_are_loaded(self):
        config = load_config(self.config_file)
        judge = get_judge_config(config)

        self.assertEqual(
            judge["request_pacing"],
            {
                "enabled": True,
                "min_interval_seconds": 2.0,
            },
        )

    def test_resolve_project_path(self):
        config = load_config(self.config_file)

        result = resolve_project_path(
            config,
            "results",
            self.config_file,
        )

        self.assertEqual(
            result,
            self.project_root / "results",
        )

    def test_resolve_project_path_nested(self):
        self.config_file.write_text(
            """
paths:
  filter_v2:
    questions: data/processed/filter_v2/dataset/filter_v2_questions.jsonl
filter_v2:
  judge:
    enabled: true
    provider: groq
    model: test-model
    temperature: 0
    timeout: 60
    max_retries: 1
    prompt_version: filter_v2_judge_v2
""",
            encoding="utf-8",
        )
        config = load_config(self.config_file)
        result = resolve_project_path(
            config,
            "filter_v2.questions",
            self.config_file,
        )
        self.assertEqual(
            result,
            self.project_root / "data/processed/filter_v2/dataset/filter_v2_questions.jsonl",
        )

    def test_missing_groq_api_key(self):
        config = load_config(self.config_file)

        os.environ.pop("GROQ_API_KEY", None)
        self.env_file.unlink()

        with self.assertRaises(ConfigError):
            get_judge_config(config)

    def test_unsupported_provider_rejected(self):
        config = load_config(self.config_file)
        config["filter_v2"]["judge"]["provider"] = "unsupported"

        with self.assertRaises(ConfigError):
            get_judge_config(config)

    def test_placeholder_model_rejected(self):
        config = load_config(self.config_file)
        config["filter_v2"]["judge"]["model"] = "YOUR_MODEL_ID"

        with self.assertRaises(ConfigError):
            get_judge_config(config)

    def test_negative_api_retries_rejected(self):
        config = load_config(self.config_file)
        config["filter_v2"]["judge"]["api_retry"]["max_retries"] = -1

        with self.assertRaises(ConfigError):
            get_judge_config(config)

    def test_invalid_backoff_multiplier_rejected(self):
        config = load_config(self.config_file)
        config["filter_v2"]["judge"]["api_retry"][
            "backoff_multiplier"
        ] = 0.5

        with self.assertRaises(ConfigError):
            get_judge_config(config)

    def test_max_delay_less_than_initial_delay_rejected(self):
        config = load_config(self.config_file)
        config["filter_v2"]["judge"]["api_retry"][
            "max_delay_seconds"
        ] = 2

        with self.assertRaises(ConfigError):
            get_judge_config(config)

    def test_invalid_pacing_enabled_rejected(self):
        config = load_config(self.config_file)
        config["filter_v2"]["judge"]["request_pacing"][
            "enabled"
        ] = "yes"

        with self.assertRaises(ConfigError):
            get_judge_config(config)

    def test_missing_config_file(self):
        missing_file = self.project_root / "missing.yaml"

        with self.assertRaises(ConfigError):
            load_config(missing_file)

    def test_invalid_yaml_structure(self):
        invalid_config = self.project_root / "invalid.yaml"
        invalid_config.write_text(
            "- item_one\n- item_two\n",
            encoding="utf-8",
        )

        with self.assertRaises(ConfigError):
            load_config(invalid_config)


if __name__ == "__main__":
    unittest.main(verbosity=2)