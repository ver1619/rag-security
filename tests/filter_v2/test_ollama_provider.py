import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_security_research.filter_v2.judge.base import JudgeRequest
from src.rag_security_research.filter_v2.judge.provider import (
    OllamaContextualJudge,
    create_contextual_judge,
)


class TestOllamaContextualJudge(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_file = Path(self.temp_dir.name) / "config.yaml"

        self.config_file.write_text(
            """
filter_v2:
  judge:
    provider: ollama
    model: qwen2.5:7b
    api_key_env: null
    base_url: http://localhost:11434/v1
    temperature: 0
    timeout: 120
    max_retries: 1
    response_format: json
    prompt_version: filter_v2_judge_v2
""",
            encoding="utf-8",
        )

        self.request = JudgeRequest(
            question="Who developed the theory of relativity?",
            answer_variant="Albert Einstein",
            passage_title="Albert Einstein",
            passage_text=(
                "Albert Einstein developed the theory of relativity."
            ),
            matching_method="body_only",
            occurrences=[],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch(
        "src.rag_security_research.filter_v2.judge.provider.OpenAI"
    )
    def test_provider_initialization(self, mock_openai):
        judge = OllamaContextualJudge(self.config_file)

        self.assertEqual(judge.provider, "ollama")
        self.assertEqual(judge.model, "qwen2.5:7b")

        arguments = mock_openai.call_args.kwargs

        self.assertEqual(
            arguments["base_url"],
            "http://localhost:11434/v1",
        )
        self.assertEqual(arguments["api_key"], "ollama")
        self.assertEqual(arguments["max_retries"], 0)

    @patch(
        "src.rag_security_research.filter_v2.judge.provider.OpenAI"
    )
    def test_evaluate_returns_judge_response(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_client.chat.completions.create.return_value = (
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"classification":"direct_support",'
                                '"evidence_quote":"Albert Einstein developed '
                                'the theory of relativity.",'
                                '"rationale":"The passage directly supports '
                                'the answer."}'
                            )
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                ),
            )
        )

        judge = OllamaContextualJudge(self.config_file)
        result = judge.evaluate(self.request)

        self.assertEqual(result.provider, "ollama")
        self.assertEqual(result.model, "qwen2.5:7b")
        self.assertEqual(result.usage["total_tokens"], 150)

    @patch(
        "src.rag_security_research.filter_v2.judge.provider.OpenAI"
    )
    def test_factory_creates_ollama_provider(self, mock_openai):
        judge = create_contextual_judge(self.config_file)

        self.assertIsInstance(judge, OllamaContextualJudge)
        mock_openai.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)