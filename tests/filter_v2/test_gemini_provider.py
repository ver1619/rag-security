from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

from src.rag_security_research.filter_v2.judge.base import JudgeRequest
from src.rag_security_research.filter_v2.judge.provider import (
    GeminiContextualJudge,
    create_contextual_judge,
)


class TestGeminiContextualJudge(unittest.TestCase):
    def setUp(self) -> None:
        self.original_api_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "test-gemini-api-key"

        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_file = Path(self.temp_dir.name) / "config.yaml"
        self.config_file.write_text(
            yaml.safe_dump(
                {
                    "filter_v2": {
                        "judge": {
                            "provider": "gemini",
                            "model": "gemini-3.1-flash-lite",
                            "api_key_env": "GEMINI_API_KEY",
                            "base_url": (
                                "https://generativelanguage.googleapis.com/"
                                "v1beta/openai/"
                            ),
                            "temperature": 0,
                            "timeout": 60,
                            "response_format": "json",
                            "prompt_version": "filter_v2_judge_v2",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        self.request = JudgeRequest(
            question="Who developed the theory of relativity?",
            answer_variant="Albert Einstein",
            passage_title="Albert Einstein",
            passage_text=(
                "Albert Einstein was a theoretical physicist "
                "who developed the theory of relativity."
            ),
            matching_method="body_only",
            occurrences=[],
        )

    def tearDown(self) -> None:
        if self.original_api_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = self.original_api_key
        self.temp_dir.cleanup()

    @patch("src.rag_security_research.filter_v2.judge.provider.OpenAI")
    def test_initializes_with_gemini_settings(self, mock_openai):
        judge = GeminiContextualJudge(self.config_file)

        self.assertEqual(judge.provider, "gemini")
        self.assertEqual(judge.model, "gemini-3.1-flash-lite")
        mock_openai.assert_called_once()

        arguments = mock_openai.call_args.kwargs
        self.assertEqual(arguments["api_key"], "test-gemini-api-key")
        self.assertEqual(
            arguments["base_url"],
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.assertEqual(arguments["max_retries"], 0)

    @patch("src.rag_security_research.filter_v2.judge.provider.OpenAI")
    def test_evaluate_returns_judge_response_and_usage(self, mock_openai):
        client = MagicMock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"classification":"direct_support",'
                            '"evidence_quote":"Albert Einstein was a '
                            'theoretical physicist who developed the '
                            'theory of relativity.",'
                            '"rationale":"The passage directly supports '
                            'the answer."}'
                        )
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=120,
                completion_tokens=45,
                total_tokens=165,
            ),
        )

        judge = GeminiContextualJudge(self.config_file)
        result = judge.evaluate(self.request)

        self.assertEqual(result.provider, "gemini")
        self.assertEqual(result.model, "gemini-3.1-flash-lite")
        self.assertIn("direct_support", result.raw_response)
        self.assertEqual(result.usage["total_tokens"], 165)

        call = client.chat.completions.create.call_args.kwargs
        self.assertEqual(call["model"], "gemini-3.1-flash-lite")
        self.assertEqual(call["temperature"], 0)
        self.assertEqual(call["response_format"], {"type": "json_object"})

    @patch("src.rag_security_research.filter_v2.judge.provider.OpenAI")
    def test_validation_feedback_is_included(self, mock_openai):
        client = MagicMock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"classification":"direct_support",'
                            '"evidence_quote":"Albert Einstein was a '
                            'theoretical physicist",'
                            '"rationale":"The passage supports the answer."}'
                        )
                    )
                )
            ],
            usage=None,
        )

        request = JudgeRequest(
            question=self.request.question,
            answer_variant=self.request.answer_variant,
            passage_title=self.request.passage_title,
            passage_text=self.request.passage_text,
            matching_method=self.request.matching_method,
            occurrences=self.request.occurrences,
            validation_feedback="Return an exact evidence quote.",
        )

        GeminiContextualJudge(self.config_file).evaluate(request)
        messages = (
            client.chat.completions.create.call_args.kwargs["messages"]
        )
        user_message = next(
            item["content"] for item in messages if item["role"] == "user"
        )
        self.assertIn("Return an exact evidence quote.", user_message)

    @patch("src.rag_security_research.filter_v2.judge.provider.OpenAI")
    def test_empty_response_raises_runtime_error(self, mock_openai):
        client = MagicMock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None)
                )
            ],
            usage=None,
        )

        judge = GeminiContextualJudge(self.config_file)
        with self.assertRaises(RuntimeError):
            judge.evaluate(self.request)

    @patch("src.rag_security_research.filter_v2.judge.provider.OpenAI")
    def test_factory_creates_gemini_provider(self, mock_openai):
        judge = create_contextual_judge(self.config_file)
        self.assertIsInstance(judge, GeminiContextualJudge)
        mock_openai.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
