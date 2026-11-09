import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_security_research.filter_v2.judge.base import JudgeRequest
from src.rag_security_research.filter_v2.judge.provider import (
    GroqContextualJudge,
    create_contextual_judge,
)


class TestGroqContextualJudge(unittest.TestCase):

    def setUp(self):
        self.original_api_key = os.environ.get("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = "test-groq-api-key"

        self.config_file = PROJECT_ROOT / "config/filter_v2/config.yaml"

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

    def tearDown(self):
        if self.original_api_key is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = self.original_api_key

    @patch(
        "src.rag_security_research.filter_v2.judge.provider.OpenAI"
    )
    def test_provider_initialization(self, mock_openai):
        judge = GroqContextualJudge(self.config_file)

        self.assertEqual(judge.provider, "groq")
        self.assertEqual(judge.settings["provider"], "groq")
        self.assertEqual(judge.model, judge.settings["model"])

        mock_openai.assert_called_once()

        client_arguments = mock_openai.call_args.kwargs
        self.assertEqual(client_arguments["max_retries"], 0)

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
                                '"evidence_quote":"Albert Einstein was a '
                                'theoretical physicist who developed the '
                                'theory of relativity.",'
                                '"rationale":"The passage directly '
                                'identifies Einstein as the developer."}'
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

        judge = GroqContextualJudge(self.config_file)
        result = judge.evaluate(self.request)

        self.assertEqual(result.provider, "groq")
        self.assertEqual(result.model, judge.model)
        self.assertIn("direct_support", result.raw_response)
        self.assertEqual(result.usage["total_tokens"], 150)

        api_call = mock_client.chat.completions.create.call_args
        self.assertEqual(
            api_call.kwargs["response_format"],
            {"type": "json_object"},
        )

    @patch(
        "src.rag_security_research.filter_v2.judge.provider.OpenAI"
    )
    def test_validation_feedback_is_sent_to_groq(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_client.chat.completions.create.return_value = (
            SimpleNamespace(
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
        )

        feedback = (
            "Evidence quote is not an exact substring "
            "of the original passage."
        )

        request = JudgeRequest(
            question=self.request.question,
            answer_variant=self.request.answer_variant,
            passage_title=self.request.passage_title,
            passage_text=self.request.passage_text,
            matching_method=self.request.matching_method,
            occurrences=self.request.occurrences,
            validation_feedback=feedback,
        )

        judge = GroqContextualJudge(self.config_file)
        judge.evaluate(request)

        messages = (
            mock_client.chat.completions.create.call_args
            .kwargs["messages"]
        )

        user_message = next(
            message["content"]
            for message in messages
            if message["role"] == "user"
        )

        self.assertIn(feedback, user_message)

    @patch(
        "src.rag_security_research.filter_v2.judge.provider.OpenAI"
    )
    def test_empty_response_raises_error(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_client.chat.completions.create.return_value = (
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None)
                    )
                ],
                usage=None,
            )
        )

        judge = GroqContextualJudge(self.config_file)

        with self.assertRaises(RuntimeError):
            judge.evaluate(self.request)

    @patch(
        "src.rag_security_research.filter_v2.judge.provider.OpenAI"
    )
    def test_factory_creates_groq_provider(self, mock_openai):
        judge = create_contextual_judge(self.config_file)

        self.assertIsInstance(judge, GroqContextualJudge)
        mock_openai.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)