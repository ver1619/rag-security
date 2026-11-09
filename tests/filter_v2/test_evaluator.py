
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from openai import RateLimitError, APITimeoutError

from src.rag_security_research.filter_v2.evaluator import (
    ContextualEvaluator,
)
from src.rag_security_research.filter_v2.judge.base import (
    JudgeRequest,
    JudgeResponse,
)


class TestContextualEvaluator(unittest.TestCase):

    def setUp(self):
        self.request = JudgeRequest(
            question="Who developed the theory of relativity?",
            answer_variant="Albert Einstein",
            passage_title="Albert Einstein",
            passage_text=(
                "Albert Einstein was a theoretical physicist "
                "who developed the theory of relativity."
            ),
            matching_method="body",
            occurrences=[],
        )

        self.valid_response = JudgeResponse(
            raw_response=(
                '{"classification":"direct_support",'
                '"evidence_quote":"Albert Einstein was a theoretical '
                'physicist who developed the theory of relativity.",'
                '"rationale":"The passage directly supports the answer."}'
            ),
            provider="groq",
            model="test-model",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        )

        self.invalid_response = JudgeResponse(
            raw_response='{"classification":"unsupported_label"}',
            provider="groq",
            model="test-model",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        )

    @staticmethod
    def make_rate_limit_error():
        return RateLimitError(
            message="Rate limit reached",
            response=MagicMock(status_code=429, headers={}),
            body={"error": {"message": "Rate limit reached"}},
        )

    @staticmethod
    def make_timeout_error():
        return APITimeoutError(request=MagicMock())

    def test_valid_response_succeeds(self):
        mock_judge = MagicMock()
        mock_judge.evaluate.return_value = self.valid_response

        evaluator = ContextualEvaluator(
            judge=mock_judge,
            max_retries=1,
        )

        result = evaluator.evaluate_pair(self.request)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["classification"], "direct_support")
        self.assertEqual(result["attempt_count"], 1)
        self.assertEqual(result["usage"]["total_tokens"], 150)

    def test_invalid_response_retries_once(self):
        mock_judge = MagicMock()
        mock_judge.evaluate.side_effect = [
            self.invalid_response,
            self.valid_response,
        ]

        evaluator = ContextualEvaluator(
            judge=mock_judge,
            max_retries=1,
        )

        result = evaluator.evaluate_pair(self.request)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(result["classification"], "direct_support")

    def test_retry_receives_validation_feedback(self):
        mock_judge = MagicMock()
        mock_judge.evaluate.side_effect = [
            self.invalid_response,
            self.valid_response,
        ]

        evaluator = ContextualEvaluator(
            judge=mock_judge,
            max_retries=1,
        )

        result = evaluator.evaluate_pair(self.request)

        self.assertEqual(result["status"], "success")

        first_request = mock_judge.evaluate.call_args_list[0].args[0]
        second_request = mock_judge.evaluate.call_args_list[1].args[0]

        self.assertIsNone(first_request.validation_feedback)
        self.assertIsNotNone(second_request.validation_feedback)

        for error in result["attempts"][0]["validation_errors"]:
            self.assertIn(
                error,
                second_request.validation_feedback,
            )

        self.assertEqual(
            second_request.passage_text,
            self.request.passage_text,
        )
        self.assertEqual(
            second_request.question,
            self.request.question,
        )

    def test_persistent_invalid_response_requires_review(self):
        mock_judge = MagicMock()
        mock_judge.evaluate.return_value = self.invalid_response

        evaluator = ContextualEvaluator(
            judge=mock_judge,
            max_retries=1,
        )

        result = evaluator.evaluate_pair(self.request)

        self.assertEqual(result["status"], "review_required")
        self.assertIsNone(result["classification"])
        self.assertEqual(result["attempt_count"], 2)

    def test_non_retryable_api_error_is_not_retried(self):
        mock_judge = MagicMock()
        mock_judge.evaluate.side_effect = RuntimeError(
            "Simulated programming error"
        )

        evaluator = ContextualEvaluator(
            judge=mock_judge,
            max_retries=1,
            api_retry={"max_retries": 3},
        )

        result = evaluator.evaluate_pair(self.request)

        self.assertEqual(result["status"], "api_error")
        self.assertEqual(result["attempt_count"], 1)
        mock_judge.evaluate.assert_called_once()

    @patch(
        "src.rag_security_research.filter_v2.evaluator.time.sleep"
    )
    def test_rate_limit_is_retried_then_succeeds(self, mock_sleep):
        mock_judge = MagicMock()
        mock_judge.evaluate.side_effect = [
            self.make_rate_limit_error(),
            self.valid_response,
        ]

        evaluator = ContextualEvaluator(
            judge=mock_judge,
            api_retry={
                "max_retries": 2,
                "initial_delay_seconds": 2,
                "max_delay_seconds": 10,
                "backoff_multiplier": 2,
            },
        )

        result = evaluator.evaluate_pair(self.request)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(
            result["attempts"][0]["status"],
            "api_retry",
        )
        self.assertEqual(
            result["attempts"][1]["status"],
            "valid",
        )
        mock_sleep.assert_called_once_with(2.0)

    @patch(
        "src.rag_security_research.filter_v2.evaluator.time.sleep"
    )
    def test_api_retry_exhaustion_returns_api_error(self, mock_sleep):
        mock_judge = MagicMock()
        mock_judge.evaluate.side_effect = self.make_rate_limit_error()

        evaluator = ContextualEvaluator(
            judge=mock_judge,
            api_retry={
                "max_retries": 2,
                "initial_delay_seconds": 1,
                "max_delay_seconds": 10,
                "backoff_multiplier": 2,
            },
        )

        result = evaluator.evaluate_pair(self.request)

        self.assertEqual(result["status"], "api_error")
        self.assertEqual(result["attempt_count"], 3)
        self.assertEqual(mock_judge.evaluate.call_count, 3)
        self.assertEqual(mock_sleep.call_args_list, [
            call(1.0),
            call(2.0),
        ])

    @patch(
        "src.rag_security_research.filter_v2.evaluator.time.sleep"
    )
    def test_exponential_backoff_is_capped(self, mock_sleep):
        mock_judge = MagicMock()
        mock_judge.evaluate.side_effect = self.make_timeout_error()

        evaluator = ContextualEvaluator(
            judge=mock_judge,
            api_retry={
                "max_retries": 4,
                "initial_delay_seconds": 3,
                "max_delay_seconds": 5,
                "backoff_multiplier": 2,
            },
        )

        result = evaluator.evaluate_pair(self.request)

        self.assertEqual(result["status"], "api_error")
        self.assertEqual(
            mock_sleep.call_args_list,
            [
                call(3.0),
                call(5.0),
                call(5.0),
                call(5.0),
            ],
        )

    def test_api_retry_and_validation_retry_are_independent(self):
        mock_judge = MagicMock()
        mock_judge.evaluate.side_effect = [
            self.make_rate_limit_error(),
            self.invalid_response,
            self.valid_response,
        ]

        evaluator = ContextualEvaluator(
            judge=mock_judge,
            max_retries=1,
            api_retry={
                "max_retries": 1,
                "initial_delay_seconds": 0,
                "max_delay_seconds": 5,
                "backoff_multiplier": 2,
            },
        )

        with patch(
            "src.rag_security_research.filter_v2.evaluator.time.sleep"
        ):
            result = evaluator.evaluate_pair(self.request)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["attempt_count"], 3)
        self.assertEqual(mock_judge.evaluate.call_count, 3)

        # The third request should contain validation feedback because
        # the second API response was received but failed validation.
        third_request = mock_judge.evaluate.call_args_list[2].args[0]
        self.assertIsNotNone(third_request.validation_feedback)

    @patch(
        "src.rag_security_research.filter_v2.evaluator.time.sleep"
    )
    def test_request_pacing_waits_between_requests(self, mock_sleep):
        mock_judge = MagicMock()
        mock_judge.evaluate.side_effect = [
            self.invalid_response,
            self.valid_response,
        ]

        evaluator = ContextualEvaluator(
            judge=mock_judge,
            max_retries=1,
            request_pacing={
                "enabled": True,
                "min_interval_seconds": 2,
            },
        )

        with patch(
            "src.rag_security_research.filter_v2.evaluator.time.monotonic",
            side_effect=[10.0, 10.0, 10.0, 12.0],
        ):
            result = evaluator.evaluate_pair(self.request)

        self.assertEqual(result["status"], "success")
        mock_sleep.assert_called_once_with(2.0)

    def test_pacing_disabled_does_not_sleep(self):
        mock_judge = MagicMock()
        mock_judge.evaluate.return_value = self.valid_response

        evaluator = ContextualEvaluator(
            judge=mock_judge,
            request_pacing={
                "enabled": False,
                "min_interval_seconds": 10,
            },
        )

        with patch(
            "src.rag_security_research.filter_v2.evaluator.time.sleep"
        ) as mock_sleep:
            result = evaluator.evaluate_pair(self.request)

        self.assertEqual(result["status"], "success")
        mock_sleep.assert_not_called()

    def test_zero_validation_retries(self):
        mock_judge = MagicMock()
        mock_judge.evaluate.return_value = self.invalid_response

        evaluator = ContextualEvaluator(
            judge=mock_judge,
            max_retries=0,
        )

        result = evaluator.evaluate_pair(self.request)

        self.assertEqual(result["status"], "review_required")
        self.assertEqual(result["attempt_count"], 1)

    def test_negative_validation_retries_rejected(self):
        with self.assertRaises(ValueError):
            ContextualEvaluator(
                judge=MagicMock(),
                max_retries=-1,
            )

    def test_invalid_api_retry_settings_rejected(self):
        with self.assertRaises(ValueError):
            ContextualEvaluator(
                judge=MagicMock(),
                api_retry={"max_retries": -1},
            )

        with self.assertRaises(ValueError):
            ContextualEvaluator(
                judge=MagicMock(),
                api_retry={"backoff_multiplier": 0.5},
            )

    def test_invalid_pacing_settings_rejected(self):
        with self.assertRaises(ValueError):
            ContextualEvaluator(
                judge=MagicMock(),
                request_pacing={
                    "enabled": "yes",
                    "min_interval_seconds": 1,
                },
            )

        with self.assertRaises(ValueError):
            ContextualEvaluator(
                judge=MagicMock(),
                request_pacing={
                    "enabled": True,
                    "min_interval_seconds": -1,
                },
            )

    def test_missing_usage_is_supported(self):
        response = JudgeResponse(
            raw_response=self.valid_response.raw_response,
            provider="groq",
            model="test-model",
            usage=None,
        )

        mock_judge = MagicMock()
        mock_judge.evaluate.return_value = response

        evaluator = ContextualEvaluator(judge=mock_judge)
        result = evaluator.evaluate_pair(self.request)

        self.assertEqual(result["status"], "success")
        self.assertIsNone(result["usage"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
