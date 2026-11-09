
import unittest

from rag_security_research.filter_v2.judge.response_validator import (
    validate_judge_response,
)


class TestResponseValidator(unittest.TestCase):

    def setUp(self):
        self.passage = (
            "Chicago Fire Season 4 consists of 23 episodes."
        )

    def test_valid_response(self):
        response = """
        {
            "classification": "direct_support",
            "evidence_quote": "23 episodes",
            "rationale": "The passage explicitly states the count."
        }
        """

        result = validate_judge_response(
            response,
            self.passage,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.errors, [])

    def test_invalid_json(self):
        result = validate_judge_response(
            "This is not JSON",
            self.passage,
        )

        self.assertFalse(result.valid)

    def test_invalid_classification(self):
        response = """
        {
            "classification": "probably_supported",
            "evidence_quote": "23 episodes",
            "rationale": "The passage states the count."
        }
        """

        result = validate_judge_response(
            response,
            self.passage,
        )

        self.assertFalse(result.valid)

    def test_invented_evidence_quote(self):
        response = """
        {
            "classification": "direct_support",
            "evidence_quote": "The show has 24 episodes.",
            "rationale": "The passage establishes the count."
        }
        """

        result = validate_judge_response(
            response,
            self.passage,
        )

        self.assertFalse(result.valid)

    def test_empty_quote_allowed(self):
        response = """
        {
            "classification": "no_support",
            "evidence_quote": "",
            "rationale": "The passage does not establish the answer."
        }
        """

        result = validate_judge_response(
            response,
            self.passage,
        )

        self.assertTrue(result.valid)

    def test_missing_rationale(self):
        response = """
        {
            "classification": "direct_support",
            "evidence_quote": "23 episodes"
        }
        """

        result = validate_judge_response(
            response,
            self.passage,
        )

        self.assertFalse(result.valid)

    def test_unexpected_field(self):
        response = """
        {
            "classification": "direct_support",
            "evidence_quote": "23 episodes",
            "rationale": "The passage establishes the count.",
            "confidence": 0.99
        }
        """

        result = validate_judge_response(
            response,
            self.passage,
        )

        self.assertFalse(result.valid)


if __name__ == "__main__":
    unittest.main()
