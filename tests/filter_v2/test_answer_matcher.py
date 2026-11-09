
import unittest

from rag_security_research.filter_v2.answer_matcher import (
    find_answer_occurrences,
    match_pair,
)


class TestAnswerMatcher(unittest.TestCase):

    def test_case_insensitive_matching(self):
        matches = find_answer_occurrences(
            "Chicago",
            "The CHICAGO Fire season was successful.",
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["matched_text"], "CHICAGO")

    def test_punctuation_normalization(self):
        matches = find_answer_occurrences(
            "Spider-Man",
            "Spider-Man appeared in the movie.",
        )

        self.assertEqual(len(matches), 1)

    def test_token_boundary(self):
        matches = find_answer_occurrences(
            "art",
            "The artist created an artwork.",
        )

        self.assertEqual(len(matches), 0)

    def test_multiple_occurrences(self):
        matches = find_answer_occurrences(
            "23",
            "There were 23 episodes in season 4 and 23 in season 5.",
        )

        self.assertEqual(len(matches), 2)

    def test_no_match(self):
        matches = find_answer_occurrences(
            "London",
            "The capital of France is Paris.",
        )

        self.assertEqual(len(matches), 0)

    def test_body_match_is_detected(self):
        pair = {
            "pair_id": "test_pair",
            "beir_query_id": "test1",
            "answer_variant_index": 0,
            "answer_variant": "23",
            "passage_id": "doc6",
            "passage_title": "Chicago Fire Season 4",
            "passage_text": "The season consists of 23 episodes.",
        }

        result = match_pair(pair)

        self.assertTrue(result["match_found"])
        self.assertEqual(
            result["matching_method"],
            "body_only",
        )
        self.assertEqual(result["title_match_count"], 0)
        self.assertEqual(result["body_match_count"], 1)
        self.assertEqual(len(result["occurrences"]), 1)
        self.assertEqual(
            result["occurrences"][0]["match_source"],
            "body",
        )

    def test_title_only_match_is_ignored(self):
        pair = {
            "pair_id": "p1",
            "beir_query_id": "q1",
            "answer_variant_index": 0,
            "answer_variant": "Charles Darwin",
            "passage_id": "d1",
            "passage_title": "Charles Darwin",
            "passage_text": "He published several scientific works.",
        }

        result = match_pair(pair)

        self.assertFalse(result["match_found"])
        self.assertEqual(
            result["matching_method"],
            "no_exact_match",
        )
        self.assertEqual(result["title_match_count"], 0)
        self.assertEqual(result["body_match_count"], 0)
        self.assertEqual(result["occurrences"], [])

    def test_title_match_does_not_affect_body_match(self):
        pair = {
            "pair_id": "p2",
            "beir_query_id": "q2",
            "answer_variant_index": 0,
            "answer_variant": "Charles Darwin",
            "passage_id": "d2",
            "passage_title": "A scientific biography",
            "passage_text": "Charles Darwin wrote On the Origin of Species.",
        }

        result = match_pair(pair)

        self.assertTrue(result["match_found"])
        self.assertEqual(result["matching_method"], "body_only")
        self.assertEqual(result["title_match_count"], 0)
        self.assertEqual(result["body_match_count"], 1)

    def test_non_breaking_space_matching(self):
        matches = find_answer_occurrences(
            "70 million",
            "The population was 70\u00a0million people.",
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(
            matches[0]["matched_text"],
            "70\u00a0million",
        )

    def test_mojibake_before_non_breaking_space(self):
        original_text = "The population was 70\u00c2\u00a0million people."

        matches = find_answer_occurrences(
            "70 million",
            original_text,
        )

        self.assertEqual(len(matches), 1)

        # The original text, including the artefact, is preserved.
        self.assertEqual(
            matches[0]["matched_text"],
            "70\u00c2\u00a0million",
        )

    def test_mojibake_repair_does_not_remove_standalone_a_circumflex(self):
        matches = find_answer_occurrences(
            "Â million",
            "The text contains Â million people.",
        )

        self.assertEqual(len(matches), 1)

    def test_mojibake_match_offsets(self):
        original_text = "Population: 70\u00c2\u00a0million people."

        matches = find_answer_occurrences(
            "70 million",
            original_text,
        )

        self.assertEqual(len(matches), 1)

        occurrence = matches[0]

        self.assertEqual(
            original_text[
                occurrence["original_start"]:
                occurrence["original_end"]
            ],
            "70\u00c2\u00a0million",
        )


if __name__ == "__main__":
    unittest.main()
