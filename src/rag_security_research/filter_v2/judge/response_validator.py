
"""Validate contextual judge responses."""

from __future__ import annotations

import json
from dataclasses import dataclass


ALLOWED_CLASSIFICATIONS = {
    "direct_support",
    "semantic_support",
    "partial_support",
    "incidental_match",
    "contradictory",
    "no_support",
    "ambiguous",
}


@dataclass
class ValidationResult:
    valid: bool
    data: dict | None
    errors: list[str]


def validate_judge_response(
    raw_response: str,
    original_passage: str,
) -> ValidationResult:
    """Validate JSON structure, classification and evidence quotation."""

    errors = []

    try:
        data = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        return ValidationResult(
            valid=False,
            data=None,
            errors=["Response is not valid JSON."],
        )

    if not isinstance(data, dict):
        return ValidationResult(
            valid=False,
            data=None,
            errors=["Response root must be a JSON object."],
        )

    required_fields = {
        "classification",
        "evidence_quote",
        "rationale",
    }

    missing = required_fields - data.keys()

    if missing:
        errors.append(
            f"Missing required fields: {sorted(missing)}"
        )

    extra = data.keys() - required_fields

    if extra:
        errors.append(
            f"Unexpected fields: {sorted(extra)}"
        )

    classification = data.get("classification")

    if classification not in ALLOWED_CLASSIFICATIONS:
        errors.append("Invalid classification label.")

    evidence_quote = data.get("evidence_quote")

    if not isinstance(evidence_quote, str):
        errors.append("Evidence quote must be a string.")
    elif evidence_quote and evidence_quote not in original_passage:
        errors.append(
            "Evidence quote is not an exact substring "
            "of the original passage."
        )

    rationale = data.get("rationale")

    if not isinstance(rationale, str) or not rationale.strip():
        errors.append("Rationale must be a non-empty string.")

    if errors:
        return ValidationResult(
            valid=False,
            data=data,
            errors=errors,
        )

    return ValidationResult(
        valid=True,
        data=data,
        errors=[],
    )
