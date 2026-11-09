
"""Contextual judging, response validation, API retries and request pacing."""

from __future__ import annotations

import re
import time
from dataclasses import replace
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from .judge.base import ContextualJudge, JudgeRequest
from .judge.response_validator import validate_judge_response


RETRYABLE_API_EXCEPTIONS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)


class RequestPacer:
    """Enforce a minimum interval between API request starts."""

    def __init__(
        self,
        enabled: bool = False,
        min_interval_seconds: float = 0.0,
    ) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("request_pacing.enabled must be a boolean.")

        if (
            isinstance(min_interval_seconds, bool)
            or not isinstance(min_interval_seconds, (int, float))
            or min_interval_seconds < 0
        ):
            raise ValueError(
                "request_pacing.min_interval_seconds must be "
                "a non-negative number."
            )

        self.enabled = enabled
        self.min_interval_seconds = float(min_interval_seconds)
        self._last_request_started_at: float | None = None

    def wait(self) -> float:
        """Wait if needed and return the time spent waiting."""

        if not self.enabled or self.min_interval_seconds == 0:
            self._last_request_started_at = time.monotonic()
            return 0.0

        now = time.monotonic()

        if self._last_request_started_at is None:
            self._last_request_started_at = now
            return 0.0

        elapsed = now - self._last_request_started_at
        remaining = self.min_interval_seconds - elapsed

        if remaining > 0:
            time.sleep(remaining)

        self._last_request_started_at = time.monotonic()
        return max(remaining, 0.0)


class ContextualEvaluator:
    """
    Coordinate contextual judging, validation, API retries and pacing.

    Validation retries and API retries are deliberately independent.

    - Validation retry: the API returned a response, but it was invalid.
    - API retry: the request failed with a retryable temporary API error.
    - review_required: all validation attempts returned invalid responses.
    - api_error: a non-retryable API error occurred or API retries were
      exhausted.
    """

    def __init__(
        self,
        judge: ContextualJudge,
        max_retries: int = 1,
        api_retry: dict[str, Any] | None = None,
        request_pacing: dict[str, Any] | None = None,
    ) -> None:
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
        ):
            raise ValueError("max_retries must be an integer.")

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        self.judge = judge
        self.max_retries = max_retries

        self.api_retry = self._validate_api_retry(api_retry or {})
        pacing = request_pacing or {}

        self.pacer = RequestPacer(
            enabled=pacing.get("enabled", False),
            min_interval_seconds=pacing.get(
                "min_interval_seconds", 0.0
            ),
        )

    @staticmethod
    def _validate_api_retry(
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate and normalize API retry configuration."""

        defaults = {
            "max_retries": 0,
            "initial_delay_seconds": 1.0,
            "max_delay_seconds": 60.0,
            "backoff_multiplier": 2.0,
        }
        config = {**defaults, **settings}

        max_retries = config["max_retries"]
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or max_retries < 0
        ):
            raise ValueError(
                "api_retry.max_retries must be a non-negative integer."
            )

        for key in (
            "initial_delay_seconds",
            "max_delay_seconds",
            "backoff_multiplier",
        ):
            value = config[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise ValueError(
                    f"api_retry.{key} must be a non-negative number."
                )

        if config["backoff_multiplier"] < 1:
            raise ValueError(
                "api_retry.backoff_multiplier must be at least 1."
            )

        if (
            config["max_delay_seconds"]
            < config["initial_delay_seconds"]
        ):
            raise ValueError(
                "api_retry.max_delay_seconds cannot be less than "
                "initial_delay_seconds."
            )

        return config

    @staticmethod
    def _is_retryable_api_error(exc: Exception) -> bool:
        """Return whether an exception represents a temporary API failure."""

        if isinstance(exc, RETRYABLE_API_EXCEPTIONS):
            return True

        if isinstance(exc, APIStatusError):
            return exc.status_code == 429 or exc.status_code >= 500

        return False

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        """Extract a provider-suggested retry delay when available."""

        headers = getattr(exc, "headers", None)

        if headers:
            retry_after = headers.get("retry-after")
            if retry_after is not None:
                try:
                    return max(float(retry_after), 0.0)
                except (TypeError, ValueError):
                    pass

        # Some providers include text such as:
        # "Please try again in 7.07s"
        match = re.search(
            r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
            str(exc),
            flags=re.IGNORECASE,
        )

        if match:
            return float(match.group(1))

        return None

    def _calculate_retry_delay(
        self,
        retry_number: int,
        exc: Exception,
    ) -> float:
        """Calculate bounded exponential backoff."""

        initial = float(
            self.api_retry["initial_delay_seconds"]
        )
        maximum = float(self.api_retry["max_delay_seconds"])
        multiplier = float(
            self.api_retry["backoff_multiplier"]
        )

        exponential_delay = min(
            initial * (multiplier ** (retry_number - 1)),
            maximum,
        )

        provider_delay = self._retry_after_seconds(exc)

        if provider_delay is not None:
            return min(max(provider_delay, exponential_delay), maximum)

        return exponential_delay

    def _call_judge(
        self,
        request: JudgeRequest,
        validation_attempt: int,
        validation_feedback: str | None,
        attempts: list[dict[str, Any]],
    ):
        """
        Call the judge with bounded API retries.

        Each actual API call is recorded in `attempts`.
        Returns (response, error).
        """

        max_api_retries = self.api_retry["max_retries"]

        for api_attempt in range(1, max_api_retries + 2):
            pacing_wait = self.pacer.wait()
            request_started_at = time.perf_counter()

            try:
                response = self.judge.evaluate(request)

                attempts.append(
                    {
                        "attempt": len(attempts) + 1,
                        "validation_attempt": validation_attempt,
                        "api_attempt": api_attempt,
                        "status": "api_response_received",
                        "validation_feedback": validation_feedback,
                        "provider": response.provider,
                        "model": response.model,
                        "usage": response.usage,
                        "pacing_wait_seconds": pacing_wait,
                        "latency_seconds": (
                            time.perf_counter() - request_started_at
                        ),
                    }
                )

                return response, None

            except Exception as exc:
                latency = time.perf_counter() - request_started_at
                retryable = self._is_retryable_api_error(exc)
                retries_remaining = api_attempt <= max_api_retries

                if retryable and retries_remaining:
                    delay = self._calculate_retry_delay(
                        api_attempt,
                        exc,
                    )

                    attempts.append(
                        {
                            "attempt": len(attempts) + 1,
                            "validation_attempt": validation_attempt,
                            "api_attempt": api_attempt,
                            "status": "api_retry",
                            "validation_feedback": validation_feedback,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "retry_delay_seconds": delay,
                            "pacing_wait_seconds": pacing_wait,
                            "latency_seconds": latency,
                        }
                    )

                    time.sleep(delay)
                    continue

                attempts.append(
                    {
                        "attempt": len(attempts) + 1,
                        "validation_attempt": validation_attempt,
                        "api_attempt": api_attempt,
                        "status": "api_error",
                        "validation_feedback": validation_feedback,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "retryable": retryable,
                        "pacing_wait_seconds": pacing_wait,
                        "latency_seconds": latency,
                    }
                )

                return None, exc

        # Defensive fallback; the loop should always return.
        return None, RuntimeError("API retry loop ended unexpectedly.")

    def evaluate_pair(
        self,
        request: JudgeRequest,
    ) -> dict[str, Any]:
        """Evaluate one question-answer-passage pair."""

        started_at = time.perf_counter()
        attempts: list[dict[str, Any]] = []

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        usage_available = True

        validation_feedback = None
        last_response = None

        for validation_attempt in range(
            1,
            self.max_retries + 2,
        ):
            attempt_request = replace(
                request,
                validation_feedback=validation_feedback,
            )

            response, api_error = self._call_judge(
                request=attempt_request,
                validation_attempt=validation_attempt,
                validation_feedback=validation_feedback,
                attempts=attempts,
            )

            if api_error is not None:
                return self._build_result(
                    status="api_error",
                    classification=None,
                    evidence_quote=None,
                    rationale=None,
                    raw_response=None,
                    provider=(
                        last_response.provider
                        if last_response is not None
                        else None
                    ),
                    model=(
                        last_response.model
                        if last_response is not None
                        else None
                    ),
                    usage=self._build_usage(
                        total_prompt_tokens,
                        total_completion_tokens,
                        total_tokens,
                        usage_available,
                    ),
                    attempts=attempts,
                    total_latency=time.perf_counter() - started_at,
                )

            last_response = response
            usage = response.usage

            if usage is None:
                usage_available = False
            else:
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get(
                    "completion_tokens", 0
                )
                total_tokens += usage.get("total_tokens", 0)

            validation = validate_judge_response(
                raw_response=response.raw_response,
                original_passage=request.passage_text,
            )

            # Update the API response record with its validation outcome.
            attempts[-1]["status"] = (
                "valid" if validation.valid else "invalid_response"
            )
            attempts[-1]["raw_response"] = response.raw_response
            attempts[-1]["validation_errors"] = validation.errors

            if validation.valid:
                data = validation.data

                return self._build_result(
                    status="success",
                    classification=data["classification"],
                    evidence_quote=data["evidence_quote"],
                    rationale=data["rationale"],
                    raw_response=response.raw_response,
                    provider=response.provider,
                    model=response.model,
                    usage=self._build_usage(
                        total_prompt_tokens,
                        total_completion_tokens,
                        total_tokens,
                        usage_available,
                    ),
                    attempts=attempts,
                    total_latency=time.perf_counter() - started_at,
                )

            validation_feedback = (
                "Your previous response failed validation.\n"
                "Validation errors:\n"
                + "\n".join(
                    f"- {error}" for error in validation.errors
                )
                + "\n\nCorrect the response. The evidence_quote must "
                "be copied exactly from the original passage. "
                "Preserve unusual characters and whitespace. "
                "Prefer a shorter exact quotation if necessary. "
                "Return the required JSON object."
            )

        return self._build_result(
            status="review_required",
            classification=None,
            evidence_quote=None,
            rationale=None,
            raw_response=(
                last_response.raw_response
                if last_response is not None
                else None
            ),
            provider=(
                last_response.provider
                if last_response is not None
                else None
            ),
            model=(
                last_response.model
                if last_response is not None
                else None
            ),
            usage=self._build_usage(
                total_prompt_tokens,
                total_completion_tokens,
                total_tokens,
                usage_available,
            ),
            attempts=attempts,
            total_latency=time.perf_counter() - started_at,
        )

    @staticmethod
    def _build_usage(
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        usage_available: bool,
    ) -> dict[str, int] | None:
        if not usage_available:
            return None

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _build_result(
        *,
        status: str,
        classification: str | None,
        evidence_quote: str | None,
        rationale: str | None,
        raw_response: str | None,
        provider: str | None,
        model: str | None,
        usage: dict[str, Any] | None,
        attempts: list[dict[str, Any]],
        total_latency: float,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "classification": classification,
            "evidence_quote": evidence_quote,
            "rationale": rationale,
            "raw_response": raw_response,
            "provider": provider,
            "model": model,
            "usage": usage,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "latency_seconds": total_latency,
        }
