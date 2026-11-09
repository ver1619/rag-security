
"""Provider-independent interface for the Filter v2 contextual judge."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class JudgeRequest:
    question: str
    answer_variant: str
    passage_title: str
    passage_text: str
    matching_method: str
    occurrences: list[dict[str, Any]]
    validation_feedback: str | None = None


@dataclass
class JudgeResponse:
    raw_response: str
    provider: str
    model: str
    usage: dict[str, Any] | None = None


class ContextualJudge(ABC):
    """Common interface implemented by all judge providers."""

    @abstractmethod
    def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        """Evaluate one answer-variant/passage pair."""
        raise NotImplementedError
