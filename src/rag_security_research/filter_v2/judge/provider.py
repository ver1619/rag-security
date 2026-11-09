from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import OpenAI

from ..config_loader import get_judge_config, load_config
from .base import ContextualJudge, JudgeRequest, JudgeResponse
from .prompt import SYSTEM_PROMPT, build_user_prompt


class OpenAICompatibleContextualJudge(ContextualJudge):
    """Shared implementation for OpenAI-compatible judge providers."""

    SUPPORTED_PROVIDERS = {"groq", "ollama", "gemini"}

    def __init__(
        self,
        config_path: str | Path = "config.yaml",
        *,
        expected_provider: str | None = None,
    ) -> None:
        config = load_config(config_path)
        self.settings = get_judge_config(config)

        self.provider = self.settings["provider"]

        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {self.provider}"
            )

        if (
            expected_provider is not None
            and self.provider != expected_provider
        ):
            raise ValueError(
                f"This judge requires provider='{expected_provider}', "
                f"but configuration specifies '{self.provider}'."
            )

        self.model = self.settings["model"]
        self.temperature = self.settings["temperature"]

        # SDK retries remain disabled. Retry handling belongs to
        # ContextualEvaluator, where every attempt can be recorded.
        self.client = OpenAI(
            api_key=self.settings["api_key"],
            base_url=self.settings["base_url"],
            timeout=self.settings["timeout"],
            max_retries=0,
        )

    def evaluate(self, request: JudgeRequest) -> JudgeResponse:
        """Send one evaluation request and return the raw response."""

        user_prompt = build_user_prompt(
            question=request.question,
            answer_variant=request.answer_variant,
            passage_title=request.passage_title,
            passage_text=request.passage_text,
            matching_method=request.matching_method,
            occurrences=request.occurrences,
            validation_feedback=request.validation_feedback,
        )

        api_arguments: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": self.temperature,
        }

        if self.settings["response_format"] == "json":
            api_arguments["response_format"] = {
                "type": "json_object"
            }

        response = self.client.chat.completions.create(
            **api_arguments
        )

        if not response.choices:
            raise RuntimeError(
                f"{self.provider} returned an empty choices list."
            )

        raw_response = response.choices[0].message.content

        if not isinstance(raw_response, str) or not raw_response.strip():
            raise RuntimeError(
                f"{self.provider} returned empty or invalid message content."
            )

        usage: dict[str, Any] | None = None

        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return JudgeResponse(
            raw_response=raw_response,
            provider=self.provider,
            model=self.model,
            usage=usage,
        )


class GroqContextualJudge(OpenAICompatibleContextualJudge):
    """Groq implementation of the contextual judge."""

    def __init__(
        self,
        config_path: str | Path = "config.yaml",
    ) -> None:
        super().__init__(
            config_path,
            expected_provider="groq",
        )


class OllamaContextualJudge(OpenAICompatibleContextualJudge):
    """Local Ollama implementation of the contextual judge."""

    def __init__(
        self,
        config_path: str | Path = "config.yaml",
    ) -> None:
        super().__init__(
            config_path,
            expected_provider="ollama",
        )



class GeminiContextualJudge(OpenAICompatibleContextualJudge):
    """Google Gemini judge using Google's OpenAI-compatible API endpoint."""

    def __init__(
        self,
        config_path: str | Path = "config.yaml",
    ) -> None:
        super().__init__(
            config_path,
            expected_provider="gemini",
        )


def create_contextual_judge(
    config_path: str | Path = "config.yaml",
) -> ContextualJudge:
    """Create the configured contextual judge implementation."""

    config = load_config(config_path)
    settings = get_judge_config(config)

    provider = settings["provider"]

    if provider == "groq":
        return GroqContextualJudge(config_path)

    if provider == "ollama":
        return OllamaContextualJudge(config_path)

    if provider == "gemini":
        return GeminiContextualJudge(config_path)

    raise ValueError(
        f"No contextual judge implementation registered for '{provider}'."
    )