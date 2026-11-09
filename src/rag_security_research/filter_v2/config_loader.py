from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


DEFAULT_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


class ConfigError(Exception):
    """Raised when the project configuration is missing or invalid."""


def _require_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{name} must be a number.")

    if strictly_positive and value <= 0:
        raise ConfigError(f"{name} must be positive.")

    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}.")

    return float(value)


def _require_non_negative_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer.")

    return value


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load project YAML and environment variables from the project root."""

    config_file = Path(config_path).resolve()

    if not config_file.is_file():
        raise ConfigError(f"Configuration file not found: {config_file}")

    project_root = config_file.parent
    env_file = project_root / ".env"
    load_dotenv(dotenv_path=env_file, override=False)

    try:
        with config_file.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Invalid YAML in {config_file}: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ConfigError(
            "The configuration file must contain a YAML mapping."
        )

    return config


def get_judge_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and return provider-independent judge settings."""

    try:
        settings = config["filter_v2"]["judge"]
    except (KeyError, TypeError) as exc:
        raise ConfigError(
            "Missing configuration section: filter_v2.judge"
        ) from exc

    if not isinstance(settings, dict):
        raise ConfigError("filter_v2.judge must be a YAML mapping.")

    provider = settings.get("provider")
    model = settings.get("model")

    if not isinstance(provider, str) or not provider.strip():
        raise ConfigError("Judge provider must be a non-empty string.")

    provider = provider.strip().lower()

    if provider not in DEFAULT_BASE_URLS:
        raise ConfigError(
            f"Unsupported judge provider: {provider}. "
            f"Supported providers: {', '.join(DEFAULT_BASE_URLS)}."
        )

    if not isinstance(model, str) or not model.strip():
        raise ConfigError("Judge model must be a non-empty string.")

    model = model.strip()

    if model.startswith("YOUR_"):
        raise ConfigError(
            "Replace the placeholder model in config.yaml "
            "with an actual model identifier."
        )

    api_key_env = settings.get("api_key_env")

    if api_key_env is not None:
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise ConfigError(
                "Judge api_key_env must be a non-empty string or null."
            )
        api_key_env = api_key_env.strip()

    if provider == "groq" and not api_key_env:
        api_key_env = "GROQ_API_KEY"
    elif provider == "gemini" and not api_key_env:
        api_key_env = "GEMINI_API_KEY"

    if provider == "ollama":
        # Local Ollama normally does not require an API key.
        api_key = (
            os.getenv(api_key_env)
            if api_key_env
            else "ollama"
        )
        api_key = api_key or "ollama"
    else:
        api_key = os.getenv(api_key_env) if api_key_env else None

        if not api_key:
            raise ConfigError(
                f"{api_key_env} was not found. "
                "Add it to the project-root .env file."
            )

    base_url = settings.get("base_url") or DEFAULT_BASE_URLS[provider]

    if not isinstance(base_url, str) or not base_url.strip():
        raise ConfigError("Judge base_url must be a non-empty string.")

    temperature = _require_number(
        settings.get("temperature", 0),
        "Judge temperature",
        minimum=0,
    )

    timeout = _require_number(
        settings.get("timeout", 60),
        "Judge timeout",
        strictly_positive=True,
    )

    max_retries = _require_non_negative_integer(
        settings.get("max_retries", 1),
        "Judge max_retries",
    )

    api_retry = settings.get("api_retry", {})
    if not isinstance(api_retry, dict):
        raise ConfigError("Judge api_retry must be a YAML mapping.")

    api_max_retries = _require_non_negative_integer(
        api_retry.get("max_retries", 3),
        "Judge api_retry.max_retries",
    )

    initial_delay = _require_number(
        api_retry.get("initial_delay_seconds", 5),
        "Judge api_retry.initial_delay_seconds",
        minimum=0,
    )

    max_delay = _require_number(
        api_retry.get("max_delay_seconds", 60),
        "Judge api_retry.max_delay_seconds",
        minimum=0,
    )

    backoff_multiplier = _require_number(
        api_retry.get("backoff_multiplier", 2),
        "Judge api_retry.backoff_multiplier",
        minimum=1,
    )

    if max_delay < initial_delay:
        raise ConfigError(
            "Judge api_retry.max_delay_seconds cannot be less than "
            "initial_delay_seconds."
        )

    pacing = settings.get("request_pacing", {})
    if not isinstance(pacing, dict):
        raise ConfigError(
            "Judge request_pacing must be a YAML mapping."
        )

    pacing_enabled = pacing.get("enabled", True)
    if not isinstance(pacing_enabled, bool):
        raise ConfigError(
            "Judge request_pacing.enabled must be a boolean."
        )

    min_interval = _require_number(
        pacing.get("min_interval_seconds", 2),
        "Judge request_pacing.min_interval_seconds",
        minimum=0,
    )

    response_format = settings.get("response_format", "json")
    if response_format not in {"json", "text"}:
        raise ConfigError(
            "Judge response_format must be 'json' or 'text'."
        )

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "api_key_env": api_key_env,
        "base_url": base_url.strip(),
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": max_retries,
        "api_retry": {
            "max_retries": api_max_retries,
            "initial_delay_seconds": initial_delay,
            "max_delay_seconds": max_delay,
            "backoff_multiplier": backoff_multiplier,
        },
        "request_pacing": {
            "enabled": pacing_enabled,
            "min_interval_seconds": min_interval,
        },
        "response_format": response_format,
        "prompt_version": settings.get(
            "prompt_version",
            "filter_v2_judge_v2",
        ),
    }


def resolve_project_path(
    config: dict[str, Any],
    path_key: str,
    config_path: str | Path = "config.yaml",
) -> Path:
    """Resolve a configured project path relative to the project root.

    Supports top-level keys (e.g. 'results') and dotted nested keys
    (e.g. 'filter_v2.questions', 'datasets.beir_nq.corpus').
    """

    try:
        paths_section = config["paths"]
        if path_key in paths_section:
            configured_path = paths_section[path_key]
        else:
            current: Any = paths_section
            for part in path_key.split("."):
                current = current[part]
            configured_path = current
    except (KeyError, TypeError) as exc:
        raise ConfigError(
            f"Missing configured path: paths.{path_key}"
        ) from exc

    project_root = Path(config_path).resolve().parent
    path = Path(configured_path)

    if path.is_absolute():
        return path

    return (project_root / path).resolve()