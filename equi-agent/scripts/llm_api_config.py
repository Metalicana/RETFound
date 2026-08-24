from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OPENAI_MODELS = {"gpt-5.1", "gpt-5.4", "gpt-5.6-luna"}
CLAUDE_MODELS = {"claude-haiku-4.5"}
SUPPORTED_MODELS = OPENAI_MODELS | CLAUDE_MODELS


@dataclass(frozen=True)
class LLMAPIConfig:
    model_name: str
    provider: str
    deployment: str
    endpoint: str
    api_version: str


class LLMAPIError(RuntimeError):
    def __init__(self, provider: str, status_code: int | None, message: str):
        self.provider = provider
        self.status_code = status_code
        super().__init__(
            f"{provider} API error"
            + (f" HTTP {status_code}" if status_code is not None else "")
            + f": {message}"
        )


def deployment_for_model(model_name: str) -> str:
    variables = {
        "gpt-5.1": "GPT51_DEPLOYMENT",
        "gpt-5.4": "GPT54_DEPLOYMENT",
        "gpt-5.6-luna": "GPT56_DEPLOYMENT",
        "claude-haiku-4.5": "CLAUDE_HAIKU45_DEPLOYMENT",
    }
    defaults = {
        "gpt-5.1": "gpt-5.1",
        "gpt-5.4": "gpt-5.4",
        "gpt-5.6-luna": "gpt-5.6-luna",
        "claude-haiku-4.5": "claude-haiku-4-5",
    }
    if model_name not in variables:
        raise ValueError(f"Unsupported model: {model_name}")
    return os.getenv(variables[model_name], defaults[model_name])


def config_for_model(model_name: str, deployment: str | None = None) -> LLMAPIConfig:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    if model_name in OPENAI_MODELS:
        return LLMAPIConfig(
            model_name=model_name,
            provider="azure_openai",
            deployment=deployment or deployment_for_model(model_name),
            endpoint=os.getenv(
                "AZURE_OPENAI_ENDPOINT",
                "https://azure-openai-radi.cognitiveservices.azure.com/",
            ).rstrip("/")
            + "/",
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
    return LLMAPIConfig(
        model_name=model_name,
        provider="azure_anthropic",
        deployment=deployment or deployment_for_model(model_name),
        endpoint=os.getenv(
            "CLAUDE_HAIKU45_MESSAGES_URL",
            "https://azure-openai-radi.services.ai.azure.com/anthropic/v1/messages",
        ),
        api_version=os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
    )


def require_shared_api_key() -> str:
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set AZURE_OPENAI_API_KEY; the same key is used for all four models")
    return api_key


def public_config(config: LLMAPIConfig) -> dict[str, str]:
    return asdict(config)


def call_claude_messages(
    config: LLMAPIConfig,
    payload: dict[str, Any],
    *,
    timeout: float = 120.0,
) -> dict[str, Any]:
    if config.provider != "azure_anthropic":
        raise ValueError(f"Expected Azure Anthropic config, found {config.provider}")
    request = Request(
        config.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": require_shared_api_key(),
            "anthropic-version": config.api_version,
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise LLMAPIError(config.provider, exc.code, body) from exc
    except URLError as exc:
        raise LLMAPIError(config.provider, None, str(exc.reason)) from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise LLMAPIError(config.provider, None, "response was not a JSON object")
    return parsed


def is_non_retryable_api_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    return status_code in {400, 401, 403, 404}
