from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

from llm_api_config import (
    SUPPORTED_MODELS,
    call_claude_messages,
    config_for_model,
    public_config,
    require_shared_api_key,
)


DEFAULT_MODELS = ["gpt-5.1", "gpt-5.4", "gpt-5.6-luna", "claude-haiku-4.5"]


def smoke_openai(config: Any) -> dict[str, Any]:
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=config.endpoint,
        api_key=require_shared_api_key(),
        api_version=config.api_version,
        timeout=60.0,
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=config.deployment,
        messages=[{"role": "user", "content": "Reply with exactly API_OK."}],
        max_completion_tokens=128,
    )
    content = response.choices[0].message.content or ""
    return {
        "request_id": getattr(response, "id", ""),
        "response_text": content.strip()[:120],
    }


def smoke_claude(config: Any) -> dict[str, Any]:
    response = call_claude_messages(
        config,
        {
            "model": config.deployment,
            "max_tokens": 64,
            "temperature": 0,
            "messages": [
                {"role": "user", "content": "Reply with exactly API_OK."},
            ],
        },
        timeout=60.0,
    )
    text = "\n".join(
        str(block.get("text", ""))
        for block in response.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
    return {
        "request_id": response.get("id", ""),
        "response_text": text[:120],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send exactly one minimal request to each configured Azure LLM deployment."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(SUPPORTED_MODELS),
        default=DEFAULT_MODELS,
    )
    parser.add_argument("--show-config-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    api_key = require_shared_api_key()
    key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
    results = []
    failures = 0

    print(
        json.dumps(
            {
                "shared_api_key": "SET",
                "shared_api_key_sha256_prefix": key_fingerprint,
                "models": [public_config(config_for_model(model)) for model in args.models],
            },
            indent=2,
        )
    )
    if args.show_config_only:
        return

    for model in args.models:
        config = config_for_model(model)
        try:
            response = (
                smoke_openai(config)
                if config.provider == "azure_openai"
                else smoke_claude(config)
            )
            result = {"model": model, "status": "PASS", **response}
        except Exception as exc:
            failures += 1
            result = {
                "model": model,
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results.append(result)
        print(json.dumps(result, ensure_ascii=True))

    print(json.dumps({"smoke_test_results": results}, indent=2, ensure_ascii=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
