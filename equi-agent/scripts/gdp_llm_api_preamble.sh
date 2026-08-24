#!/usr/bin/env bash

# Shared Azure API constants for GDP progression LLM runs. These are fixed to
# the deployed resource so stale shell or .env endpoint values cannot silently
# redirect a run. The API key is intentionally not stored here.
export AZURE_OPENAI_ENDPOINT="https://azure-openai-radi.cognitiveservices.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"

export GPT51_DEPLOYMENT="gpt-5.1"
export GPT54_DEPLOYMENT="gpt-5.4"
export GPT56_DEPLOYMENT="gpt-5.6-luna"

export CLAUDE_HAIKU45_DEPLOYMENT="claude-haiku-4-5"
export CLAUDE_HAIKU45_MESSAGES_URL="https://azure-openai-radi.services.ai.azure.com/anthropic/v1/messages"
export ANTHROPIC_VERSION="2023-06-01"

# Compatibility value for code that still uses the Anthropic Foundry SDK.
export ANTHROPIC_FOUNDRY_BASE_URL="https://azure-openai-radi.services.ai.azure.com/anthropic"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '%s\n' \
    "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT" \
    "AZURE_OPENAI_API_VERSION=$AZURE_OPENAI_API_VERSION" \
    "GPT51_DEPLOYMENT=$GPT51_DEPLOYMENT" \
    "GPT54_DEPLOYMENT=$GPT54_DEPLOYMENT" \
    "GPT56_DEPLOYMENT=$GPT56_DEPLOYMENT" \
    "CLAUDE_HAIKU45_DEPLOYMENT=$CLAUDE_HAIKU45_DEPLOYMENT" \
    "CLAUDE_HAIKU45_MESSAGES_URL=$CLAUDE_HAIKU45_MESSAGES_URL" \
    "ANTHROPIC_VERSION=$ANTHROPIC_VERSION" \
    "AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY:+SET (shared by all four models)}"
fi
