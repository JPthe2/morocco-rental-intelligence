"""Thin client for an OpenAI-compatible chat completions endpoint (OpenRouter
or Gemini, selected via config.LLM_PROVIDER). Knows nothing about tools.py or
any particular agent — every caller passes its own tool schemas.
"""
import requests

import config


class DegradedResponseError(Exception):
    """Raised when the LLM backend can't be used (missing key, quota exhausted,
    network error). Callers should catch this and degrade gracefully rather
    than crash."""


def call_llm(messages: list[dict], tool_schemas: list[dict] | None = None, tool_choice="auto") -> dict:
    if not config.LLM_API_KEY:
        raise DegradedResponseError(
            f"No API key is configured for LLM_PROVIDER={config.LLM_PROVIDER}. "
            "Set it in agent_service/.env to enable chat."
        )
    try:
        resp = requests.post(
            f"{config.LLM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.LLM_MODEL,
                "messages": messages,
                "tools": tool_schemas or [],
                "tool_choice": tool_choice,
            },
            timeout=60,
        )
    except requests.RequestException as exc:
        raise DegradedResponseError(f"Could not reach {config.LLM_PROVIDER}: {exc}") from exc

    if resp.status_code == 429:
        raise DegradedResponseError(
            f"{config.LLM_PROVIDER}'s free-tier quota looks exhausted right now (HTTP 429). "
            "Try again shortly, or switch LLM_PROVIDER / add credits in agent_service/.env."
        )
    if resp.status_code == 402:
        raise DegradedResponseError(f"{config.LLM_PROVIDER} reports insufficient credits for this model (HTTP 402).")
    if not resp.ok:
        raise DegradedResponseError(f"{config.LLM_PROVIDER} request failed: HTTP {resp.status_code} — {resp.text[:300]}")

    return resp.json()
