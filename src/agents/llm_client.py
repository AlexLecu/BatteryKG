"""Thin, provider-agnostic LLM client (groq / ollama / anthropic).

One function — complete(system, user, config) -> str — behind which providers
are interchangeable. No provider SDKs; plain HTTPS via requests. Credentials
come from .env / environment:
    GROQ_API_KEY       (provider 'groq')
    ANTHROPIC_API_KEY  (provider 'anthropic')
    OLLAMA_HOST        (provider 'ollama'; default http://localhost:11434)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass(frozen=True)
class ModelConfig:
    provider: str                    # groq | ollama | anthropic
    model: str
    temperature: float = 0.0
    max_tokens: int = 8000
    seed: int | None = 42            # honored where the provider supports it
    extra: dict = field(default_factory=dict)


GROQ_LLAMA33 = ModelConfig(provider="groq", model="llama-3.3-70b-versatile")

# Cumulative usage across all complete() calls in this process (cost accounting).
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def reset_usage() -> None:
    USAGE.update(calls=0, prompt_tokens=0, completion_tokens=0)


def _record_usage(prompt_tokens: int, completion_tokens: int) -> None:
    USAGE["calls"] += 1
    USAGE["prompt_tokens"] += int(prompt_tokens or 0)
    USAGE["completion_tokens"] += int(completion_tokens or 0)


def _require_key(name: str) -> str:
    key = os.environ.get(name)
    if not key:
        raise RuntimeError(
            f"{name} is not set. Add it to .env (see .env.example) to use this provider.")
    return key


def _post_json(url: str, headers: dict, payload: dict, timeout: int = 120) -> dict:
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code == 429:                       # rate limit -> let caller retry
        raise TimeoutError(f"rate-limited: {r.text[:200]}")
    r.raise_for_status()
    return r.json()


def complete(system: str, user: str, config: ModelConfig,
             max_retries: int = 3, backoff: float = 5.0) -> str:
    """One chat completion; retries transient network/rate-limit errors."""
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return _complete_once(system, user, config)
        except (requests.exceptions.RequestException, TimeoutError) as e:
            last = e
            if attempt < max_retries:
                wait = backoff * attempt
                print(f"[llm] {config.provider} error ({type(e).__name__}); "
                      f"retry {attempt}/{max_retries} in {wait:.0f}s")
                time.sleep(wait)
    raise RuntimeError(f"{config.provider}/{config.model}: exhausted retries: {last!r}")


def _complete_once(system: str, user: str, c: ModelConfig) -> str:
    if c.provider == "groq":
        key = _require_key("GROQ_API_KEY")
        payload = {
            "model": c.model,
            "temperature": c.temperature,
            "max_tokens": c.max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            **({"seed": c.seed} if c.seed is not None else {}),
            **c.extra,
        }
        data = _post_json("https://api.groq.com/openai/v1/chat/completions",
                          {"Authorization": f"Bearer {key}"}, payload)
        u = data.get("usage", {})
        _record_usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        return data["choices"][0]["message"]["content"]

    if c.provider == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        payload = {
            "model": c.model,
            "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"temperature": c.temperature,
                        **({"seed": c.seed} if c.seed is not None else {}),
                        **c.extra},
        }
        data = _post_json(f"{host}/api/chat", {}, payload, timeout=600)
        _record_usage(data.get("prompt_eval_count", 0), data.get("eval_count", 0))
        return data["message"]["content"]

    if c.provider == "anthropic":
        key = _require_key("ANTHROPIC_API_KEY")
        payload = {
            "model": c.model,
            "max_tokens": c.max_tokens,
            "temperature": c.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            **c.extra,
        }
        data = _post_json("https://api.anthropic.com/v1/messages",
                          {"x-api-key": key, "anthropic-version": "2023-06-01"},
                          payload)
        u = data.get("usage", {})
        _record_usage(u.get("input_tokens", 0), u.get("output_tokens", 0))
        return "".join(b["text"] for b in data["content"] if b["type"] == "text")

    raise ValueError(f"unknown provider '{c.provider}' (groq | ollama | anthropic)")
