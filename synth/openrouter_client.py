"""Minimal OpenRouter chat client (OpenAI-compatible) for synthetic-data generation.

OpenRouter exposes Anthropic models under ids like ``anthropic/claude-opus-4.8`` via an
OpenAI-style ``/chat/completions`` endpoint. Auth: ``OPENROUTER_API_KEY``.

Only depends on ``requests`` so it runs on the CPU box. Retries with exponential backoff
on 429/5xx and transient network errors.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-opus-4.8"   # configurable; any OpenRouter Opus id works


class OpenRouterError(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
                 timeout: int = 120, max_retries: int = 5):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is not set. `export OPENROUTER_API_KEY=...` first.")

    def chat(self, system: str, user: str, temperature: float = 1.0,
             max_tokens: int = 4096) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # OpenRouter attribution headers (optional, recommended)
            "HTTP-Referer": "https://github.com/wadebalsamo/autoresearch",
            "X-Title": "MindfulBERT fine-tuning workshop",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        backoff = 2.0
        last = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(API_URL, headers=headers, json=body, timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json()
                    return data["choices"][0]["message"]["content"]
                if r.status_code in (429, 500, 502, 503, 504):
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                else:
                    raise OpenRouterError(f"HTTP {r.status_code}: {r.text[:400]}")
            except (requests.RequestException, KeyError, ValueError) as e:
                last = str(e)
            time.sleep(backoff)
            backoff *= 2
        raise OpenRouterError(f"OpenRouter failed after {self.max_retries} retries: {last}")


def list_opus_models() -> List[str]:
    """Best-effort listing of available Opus ids (no key needed)."""
    try:
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
        return sorted(m["id"] for m in r.json().get("data", []) if "opus" in m["id"].lower())
    except Exception:
        return []
