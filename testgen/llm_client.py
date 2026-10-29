"""LLM clients for the test layer.

Both clients expose the same interface so an experiment can switch between a
closed API model and a local open-source model by changing one setting:

    client = create_client("openai")   # or "ollama"
    result = client.generate("Write pytest tests for ...")
    result.text, result.prompt_tokens, result.completion_tokens, result.duration_seconds

Token counts and duration are returned with every call because they feed the
cost metrics stored in the ``test_runs`` table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings, get_settings

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when a provider call fails after all retries."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    duration_seconds: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    model: str

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResult: ...


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return messages


def _post_with_retry(
    http: httpx.Client,
    url: str,
    payload: dict,
    headers: dict[str, str],
    max_retries: int,
    backoff_seconds: float,
) -> dict:
    """POST and return JSON; retry on rate limits, server errors and timeouts."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = http.post(url, json=payload, headers=headers)
        except httpx.TransportError as exc:  # timeouts, connection errors
            last_error = exc
        else:
            if response.status_code not in RETRYABLE_STATUS:
                if response.is_error:
                    raise LLMError(f"{url} returned {response.status_code}: {response.text[:300]}")
                return response.json()
            last_error = LLMError(f"{url} returned {response.status_code}")
        if attempt < max_retries:
            time.sleep(backoff_seconds * (2**attempt))
    raise LLMError(f"giving up after {max_retries + 1} attempts: {last_error}")


class OpenAIClient:
    """Client for OpenAI or any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        http: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("LLM_API_KEY is empty; set it in .env")
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._http = http or httpx.Client(timeout=timeout_seconds)

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": _messages(prompt, system),
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        started = time.perf_counter()
        data = _post_with_retry(
            self._http, self._url, payload, self._headers, self._max_retries, self._backoff
        )
        usage = data.get("usage") or {}
        return LLMResult(
            text=data["choices"][0]["message"]["content"] or "",
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            duration_seconds=time.perf_counter() - started,
        )


class OllamaClient:
    """Client for a local Ollama server (open-source models)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        http: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self._url = base_url.rstrip("/") + "/api/chat"
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._http = http or httpx.Client(timeout=timeout_seconds)

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": _messages(prompt, system),
            "stream": False,
            "options": {"temperature": self._temperature, "num_predict": self._max_tokens},
        }
        started = time.perf_counter()
        data = _post_with_retry(self._http, self._url, payload, {}, self._max_retries, self._backoff)
        return LLMResult(
            text=data["message"]["content"] or "",
            model=data.get("model", self.model),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            duration_seconds=time.perf_counter() - started,
        )


def create_client(provider: str, settings: Settings | None = None) -> LLMClient:
    """Build a client from settings. ``provider`` is "openai" or "ollama"."""
    s = settings or get_settings()
    common = {
        "temperature": s.llm_temperature,
        "max_tokens": s.llm_max_tokens,
        "timeout_seconds": s.llm_timeout_seconds,
        "max_retries": s.llm_max_retries,
    }
    if provider == "openai":
        return OpenAIClient(
            api_key=s.llm_api_key, model=s.llm_api_model, base_url=s.llm_api_base_url, **common
        )
    if provider == "ollama":
        return OllamaClient(model=s.ollama_model, base_url=s.ollama_base_url, **common)
    raise ValueError(f"unknown provider: {provider!r} (expected 'openai' or 'ollama')")
