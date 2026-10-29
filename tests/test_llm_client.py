"""LLM client tests. No network: every request goes to an httpx.MockTransport."""

import json

import httpx
import pytest

from app.config import Settings
from testgen.llm_client import (
    LLMError,
    OllamaClient,
    OpenAIClient,
    create_client,
)


def mock_http(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def openai_body(text="def test_x(): assert True", prompt=12, completion=8):
    return {
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def test_openai_parses_text_and_usage():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=openai_body())

    client = OpenAIClient(api_key="sk-test", model="gpt-4o-mini", http=mock_http(handler))
    result = client.generate("write tests", system="You write pytest tests.")

    assert result.text == "def test_x(): assert True"
    assert result.prompt_tokens == 12 and result.completion_tokens == 8
    assert result.total_tokens == 20
    assert result.duration_seconds >= 0
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "You write pytest tests."}
    assert seen["body"]["messages"][1] == {"role": "user", "content": "write tests"}
    assert seen["body"]["temperature"] == 0.0


def test_openai_requires_api_key():
    with pytest.raises(ValueError):
        OpenAIClient(api_key="", model="gpt-4o-mini")


def test_ollama_parses_text_and_token_counts():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "llama3.1:8b",
                "message": {"role": "assistant", "content": "def test_y(): pass"},
                "prompt_eval_count": 30,
                "eval_count": 15,
            },
        )

    client = OllamaClient(model="llama3.1:8b", http=mock_http(handler))
    result = client.generate("write tests")

    assert result.text == "def test_y(): pass"
    assert (result.prompt_tokens, result.completion_tokens) == (30, 15)
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["body"]["stream"] is False
    assert seen["body"]["options"]["temperature"] == 0.0


def test_retries_on_rate_limit_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=openai_body())

    client = OpenAIClient(
        api_key="sk-test", model="m", backoff_seconds=0, http=mock_http(handler)
    )
    assert client.generate("x").text
    assert calls["n"] == 3


def test_gives_up_after_max_retries():
    def handler(request: httpx.Request):
        return httpx.Response(503)

    client = OpenAIClient(
        api_key="sk-test", model="m", max_retries=2, backoff_seconds=0, http=mock_http(handler)
    )
    with pytest.raises(LLMError):
        client.generate("x")


def test_client_error_is_not_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request):
        calls["n"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    client = OpenAIClient(api_key="sk-bad", model="m", backoff_seconds=0, http=mock_http(handler))
    with pytest.raises(LLMError):
        client.generate("x")
    assert calls["n"] == 1


def test_create_client_from_settings():
    settings = Settings(llm_api_key="sk-test", llm_api_model="gpt-4o-mini", ollama_model="qwen2.5-coder:7b")
    assert create_client("openai", settings).model == "gpt-4o-mini"
    assert create_client("ollama", settings).model == "qwen2.5-coder:7b"
    with pytest.raises(ValueError):
        create_client("gemini", settings)
