from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from rally_health.integrations import gpt4o_chat


def test_chat_provider_builds_openai_compatible_pinkyne_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({
                "answer": "HR is 60 bpm.", "cited_source_ids": ["obs-1"], "follow_up_question": None,
                "intent": "question", "observation_proposal": None,
            })}}]}

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, headers, json):
            captured.update({"url": url, "headers": headers, "payload": json})
            return FakeResponse()

    monkeypatch.setattr(gpt4o_chat.httpx, "AsyncClient", FakeClient)
    settings = SimpleNamespace(chat_api_key="test-token", chat_base_url="https://mock.pinkyne.local/v1",
                               chat_model="gpt-4o", chat_timeout_seconds=20)
    provider = gpt4o_chat.GPT4oChatProvider(settings)
    result = asyncio.run(provider.respond("What is my HR?", [{"role":"user", "content":"Earlier"}], {"metrics": []}))
    assert result["cited_source_ids"] == ["obs-1"]
    assert captured["url"] == "https://mock.pinkyne.local/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert captured["payload"]["model"] == "gpt-4o"
    assert captured["payload"]["response_format"]["json_schema"]["strict"] is True
    assert captured["payload"]["messages"][1] == {"role":"user", "content":"Earlier"}


def test_chat_provider_rejects_invalid_json_response(monkeypatch):
    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "not json"}}]}

    class FakeClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(gpt4o_chat.httpx, "AsyncClient", FakeClient)
    settings = SimpleNamespace(chat_api_key="test-token", chat_base_url="https://mock.local/v1",
                               chat_model="gpt-4o", chat_timeout_seconds=20)
    provider = gpt4o_chat.GPT4oChatProvider(settings)
    try:
        asyncio.run(provider.respond("Hi", [], {"metrics": []}))
    except RuntimeError as error:
        assert "invalid structured output" in str(error)
    else:
        raise AssertionError("invalid model JSON must be rejected")
