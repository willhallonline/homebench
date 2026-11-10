import json
import os
import subprocess
from unittest.mock import patch

import httpx
import pytest

from homebench.copilot_bridge import (
    CopilotBridge,
    _assistant_text,
    check_copilot_cli,
)
from homebench.providers.base import ProviderError
from homebench.providers.copilot_cli import CopilotCLIProvider
from homebench.providers.openai_compat import OpenAICompatibleProvider


def test_assistant_text_joins_jsonl_deltas():
    output = "\n".join([
        json.dumps({"type": "user.message", "data": {"content": "ignored"}}),
        json.dumps({
            "type": "assistant.message_delta",
            "data": {"deltaContent": "Hello"},
        }),
        json.dumps({
            "type": "assistant.message_delta",
            "data": {"deltaContent": " world"},
        }),
        json.dumps({"type": "assistant.turn_end", "data": {}}),
    ])
    assert _assistant_text(output) == "Hello world"


def test_assistant_text_handles_reasoning_and_final_message_events():
    output = "\n".join([
        json.dumps({
            "type": "assistant.reasoning_delta",
            "data": {"deltaContent": "internal reasoning"},
        }),
        json.dumps({
            "type": "assistant.message_delta",
            "data": {"deltaContent": "Hello"},
        }),
        json.dumps({
            "type": "assistant.message_delta",
            "data": {"deltaContent": " there"},
        }),
        json.dumps({
            "type": "assistant.message",
            "data": {"content": "Hello there"},
        }),
    ])
    assert _assistant_text(output) == "Hello there"


def test_bridge_models_and_streamed_chat():
    result = subprocess.CompletedProcess(
        args=["copilot"], returncode=0,
        stdout=json.dumps({
            "type": "assistant.message_delta",
            "data": {"deltaContent": "42"},
        }) + "\n",
        stderr="",
    )
    with patch("homebench.copilot_bridge.subprocess.run", return_value=result) as run:
        with CopilotBridge(["model-a"], "copilot") as bridge:
            provider = CopilotCLIProvider(host=bridge.url)
            assert provider.is_available()
            assert [model.name for model in provider.list_models()] == ["model-a"]
            response = provider.generate("model-a", "Answer with 42.")

    assert response.text == "42"
    command = run.call_args.args[0]
    assert command[0] == "copilot"
    assert "--model=model-a" in command
    assert "--output-format=json" in command
    assert "--available-tools=" in command
    assert "--disable-builtin-mcps" in command
    session_id_index = command.index("--session-id") + 1
    assert command[session_id_index]
    assert run.call_args.kwargs["shell"] is False
    copilot_home = run.call_args.kwargs["env"]["COPILOT_HOME"]
    assert copilot_home != os.path.join(os.path.expanduser("~"), ".copilot")
    assert not os.path.exists(copilot_home)
    assert run.call_args.kwargs["cwd"] == os.path.join(os.path.expanduser("~"), ".homebench", "copilot")


def test_bridge_rejects_unlisted_models_without_invoking_cli():
    with patch("homebench.copilot_bridge.subprocess.run") as run:
        with CopilotBridge(["model-a"], "copilot") as bridge:
            response = httpx.post(
                f"{bridge.url}/v1/chat/completions",
                json={
                    "model": "model-b",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
    assert response.status_code == 400
    run.assert_not_called()


def test_bridge_reports_cli_timeout():
    with patch(
        "homebench.copilot_bridge.subprocess.run",
        side_effect=subprocess.TimeoutExpired("copilot", 1),
    ):
        with CopilotBridge(["model-a"], "copilot", timeout=1) as bridge:
            response = httpx.post(
                f"{bridge.url}/v1/chat/completions",
                json={
                    "model": "model-a",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
    assert response.status_code == 504
    assert "timed out" in response.json()["error"]["message"]


def test_provider_uses_bridge_error_body_for_http_failures():
    class _StreamingResponse:
        def __init__(self):
            self._body = b'{"error": {"message": "copilot CLI rejected model \'gpt-6-luna\'"}}'

        def read(self):
            return self._body

        @property
        def content(self):
            raise httpx.ResponseNotRead()

        def json(self):
            return json.loads(self._body.decode("utf-8"))

    response = _StreamingResponse()
    status_error = httpx.HTTPStatusError(
        "Server error '502 Bad Gateway' for url 'http://127.0.0.1:1/v1/chat/completions'",
        request=httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions"),
        response=response,
    )

    assert OpenAICompatibleProvider._http_error_message(status_error) == (
        "copilot CLI rejected model 'gpt-6-luna'"
    )

    with patch("homebench.providers.openai_compat.httpx.stream", side_effect=status_error):
        with pytest.raises(ProviderError, match="copilot CLI rejected model 'gpt-6-luna'"):
            CopilotCLIProvider(host="http://127.0.0.1:1").generate(
                "gpt-6-luna",
                "hello",
            )


def test_cli_preflight_reports_install_failure():
    result = subprocess.CompletedProcess(
        args=["copilot", "--version"], returncode=127,
        stdout="", stderr="Copilot CLI is not installed",
    )
    with patch("homebench.copilot_bridge.subprocess.run", return_value=result):
        try:
            check_copilot_cli("copilot")
        except RuntimeError as exc:
            assert str(exc) == "Copilot CLI is not installed"
        else:
            raise AssertionError("expected Copilot CLI preflight to fail")
