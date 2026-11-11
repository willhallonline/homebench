import subprocess
from unittest.mock import patch

import pytest

from homebench.models import ModelInfo
from homebench.providers.base import ProviderError
from homebench.providers.hermes_agent import HermesAgentProvider, _final_response


def test_final_response_extracts_summary():
    output = """\
🤖 Assistant: intermediate
🎯 FINAL RESPONSE:
------------------------------
The final answer.

👋 Agent execution completed!
"""
    assert _final_response(output) == "The final answer."


def test_hermes_provider_runs_one_shot_query():
    result = subprocess.CompletedProcess(
        args=["hermes"],
        returncode=0,
        stdout="42\n",
        stderr="",
    )
    with patch(
        "homebench.providers.hermes_agent.subprocess.run", return_value=result
    ) as run:
        provider = HermesAgentProvider(["gpt-5.6-luna"], executable="hermes")
        assert provider.is_available()
        assert provider.list_models() == [
            ModelInfo(name="gpt-5.6-luna", provider="hermes-agent")
        ]
        response = provider.generate("gpt-5.6-luna", "Answer with 42.")

    assert response.text == "42"
    command = run.call_args.args[0]
    assert command[:3] == ["hermes", "-z", "Answer with 42."]
    assert "-m" in command
    assert command[command.index("-m") + 1] == "gpt-5.6-luna"
    assert command[command.index("--provider") + 1] == "copilot"


def test_hermes_provider_reports_cli_failure():
    result = subprocess.CompletedProcess(
        args=["hermes"],
        returncode=1,
        stdout="",
        stderr="HTTP 400: The requested model is not supported",
    )
    with patch(
        "homebench.providers.hermes_agent.subprocess.run", return_value=result
    ):
        with pytest.raises(ProviderError, match="requested model is not supported"):
            HermesAgentProvider(["auto"]).generate("auto", "hello")


def test_hermes_provider_reports_model_error_in_success_transcript():
    result = subprocess.CompletedProcess(
        args=["hermes"],
        returncode=0,
        stdout=(
            "🎯 FINAL RESPONSE:\n"
            "OpenRouter rejected this request as malformed; "
            "the model didn't answer.\n"
            "👋 Agent execution completed!\n"
        ),
        stderr="",
    )
    with patch(
        "homebench.providers.hermes_agent.subprocess.run", return_value=result
    ):
        with pytest.raises(ProviderError, match="could not access model"):
            HermesAgentProvider(["gpt-6.0-luna"]).generate(
                "gpt-6.0-luna", "hello"
            )
