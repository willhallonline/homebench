import json
import subprocess
from pathlib import Path

import pytest

from homebench.providers.base import ProviderError
from homebench.providers.harness_cli import HarnessCLIProvider


@pytest.mark.parametrize(("backend", "model", "events"), [
    ("claude", "sonnet", [{"type": "result", "subtype": "success", "result": "42"}]),
    ("codex", "codex-model", [
        {"type": "error", "message": "Reconnecting"},
        {"type": "item.completed", "item": {"type": "command_execution", "aggregated_output": "not the answer"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "first"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "42"}},
        {"type": "turn.completed"},
    ]),
    ("opencode", "openai/model:tag", [
        {"type": "tool_use", "part": {"text": "not the answer"}},
        {"type": "text", "part": {"text": "42"}},
        {"type": "step_finish"},
    ]),
])
def test_headless_harness_passes_prompt_via_stdin_and_extracts_answer(monkeypatch, backend, model, events):
    seen = []

    def run(argv, **kwargs):
        if argv[1:] == ["--version"]:
            return subprocess.CompletedProcess(argv, 0, "test version", "")
        seen.append(kwargs["cwd"])
        assert Path(kwargs["cwd"]).is_dir()
        assert kwargs["cwd"] != str(Path.cwd())
        assert kwargs["input"] == "--not-an-option\nsecond line"
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == 10
        assert f"--model={model}" in argv
        return subprocess.CompletedProcess(argv, 0, "\n".join(map(json.dumps, events)), "")

    monkeypatch.setattr("homebench.providers.harness_cli.shutil.which", lambda cmd: f"/path/{cmd}")
    monkeypatch.setattr("homebench.providers.harness_cli.subprocess.run", run)
    provider = HarnessCLIProvider(backend, [model], timeout=10)
    assert provider.generate(model, "--not-an-option\nsecond line").text == "42"
    assert not Path(seen[0]).exists()


def test_failed_codex_turn_does_not_grade_partial_text(monkeypatch):
    def run(argv, **kwargs):
        events = [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "42"}},
            {"type": "turn.failed", "error": {"message": "model unavailable"}},
        ]
        return subprocess.CompletedProcess(argv, 0, "\n".join(map(json.dumps, events)), "")

    monkeypatch.setattr("homebench.providers.harness_cli.shutil.which", lambda cmd: f"/path/{cmd}")
    monkeypatch.setattr("homebench.providers.harness_cli.subprocess.run", run)
    with pytest.raises(ProviderError, match="model unavailable"):
        HarnessCLIProvider("codex", ["m"], timeout=10).generate("m", "Answer")
