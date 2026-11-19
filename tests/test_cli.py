import json
import subprocess
from io import StringIO

import pytest
from rich.console import Console

from homebench.cli import _inject_default_command, build_parser, cmd_run, main
from homebench.copilot_bridge import CopilotBridge
from homebench.models import ModelInfo, SpeedMetrics
from homebench.providers.base import GenerationResult, Provider


def test_inject_default_command():
    assert _inject_default_command([]) == ["run"]
    assert _inject_default_command(["--no-tui"]) == ["run", "--no-tui"]
    assert _inject_default_command(["-m", "x"]) == ["run", "-m", "x"]
    # explicit subcommands are left untouched
    assert _inject_default_command(["list"]) == ["list"]
    assert _inject_default_command(["copilot", "-m", "model"]) == [
        "copilot", "-m", "model"
    ]
    assert _inject_default_command(["run", "--limit", "2"]) == ["run", "--limit", "2"]
    # global help/version bypass the default
    assert _inject_default_command(["--version"]) == ["--version"]


def test_provider_flag_reaches_list_command():
    parser = build_parser()
    args = parser.parse_args(["list", "--provider", "lmstudio"])
    assert args.command == "list"
    assert args.provider == "lmstudio"


def test_run_flags_parse():
    parser = build_parser()
    args = parser.parse_args(_inject_default_command(
        ["--provider", "lmstudio", "--no-tui", "--limit", "3", "--judge", "m"]
    ))
    assert args.command == "run"
    assert args.provider == "lmstudio"
    assert args.no_tui is True
    assert args.limit == 3
    assert args.judge == "m"


def test_copilot_command_defaults_to_quality_only():
    args = build_parser().parse_args(["copilot", "-m", "model-a"])
    assert args.command == "copilot"
    assert args.models == "model-a"
    assert args.no_speed is True
    assert args.no_warmup is True
    assert args.no_rss is True


def test_hermes_provider_and_model_flags_parse():
    args = build_parser().parse_args([
        "hermes", "-m", "qwen3.5:9b", "--provider", "ollama"
    ])
    assert args.models == "qwen3.5:9b"
    assert args.provider == "ollama"


def test_task_flags_accept_multiple_and_repeated_packs():
    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--tasks", "first.json", "second.json",
        "--tasks", "third.json",
        "--add-tasks", "fourth.json", "fifth.json",
        "--add-tasks", "sixth.json",
    ])
    assert args.tasks == ["first.json", "second.json", "third.json"]
    assert args.add_tasks == ["fourth.json", "fifth.json", "sixth.json"]


class BenchmarkProvider(Provider):
    name = "test-provider"

    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def list_models(self):
        return [ModelInfo(name="benchmark-model", provider=self.name)]

    def generate(self, model, prompt, **kwargs):
        self.calls.append(model)
        return GenerationResult(text="SCORE: 5 Matches" if model == "qwen3:8b" else "answer")


@pytest.mark.parametrize("selector", [
    "copilot:judge:tag", "hermes:judge:tag", "[copilot]:judge:tag", "[hermes]:judge:tag", "qwen3:8b",
])
def test_open_ended_judge_routes_separately_and_scores(selector, tmp_path, monkeypatch):
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(json.dumps({"tasks": [
        {"id": "test", "prompt": "Describe this", "reference": "reference answer"},
    ]}))
    output = tmp_path / "result.json"
    provider = BenchmarkProvider()
    commands = []
    bridges = []
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("homebench.copilot_bridge.find_copilot_cli", lambda: "fake-copilot")
    monkeypatch.setattr("homebench.providers.hermes_agent.find_hermes", lambda: "fake-hermes")

    class TrackedBridge(CopilotBridge):
        def __enter__(self):
            bridges.append(self)
            return super().__enter__()

    def fake_run(command, **kwargs):
        if command[0] not in ("fake-copilot", "fake-hermes"):
            return real_run(command, **kwargs)
        if command[1:] == ["--version"]:
            text = "test version"
        elif command[1:] == ["chat", "--help"]:
            text = "--safe-mode --toolsets --query --quiet"
        else:
            commands.append(command)
            text = "SCORE: 5 Matches"
            if command[0] == "fake-copilot":
                text = json.dumps({"type": "assistant.message", "data": {"content": text}})
        return subprocess.CompletedProcess(command, 0, stdout=text, stderr="")

    real_run = subprocess.run
    monkeypatch.setattr("homebench.copilot_bridge.CopilotBridge", TrackedBridge)
    monkeypatch.setattr(subprocess, "run", fake_run)
    args = build_parser().parse_args([
        "run", "--judge", selector, "--tasks", str(pack_path), "--json", str(output),
        "--no-tui", "--no-speed", "--no-warmup", "--no-rss", "--no-cache", "--no-save",
    ])
    assert cmd_run(args, Console(file=StringIO()), provider_override=provider) == 0
    result = json.loads(output.read_text())
    assert result["config"]["judge_model"] == selector
    assert len(result["reports"]) == 1
    assert result["reports"][0]["task_results"][0]["score"] == 1.0
    assert provider.calls == (
        ["benchmark-model", selector] if selector == "qwen3:8b" else ["benchmark-model"]
    )
    if selector == "qwen3:8b":
        assert not commands
        return
    command, = commands
    assert "--model=judge:tag" in command
    if selector.startswith(("hermes:", "[hermes]:")):
        assert "--provider=copilot" in command
        assert "--toolsets=" in command and "--safe-mode" in command
        prompt = command[command.index("--query") + 1]
    else:
        bridge, = bridges
        with pytest.raises(RuntimeError, match="not running"):
            _ = bridge.url
        prompt = command[command.index("-p") + 1]
    assert "REFERENCE (an example of a good answer):\nreference answer" in prompt
    assert "RESPONSE:\nanswer" in prompt


@pytest.mark.parametrize(("selector", "error"), [
    ("copilot:", "requires a model name"),
    ("copilot:judge", "Copilot CLI was not found"),
    ("hermes:judge", "Hermes CLI on PATH"),
])
def test_judge_startup_errors_prevent_benchmark(selector, error, monkeypatch):
    provider = BenchmarkProvider()
    monkeypatch.setattr("homebench.copilot_bridge.find_copilot_cli", lambda: None)
    monkeypatch.setattr("homebench.providers.hermes_agent.find_hermes", lambda: None)
    args = build_parser().parse_args(["run", "--judge", selector])
    output = StringIO()
    assert cmd_run(args, Console(file=output), provider_override=provider) == 1
    assert error in output.getvalue()
    assert not provider.calls


def test_mixed_model_run_uses_one_suite_and_only_measures_local_speed(tmp_path, monkeypatch, capsys):
    class LocalProvider(BenchmarkProvider):
        def list_models(self):
            return [ModelInfo(name="qwen:9b", provider=self.name)]

        def generate(self, model, prompt, **kwargs):
            self.calls.append((model, prompt))
            return GenerationResult(text="42", speed=SpeedMetrics(tokens_per_sec=20))

    provider = LocalProvider()
    monkeypatch.setattr("homebench.cli._resolve_provider", lambda args, console: provider)
    monkeypatch.setattr("homebench.copilot_bridge.find_copilot_cli", lambda: "fake-copilot")
    monkeypatch.setattr("homebench.providers.hermes_agent.find_hermes", lambda: "fake-hermes")
    monkeypatch.setenv("HOME", str(tmp_path))
    commands = []
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        if command[0] not in ("fake-copilot", "fake-hermes"):
            return real_run(command, **kwargs)
        if command[1:] == ["--version"]:
            text = "test version"
        else:
            commands.append(command)
            text = "0" if command[0] == "fake-hermes" else json.dumps(
                {"type": "assistant.message", "data": {"content": "42"}}
            )
        return subprocess.CompletedProcess(command, 0, stdout=text, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(json.dumps({"tasks": [{
        "id": "test", "prompt": "Answer the test task",
        "grader": {"type": "exact_number", "value": 42},
    }]}))
    output = tmp_path / "result.json"
    assert main([
        "-m", "qwen:9b,[hermes]:gpt-6-luna,[copilot]:gpt-6-luna", "--tasks", str(pack_path),
        "--no-tui", "--no-rss", "--no-save", "--json", str(output),
    ]) == 0
    result = json.loads(output.read_text())
    assert result["provider"] == "mixed"
    reports = result["reports"]
    assert [r["model"]["name"] for r in reports] == [
        "qwen:9b", "[hermes]:gpt-6-luna", "[copilot]:gpt-6-luna",
    ]
    assert [r["quality_score"] for r in reports] == [100.0, 0.0, 100.0]
    assert [r["speed"]["tokens_per_sec"] for r in reports] == [20, 0, 0]
    assert [(r["model"]["provider"], r["model"]["backend_model"]) for r in reports[1:]] == [
        ("hermes-agent", "gpt-6-luna"), ("copilot-cli", "gpt-6-luna"),
    ]
    assert [model for model, _ in provider.calls] == ["qwen:9b"] * 3  # warmup, speed, quality
    assert provider.calls[-1][1] == "Answer the test task"
    hermes, copilot = commands
    assert hermes[1:] == [
        "-z", "Answer the test task", "-m", "gpt-6-luna", "--provider", "copilot",
    ]
    assert "--model=gpt-6-luna" in copilot
    assert copilot[copilot.index("-p") + 1] == "Answer the test task"
    terminal = " ".join(capsys.readouterr().out.split())
    assert "[hermes]:gpt-6-luna" in terminal and "[copilot]:gpt-6-luna" in terminal


def test_qualified_only_run_does_not_discover_local_models(monkeypatch, tmp_path):
    def resolve(*args):
        pytest.fail("No local provider should be resolved")

    class CLIProvider(BenchmarkProvider):
        name = "hermes-agent"
        quality_only = True

    provider = CLIProvider()
    monkeypatch.setattr("homebench.cli._resolve_provider", resolve)
    monkeypatch.setattr("homebench.cli._open_hermes_provider", lambda *args: provider)
    args = build_parser().parse_args([
        "run", "-m", "[hermes]:model:tag", "--no-tui", "--no-save", "--no-cache",
    ])
    assert cmd_run(args, Console(file=StringIO())) == 0
    assert provider.calls and set(provider.calls) == {"model:tag"}


@pytest.mark.parametrize(("models", "error"), [
    ("[hermes]:", "invalid model selector"),
    ("[unknown]:model", "invalid model selector"),
    ("missing,[copilot]:model", "no model matching 'missing'"),
])
def test_mixed_model_selection_rejects_bad_targets(models, error, monkeypatch):
    monkeypatch.setattr("homebench.cli._resolve_provider", lambda *args: BenchmarkProvider())
    output = StringIO()
    args = build_parser().parse_args(["run", "-m", models])
    assert cmd_run(args, Console(file=output)) == 1
    assert error in output.getvalue()


def test_first_wave_harnesses_are_benchmark_only(monkeypatch):
    backend_names = []

    def adapter(backend, models, timeout):
        backend_names.append((backend, models))
        provider = BenchmarkProvider()
        provider.name = f"{backend}-cli"
        provider.quality_only = True
        return provider

    monkeypatch.setattr("homebench.cli.HarnessCLIProvider", adapter)
    args = build_parser().parse_args([
        "run", "-m", "[claude]:sonnet,[codex]:model,[opencode]:openai/model",
        "--judge", "[codex]:model",
    ])
    output = StringIO()
    assert cmd_run(args, Console(file=output)) == 1
    assert backend_names == [("claude", ["sonnet"]), ("codex", ["model"]), ("opencode", ["openai/model"])]
    assert "[codex] supports benchmarking but is not available as a judge" in output.getvalue()


@pytest.mark.parametrize("flags", [
    ["--judge", ""], ["--judge", "local,[copilot]:judge"], ["--judge", "local", "--judge", "another"],
])
def test_run_requires_exactly_one_value_when_judge_is_set(flags):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["run", *flags])
    assert exc.value.code == 2
