"""Adapters for headless CLI benchmark targets."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

from ..models import ModelInfo
from .base import GenerationResult, Provider, ProviderError, TokenCallback


def _events(output: str):
    for line in output.splitlines():
        if line.strip():
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("expected an event object")
            yield event


def _claude_text(output: str) -> str:
    event = json.loads(output)
    if not isinstance(event, dict) or event.get("type") != "result":
        raise ValueError("expected a Claude result")
    if event.get("is_error") or event.get("subtype") != "success":
        raise ValueError("Claude reported an unsuccessful run")
    return event.get("result") or ""


def _codex_text(output: str) -> str:
    text = ""
    complete = False
    diagnostic = ""
    for event in _events(output):
        if event["type"] in ("error", "turn.failed"):
            diagnostic = event.get("message") or (event.get("error") or {}).get("message") or ""
            if event["type"] == "turn.failed":
                raise ValueError(diagnostic or "Codex reported an unsuccessful run")
        if event["type"] == "turn.completed":
            complete = True
        if event["type"] == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                text = item.get("text") or ""
    if not complete:
        raise ValueError(diagnostic or "Codex did not complete the turn")
    return text


def _opencode_text(output: str) -> str:
    parts = []
    complete = False
    for event in _events(output):
        if event["type"] == "error":
            error = event.get("error") or {}
            raise ValueError((error.get("data") or {}).get("message") or "OpenCode reported an unsuccessful run")
        if event["type"] == "text":
            parts.append((event.get("part") or {}).get("text") or "")
        if event["type"] == "step_finish":
            complete = True
    return "\n".join(parts) if complete else ""


@dataclass(frozen=True)
class HarnessCLI:
    executable: str
    args: Callable[[str], list[str]]
    parse: Callable[[str], str]


HARNESS_CLIS = {
    "claude": HarnessCLI("claude", lambda model: [
        "--print", "--output-format=json", f"--model={model}",
        "--bare", "--setting-sources=", "--tools=", "--strict-mcp-config",
        "--no-session-persistence",
    ], _claude_text),
    "codex": HarnessCLI("codex", lambda model: [
        "--ask-for-approval=never", "exec", "--json", f"--model={model}",
        "--ignore-user-config", "--ephemeral", "--sandbox=read-only",
        "--skip-git-repo-check", "-",
    ], _codex_text),
    "opencode": HarnessCLI("opencode", lambda model: [
        "run", "--pure", "--format=json", f"--model={model}",
    ], _opencode_text),
}


class HarnessCLIProvider(Provider):
    quality_only = True

    def __init__(self, backend: str, models: list[str], timeout: float):
        self.backend = backend
        self.name = f"{backend}-cli"
        self.spec = HARNESS_CLIS[backend]
        self.models = models
        self.timeout = timeout
        executable = shutil.which(self.spec.executable)
        if not executable:
            raise ProviderError(f"[{backend}]:MODEL requires {self.spec.executable!r} on PATH")
        self.executable = executable
        self._run(["--version"], timeout=10)

    def is_available(self) -> bool:
        try:
            self._run(["--version"], timeout=10)
        except ProviderError:
            return False
        return True

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name=model, provider=self.name) for model in self.models]

    def _run(self, args: list[str], *, timeout: float, prompt: Optional[str] = None) -> str:
        env = os.environ.copy()
        try:
            with tempfile.TemporaryDirectory(prefix=f"homebench-{self.backend}-") as cwd:
                if self.backend == "opencode":
                    env.pop("OPENCODE_CONFIG", None)
                    env.pop("OPENCODE_CONFIG_DIR", None)
                    env["XDG_CONFIG_HOME"] = cwd
                    env["OPENCODE_CONFIG_CONTENT"] = json.dumps({
                        "permission": {"*": "deny"}, "share": "disabled", "autoupdate": False,
                    })
                result = subprocess.run(
                    [self.executable, *args], input=prompt, stdin=None if prompt is not None else subprocess.DEVNULL,
                    cwd=cwd, env=env, timeout=timeout, capture_output=True,
                    text=True, errors="replace", check=False, shell=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(f"{self.backend} CLI timed out after {timeout:g}s") from exc
        except OSError as exc:
            raise ProviderError(f"Could not run {self.backend} CLI: {exc}") from exc
        if result.returncode:
            detail = (result.stderr.strip() or result.stdout.strip())[:1000]
            raise ProviderError(f"{self.backend} CLI exited {result.returncode}: {detail or 'no diagnostic output'}")
        return result.stdout

    def generate(
        self, model: str, prompt: str, *, max_tokens: int = 256,
        temperature: float = 0.0, seed: Optional[int] = None,
        on_token: TokenCallback = None, timeout: float = 300,
    ) -> GenerationResult:
        if self.backend == "opencode" and not all(model.partition("/")[::2]):
            raise ProviderError("OpenCode model IDs must use PROVIDER/MODEL")
        output = self._run(self.spec.args(model), timeout=min(timeout, self.timeout), prompt=prompt)
        try:
            text = self.spec.parse(output)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("no final answer")
        except (ValueError, KeyError, AttributeError, TypeError) as exc:
            raise ProviderError(f"{self.backend} CLI did not return a usable answer for {model!r}: {exc}") from exc
        if on_token is not None:
            on_token(text)
        return GenerationResult(text=text)
