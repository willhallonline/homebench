"""Homebench provider backed by the local Hermes CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import List, Optional

from ..models import MemoryMetrics, ModelInfo, SpeedMetrics
from .base import GenerationResult, Provider, ProviderError, TokenCallback


def _final_response(output: str) -> str:
    clean_output = output.strip()
    if "🎯 FINAL RESPONSE:" not in clean_output:
        return clean_output

    marker = "🎯 FINAL RESPONSE:"
    response = clean_output.split(marker, 1)[1]
    response = response.split("👋 Agent execution completed!", 1)[0]
    response = "\n".join(
        line for line in response.strip().splitlines()
        if set(line.strip()) != {"-"}
    ).strip()
    if response:
        return response

    for line in reversed(output.splitlines()):
        if line.startswith("🤖 Assistant:"):
            response = line.split(":", 1)[1].strip()
            if response:
                return response
    return ""


class HermesAgentProvider(Provider):
    """Run one-shot Hermes queries for quality evaluation."""

    name = "hermes-agent"

    def __init__(
        self,
        models: Optional[List[str]] = None,
        executable: str = "hermes",
        timeout: float = 300.0,
    ):
        self.models = list(dict.fromkeys(model.strip() for model in (models or []) if model.strip()))
        self.executable = executable
        self.timeout = timeout

    @property
    def process_hint(self) -> str:
        return "hermes"

    def is_available(self) -> bool:
        try:
            result = subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=10.0,
                shell=False,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def list_models(self) -> List[ModelInfo]:
        return [ModelInfo(name=model, provider=self.name) for model in sorted(self.models)]

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        on_token: TokenCallback = None,
        timeout: float = 300.0,
    ) -> GenerationResult:
        command = [
            self.executable,
            "-z",
            prompt,
            "-m",
            model,
            "--provider",
            "copilot",
        ]
        start = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=min(timeout, self.timeout),
                shell=False,
                check=False,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(f"Hermes timed out for {model!r}") from exc
        except OSError as exc:
            raise ProviderError(f"Could not start Hermes: {exc}") from exc

        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise ProviderError(
                f"Hermes failed for {model!r}: "
                f"{detail[:2000] or f'exited with status {result.returncode}'}"
            )
        content = _final_response(result.stdout)
        if not content:
            raise ProviderError(
                "Hermes returned no final response; check that -z output is supported"
            )
        if (
            "The requested model is not supported" in content
            or "is not accessible via the /chat/completions endpoint" in content
            or "rejected this request as malformed" in content
            or "Your request was not processed" in content
        ):
            raise ProviderError(
                f"Hermes could not access model {model!r}: {content}"
            )
        if on_token is not None:
            on_token(content)

        speed = SpeedMetrics(total_s=time.perf_counter() - start)
        return GenerationResult(text=content, speed=speed)


def find_hermes() -> Optional[str]:
    """Return the full Hermes CLI executable path, if installed."""
    return shutil.which("hermes")


def check_hermes(executable: str) -> str:
    """Verify the full Hermes CLI starts and return its version output."""
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10.0,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProviderError(f"could not start Hermes: {exc}") from exc
    version = (result.stdout or result.stderr).strip()
    if result.returncode:
        raise ProviderError(
            version[:2000] or f"Hermes exited with status {result.returncode}"
        )
    return version


# Backward-compatible Python names for callers importing the provider helpers.
find_hermes_agent = find_hermes
check_hermes_agent = check_hermes
