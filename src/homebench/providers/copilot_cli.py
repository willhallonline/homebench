"""Homebench provider backed by the local Copilot CLI bridge."""

from __future__ import annotations

from .openai_compat import OpenAICompatibleProvider


class CopilotCLIProvider(OpenAICompatibleProvider):
    name = "copilot-cli"
    quality_only = True
    host_env = None
    api_key_env = None
