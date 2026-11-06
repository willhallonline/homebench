"""A short-lived OpenAI-compatible bridge to GitHub Copilot CLI.

The bridge is intended for quality evaluations. Copilot CLI controls sampling
and does not expose comparable token or memory metrics through this interface.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple


_MAX_REQUEST_BYTES = 2 * 1024 * 1024


class CopilotCLIError(RuntimeError):
    """Raised when Copilot CLI cannot provide a response."""


def _event_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("deltaContent", "content", "text"):
            text = value.get(key)
            if isinstance(text, str):
                return text
        for key in ("message", "data", "result"):
            text = _event_text(value.get(key))
            if text:
                return text
    if isinstance(value, list):
        return "".join(_event_text(item) for item in value)
    return ""


def _assistant_text(output: str) -> str:
    deltas: List[str] = []
    messages: List[str] = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type", "")).lower()
        if "assistant" in event_type and "delta" in event_type:
            text = _event_text(event.get("data", event))
            if text:
                deltas.append(text)
        elif "assistant" in event_type and "message" in event_type:
            text = _event_text(event.get("data", event))
            if text:
                messages.append(text)
        elif event_type == "result":
            text = _event_text(event.get("data", event))
            if text:
                messages.append(text)

    return "".join(deltas) if deltas else (messages[-1] if messages else "")


def _prompt_from_messages(messages) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")

    parts: List[Tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str):
            raise ValueError("each message must have a role")
        text = _event_text(content)
        if not text:
            raise ValueError(f"message for role {role!r} has no text content")
        parts.append((role, text))

    if len(parts) == 1 and parts[0][0] == "user":
        return parts[0][1]
    return "\n\n".join(f"{role.upper()}:\n{text}" for role, text in parts)


class _CopilotHandler(BaseHTTPRequestHandler):
    server: "CopilotBridge._Server"

    def log_message(self, _format: str, *args) -> None:
        return

    def _send_json(self, status: int, payload: Dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/v1/models":
            self._send_json(404, {"error": {"message": "not found"}})
            return
        self._send_json(200, {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "owned_by": "github-copilot"}
                for model in self.server.bridge.models
            ],
        })

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": {"message": "invalid content length"}})
            return
        if length <= 0 or length > _MAX_REQUEST_BYTES:
            self._send_json(413, {"error": {"message": "request body is empty or too large"}})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            model = payload.get("model")
            if model not in self.server.bridge.models:
                raise ValueError(f"unknown model {model!r}")
            prompt = _prompt_from_messages(payload.get("messages"))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._send_json(400, {"error": {"message": str(exc)}})
            return

        try:
            content = self.server.bridge.generate(model, prompt)
        except subprocess.TimeoutExpired:
            self._send_json(504, {"error": {"message": "Copilot CLI timed out"}})
            return
        except (CopilotCLIError, OSError) as exc:
            self._send_json(502, {"error": {"message": str(exc)}})
            return

        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": content},
                             "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            return

        self._send_json(200, {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
        })


class CopilotBridge:
    """Serve the minimal OpenAI chat API expected by homebench."""

    class _Server(ThreadingHTTPServer):
        daemon_threads = True

    def __init__(
        self,
        models: List[str],
        executable: str,
        timeout: float = 300.0,
    ):
        self.models = list(dict.fromkeys(model.strip() for model in models if model.strip()))
        if not self.models:
            raise ValueError("at least one Copilot model is required")
        self.executable = executable
        self.timeout = timeout
        self._server: Optional[CopilotBridge._Server] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("Copilot bridge is not running")
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> "CopilotBridge":
        self._server = self._Server(("127.0.0.1", 0), _CopilotHandler)
        self._server.bridge = self
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def generate(self, model: str, prompt: str) -> str:
        command = [
            self.executable,
            "-p",
            prompt,
            f"--model={model}",
            "--output-format=json",
            "--no-ask-user",
            "--available-tools=",
        ]
        env = os.environ.copy()
        env["COPILOT_AUTO_UPDATE"] = "false"
        with tempfile.TemporaryDirectory(prefix="homebench-copilot-") as cwd:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.timeout,
                shell=False,
                check=False,
            )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            if not detail:
                detail = f"Copilot CLI exited with status {result.returncode}"
            raise CopilotCLIError(detail[:2000])
        content = _assistant_text(result.stdout)
        if not content:
            raise CopilotCLIError(
                "Copilot CLI returned no assistant text; check that it supports "
                "--output-format=json"
            )
        return content


def find_copilot_cli() -> Optional[str]:
    """Return the Copilot CLI executable path, if installed."""
    return shutil.which("copilot")


def check_copilot_cli(executable: str) -> str:
    """Verify the executable starts and return its version output."""
    env = os.environ.copy()
    env["COPILOT_AUTO_UPDATE"] = "false"
    try:
        result = subprocess.run(
            [executable, "--version"],
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10.0,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CopilotCLIError(f"could not start Copilot CLI: {exc}") from exc
    version = (result.stdout or result.stderr).strip()
    if result.returncode:
        raise CopilotCLIError(
            version[:2000] or f"Copilot CLI exited with status {result.returncode}"
        )
    return version
