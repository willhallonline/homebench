from homebench.models import SpeedMetrics
from homebench.providers.ollama import OllamaProvider, _normalize_host


class _FakeStream:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self.lines)


def test_normalize_host():
    assert _normalize_host("localhost:11434") == "http://localhost:11434"
    assert _normalize_host("http://x:1/") == "http://x:1"
    assert _normalize_host("") == "http://localhost:11434"
    assert _normalize_host("https://remote:443") == "https://remote:443"


def test_fill_timings_computes_tps():
    speed = SpeedMetrics()
    OllamaProvider._fill_timings(speed, {
        "prompt_eval_count": 31,
        "eval_count": 100,
        "load_duration": 2_000_000_000,       # 2s
        "prompt_eval_duration": 500_000_000,  # 0.5s
        "eval_duration": 2_000_000_000,       # 2s
    })
    assert speed.output_tokens == 100
    assert speed.eval_s == 2.0
    assert speed.load_s == 2.0
    assert abs(speed.tokens_per_sec - 50.0) < 1e-9


def test_fill_timings_zero_eval_is_safe():
    speed = SpeedMetrics()
    OllamaProvider._fill_timings(speed, {"eval_count": 0, "eval_duration": 0})
    assert speed.tokens_per_sec == 0.0


def test_generate_for_judge_disables_thinking(monkeypatch):
    payload = {}

    def fake_stream(*args, **kwargs):
        payload.update(kwargs["json"])
        return _FakeStream(['{"response":"SCORE: 5 — correct","done":true}'])

    import homebench.providers.ollama as mod
    monkeypatch.setattr(mod.httpx, "stream", fake_stream)

    result = OllamaProvider().generate_for_judge("model", "prompt")

    assert payload["think"] is False
    assert result.text == "SCORE: 5 — correct"
