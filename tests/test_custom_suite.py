import json

from homebench.quality import Task
from homebench.quality.graders import contains_any
from homebench.quality import LLMJudge, load_pack
from homebench.runner import RunConfig, Runner, resolve_suite
from tests.fakes import FakeProvider


def _suite():
    return [
        Task("c1", "custom", "prompt one", contains_any(["a"])),
        Task("o1", "open", "prompt two", grader=None),
    ]


def test_resolve_suite_drops_open_without_judge():
    cfg = RunConfig(suite=_suite())
    assert [t.id for t in resolve_suite(cfg, judge_present=False)] == ["c1"]


def test_resolve_suite_keeps_open_with_judge():
    cfg = RunConfig(suite=_suite(), include_open=True)
    assert [t.id for t in resolve_suite(cfg, judge_present=True)] == ["c1", "o1"]


def test_resolve_suite_defaults_to_quick_builtin():
    from homebench.quality import default_suite

    cfg = RunConfig()  # quick=True by default
    assert cfg.quick is True
    assert len(resolve_suite(cfg, judge_present=False)) == len(default_suite(quick=True))


def test_resolve_suite_full_when_quick_off():
    from homebench.quality import default_suite

    cfg = RunConfig(quick=False)
    assert len(resolve_suite(cfg, judge_present=False)) == len(default_suite())


def test_runner_honors_custom_suite():
    provider = FakeProvider()
    runner = Runner(provider, RunConfig(sample_rss=False, suite=_suite()))
    result = runner.run([provider.list_models()[0]])
    # only the one deterministic task runs (open dropped, no judge)
    assert [t.task_id for t in result.reports[0].task_results] == ["c1"]


def test_pack_grader_uses_answer_after_think_marker(tmp_path):
    path = tmp_path / "sample_tasks" / "pack.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "tasks": [{
            "id": "answer",
            "prompt": "Give the answer.",
            "grader": {"type": "exact_number", "value": 42},
        }],
    }))
    task = load_pack(str(path))[0]

    class ReasoningProvider(FakeProvider):
        response = "Reasoning mentions 17.</think>\n\n42"

        def _answer_for(self, prompt):
            return self.response

    provider = ReasoningProvider()
    runner = Runner(
        provider,
        RunConfig(
            sample_rss=False, warmup=False, run_speed=False,
            unload_between=False, suite=[task],
        ),
    )
    result = runner.run([provider.list_models()[0]])
    task_result = result.reports[0].task_results[0]

    assert task_result.passed
    assert task_result.response == provider.response


def test_pack_judge_uses_answer_after_think_marker(tmp_path):
    path = tmp_path / "sample_tasks" / "pack.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "tasks": [{
            "id": "summary",
            "category": "open",
            "prompt": "Summarize.",
            "reference": "A concise summary.",
        }],
    }))
    task = load_pack(str(path))[0]

    class RecordingProvider(FakeProvider):
        response = "Reasoning should not be judged.</think>\n\nA concise summary."
        judge_prompt = ""

        def _answer_for(self, prompt):
            if "strict grader" in prompt:
                self.judge_prompt = prompt
                return "SCORE: 5 correct"
            return self.response

    provider = RecordingProvider()
    runner = Runner(
        provider,
        RunConfig(
            sample_rss=False, warmup=False, run_speed=False,
            unload_between=False, include_open=True, suite=[task],
        ),
        judge=LLMJudge(provider, "smart:8b"),
    )
    result = runner.run([provider.list_models()[0]])

    assert result.reports[0].task_results[0].passed
    assert "A concise summary." in provider.judge_prompt
    assert "Reasoning should not be judged." not in provider.judge_prompt
