from io import StringIO

import pytest
from rich.console import Console

from homebench.plainui import PlainReporter, run_plain
from homebench.progress import QualityProgress
from homebench.providers.base import ProviderError
from homebench.quality import LLMJudge, Task
from homebench.quality.graders import Grade
from homebench.runner import (
    EV_PHASE,
    EV_TASK_DONE,
    EV_TASK_GENERATED,
    RunConfig,
    Runner,
)
from tests.fakes import FakeProvider


def _runner(provider=None):
    provider = provider or FakeProvider()
    suite = [
        Task("open_a", "writing", "first open prompt"),
        Task("auto", "custom", "auto prompt", lambda _: Grade(1.0, True)),
        Task("open_b", "writing", "second open prompt"),
    ]
    return Runner(
        provider,
        RunConfig(
            suite=suite, sample_rss=False, warmup=False, run_speed=False,
            include_open=True,
        ),
        judge=LLMJudge(provider, "smart:8b"),
    )


def _render(reporter):
    output = StringIO()
    Console(file=output, width=220, color_system=None).print(reporter.render())
    return output.getvalue()


def test_generated_events_precede_judging_and_final_events_are_unique():
    runner = _runner()
    events = []
    result = runner.run(
        [runner.provider.list_models()[0]],
        observer=lambda event, **data: events.append((event, data)),
    )
    generated = [data for event, data in events if event == EV_TASK_GENERATED]
    assert [data["task_id"] for data in generated] == ["open_a", "auto", "open_b"]
    assert [data["pending_judge"] for data in generated] == [True, False, True]
    judge_index = next(
        i for i, (event, data) in enumerate(events)
        if event == EV_PHASE and data["phase"] == "judge"
    )
    assert all(
        i < judge_index for i, (event, _) in enumerate(events)
        if event == EV_TASK_GENERATED
    )
    finals = [data for event, data in events if event == EV_TASK_DONE]
    assert [data["task_id"] for data in finals] == ["auto", "open_a", "open_b"]
    assert [data["grading"] for data in finals] == ["auto", "judge", "judge"]
    assert all(data["score"] == 1.0 and data["passed"] for data in finals)
    assert result.reports[0].quality_score == 100


def test_plain_reporter_distinguishes_generation_and_judging():
    runner = _runner()
    model = runner.provider.list_models()[0]
    reporter = PlainReporter([model], total_tasks=3)
    snapshots = []

    def observe(event, **data):
        reporter(event, **data)
        if event in (EV_PHASE, EV_TASK_GENERATED, EV_TASK_DONE):
            snapshots.append((event, data, _render(reporter)))

    runner.run([model], observer=observe)
    generation = next(
        text for event, data, text in snapshots
        if event == EV_TASK_GENERATED and data["task_id"] == "open_b"
    )
    assert "1/2 Generate and auto-grade 3/3" in generation
    assert generation.count("Awaiting judge") == 2
    assert "Auto-graded" in generation and "LLM judge" in generation
    assert reporter.quality[model.name].tasks["auto"].status == "Pass"
    judging = [
        text for event, data, text in snapshots
        if event == EV_TASK_DONE and data["grading"] == "judge"
    ]
    assert "2/2 Judge open-ended responses 1/2" in judging[0]
    assert judging[0].count("Awaiting judge") == 1
    assert "2/2 Judge open-ended responses 2/2" in judging[1]
    assert "Awaiting judge" not in judging[1]
    tasks = reporter.quality[model.name].tasks
    assert list(tasks) == ["open_a", "auto", "open_b"]
    assert all(task.score_label == "100%" for task in tasks.values())


@pytest.mark.parametrize("failure", ["generation", "judge", "unparseable"])
def test_failures_are_final_not_left_awaiting_judge(failure):
    class FailingProvider(FakeProvider):
        def generate(self, model, prompt, **kwargs):
            if failure == "generation" and prompt == "first open prompt":
                raise ProviderError("candidate unavailable")
            if "strict grader" in prompt:
                if failure == "judge":
                    raise ProviderError("judge unavailable")
                if failure == "unparseable":
                    return super().generate(model, "not a judge prompt", **kwargs)
            return super().generate(model, prompt, **kwargs)

    runner = _runner(FailingProvider())
    model = runner.provider.list_models()[0]
    reporter = PlainReporter([model], 3)
    events = []

    def observe(event, **data):
        reporter(event, **data)
        events.append((event, data))

    result = runner.run([model], observer=observe)
    text = _render(reporter)
    assert "Awaiting judge" not in text
    task = reporter.quality[model.name].tasks["open_a"]
    assert task.score == 0
    if failure == "generation":
        assert task.status == "Generation error"
        assert "generation error: candidate unavailable" in text
        phase = next(
            data for event, data in events
            if event == EV_PHASE and data["phase"] == "judge"
        )
        assert phase["total_tasks"] == 1
        assert reporter.quality[model.name].judged == {"open_b"}
    else:
        assert task.status == "Fail"
        expected = "judge error:" if failure == "judge" else "judge gave no score:"
        assert expected in text
        assert reporter.quality[model.name].judged == {"open_a", "open_b"}
    assert len(result.reports[0].task_results) == 3


def test_progress_does_not_double_count_final_scores_or_repeated_events():
    progress = QualityProgress(1, judge_tasks=1)
    generated = dict(
        task_id="open", category="open", grading="judge", pending_judge=True,
    )
    final = dict(
        task_id="open", category="open", grading="judge", score=0.5, passed=True,
    )
    for _ in range(2):
        progress.update(EV_TASK_GENERATED, generated)
        progress.update(EV_TASK_DONE, final)
    assert progress.label("quality") == "1/2 Generate and auto-grade 1/1"
    assert progress.label("judge") == "2/2 Judge open-ended responses 1/1"
    assert len(progress.tasks) == 1
    assert progress.tasks["open"].score_label == "50%"


def test_plain_auto_only_cached_and_speed_only_runs():
    provider = FakeProvider()
    model = provider.list_models()[0]
    runner = Runner(
        provider, RunConfig(sample_rss=False, warmup=False, run_speed=False),
    )
    for _ in range(2):
        reporter = PlainReporter([model], 8)
        result = runner.run([model], observer=reporter)
        progress = reporter.quality[model.name]
        assert progress.label("quality") == "Generate and auto-grade 8/8"
        assert not progress.judged
        assert all(task.grading_label == "Auto-graded" for task in progress.tasks.values())
        assert result.reports[0].quality_score == 100
    assert all(task.latency_s == 0 for task in result.reports[0].task_results)
    runner.config.run_quality = False
    output = StringIO()
    result = run_plain(runner, [model], Console(file=output, width=220))
    assert "Final leaderboard" in output.getvalue()
    assert "Quality tasks for" not in output.getvalue()
    assert result.reports[0].task_results == []


def test_plain_run_shows_final_judged_rows():
    runner = _runner()
    output = StringIO()
    result = run_plain(
        runner, runner.provider.list_models(),
        Console(file=output, width=220),
    )
    assert len(result.reports) == 2
    assert "Final leaderboard" in output.getvalue()
    assert "Awaiting judge" not in output.getvalue()
    assert "LLM judge" in output.getvalue()


@pytest.mark.parametrize("task_kind", ["open_only", "empty", "all_generation_errors"])
def test_edge_case_suites_have_exact_progress_and_no_pending_final_rows(task_kind):
    class FailingProvider(FakeProvider):
        def generate(self, model, prompt, **kwargs):
            raise ProviderError("candidate unavailable")

    runner = _runner(FailingProvider() if task_kind == "all_generation_errors" else None)
    runner.config.suite = [] if task_kind == "empty" else [
        task for task in runner.config.suite if task.grader is None
    ]
    model = runner.provider.list_models()[0]
    reporter = PlainReporter([model], len(runner.config.suite))
    phases = []

    def observe(event, **data):
        reporter(event, **data)
        if event == EV_PHASE:
            phases.append(data["phase"])

    result = runner.run([model], observer=observe)
    progress = reporter.quality[model.name]
    assert len(progress.generated) == len(runner.config.suite)
    assert len(progress.tasks) == len(result.reports[0].task_results)
    assert all(task.score is not None for task in progress.tasks.values())
    if task_kind == "open_only":
        assert phases == ["quality", "judge"]
        assert progress.label("judge") == "2/2 Judge open-ended responses 2/2"
    else:
        assert phases == ["quality"]
        assert not progress.judged
