from homebench.quality import LLMJudge
from tests.fakes import FakeProvider


def test_judge_parses_score():
    judge = LLMJudge(FakeProvider(), "smart:8b")
    grade = judge.score("summarize why sky is blue", "Rayleigh scattering.", "ref")
    # FakeProvider returns "SCORE: 5 ..." for the judge prompt
    assert grade.passed
    assert grade.score == 1.0
    assert "5/5" in grade.detail


def test_runner_includes_open_tasks_with_judge():
    from homebench.runner import RunConfig, Runner

    provider = FakeProvider()
    judge = LLMJudge(provider, "smart:8b")
    runner = Runner(provider, RunConfig(sample_rss=False, include_open=True, quick=False),
                    judge=judge)
    result = runner.run([provider.list_models()[0]])
    r = result.reports[0]
    from homebench.quality import default_suite
    assert len(r.task_results) == len(default_suite(include_open=True))
    assert any(t.category == "open" for t in r.task_results)


def test_runner_generates_all_responses_before_judging():
    from homebench.quality.tasks import Task
    from homebench.runner import RunConfig, Runner

    class RecordingProvider(FakeProvider):
        def __init__(self):
            super().__init__()
            self.calls = []

        def generate(self, model, prompt, **kwargs):
            self.calls.append(model)
            return super().generate(model, prompt, **kwargs)

    provider = RecordingProvider()
    tasks = [
        Task("open_a", "open", "first candidate task", grader=None, reference="first"),
        Task("open_b", "open", "second candidate task", grader=None, reference="second"),
    ]
    runner = Runner(
        provider,
        RunConfig(
            sample_rss=False, warmup=False, run_speed=False, include_open=True,
            suite=tasks, unload_between=False,
        ),
        judge=LLMJudge(provider, "smart:8b"),
    )
    phases = []
    result = runner.run(
        [provider.list_models()[0]],
        observer=lambda event, **data: phases.append(data["phase"])
        if event == "phase" else None,
    )

    assert provider.calls == ["fast:1b", "fast:1b", "smart:8b", "smart:8b"]
    assert phases == ["quality", "judge"]
    assert [task.passed for task in result.reports[0].task_results] == [True, True]
