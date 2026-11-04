import asyncio

import pytest
from textual.widgets import DataTable, Static

from homebench.quality import LLMJudge, Task
from homebench.quality.graders import Grade
from homebench.runner import EV_PHASE, EV_TASK_DONE, EV_TASK_GENERATED, RunConfig, Runner
from homebench.tui.app import HomebenchApp
from tests.fakes import FakeProvider


@pytest.mark.parametrize("with_judge", [False, True])
def test_tui_runs_and_fills_leaderboard(with_judge):
    provider = FakeProvider()
    runner = Runner(
        provider, RunConfig(
            sample_rss=False, include_open=with_judge, quick=not with_judge,
        ),
        judge=LLMJudge(provider, "smart:8b") if with_judge else None,
    )
    models = provider.list_models()

    async def scenario():
        app = HomebenchApp(runner, models)
        async with app.run_test() as pilot:
            for _ in range(200):
                if app.result is not None:
                    break
                await pilot.pause(0.05)
            assert app.result is not None, "benchmark did not finish"
            table = app.query_one("#board", DataTable)
            assert table.row_count == 2
            assert len(app.result.reports) == 2
            assert app.query_one("#tasks", DataTable).row_count == app.total_tasks
            assert all(
                task.score is not None
                for quality in app.quality.values() for task in quality.tasks.values()
            )
            if with_judge:
                assert all(quality.judged for quality in app.quality.values())

    asyncio.run(scenario())


def test_tui_updates_pending_rows_in_place_and_shows_phase_progress():
    provider = FakeProvider()
    models = provider.list_models()
    runner = Runner(
        provider,
        RunConfig(
            sample_rss=False, warmup=False, run_speed=False, include_open=True,
            suite=[
                Task("open", "writing", "open prompt"),
                Task("auto", "custom", "auto prompt", lambda _: Grade(1, True)),
            ],
        ),
        judge=LLMJudge(provider, "smart:8b"),
    )

    class ManualApp(HomebenchApp):
        def run_benchmark(self):
            pass

    async def scenario():
        app = ManualApp(runner, models)
        async with app.run_test() as pilot:
            await pilot.pause()
            tasks = app.query_one("#tasks", DataTable)
            pending_keys = {}
            transitions = []

            def observe(event, **data):
                app._handle_event(event, data)
                if event == EV_TASK_GENERATED and data["task_id"] == "open":
                    assert tasks.get_row("open")[2:5] == ["LLM judge", "Awaiting judge", "-"]
                    pending_keys[data["model"]] = next(iter(tasks.rows))
                    transitions.append("pending")
                if event == EV_PHASE and data["phase"] == "judge":
                    assert app._status_label(data["model"]) == (
                        "2/2 Judge open-ended responses 0/1"
                    )
                    assert "0/1" in str(app.query_one("#status", Static).render())
                    assert tasks.get_row("auto")[2:5] == ["Auto-graded", "Pass", "100%"]
                    transitions.append("judge")
                if event == EV_TASK_DONE and data["grading"] == "judge":
                    assert tasks.row_count == 2
                    assert next(iter(tasks.rows)) is pending_keys[data["model"]]
                    assert tasks.get_row("open")[2:5] == ["LLM judge", "Pass", "100%"]
                    assert app._status_label(data["model"]) == (
                        "2/2 Judge open-ended responses 1/1"
                    )
                    assert len(app.quality[data["model"]].generated) == 2
                    transitions.append("scored")
                if event == EV_TASK_GENERATED and data["task_id"] == "auto":
                    assert app._status_label(data["model"]) == (
                        "1/2 Generate and auto-grade 2/2"
                    )

            runner.run(models, observer=observe)
            assert transitions == ["pending", "judge", "scored"] * 2
            assert app.result is not None
            assert app.query_one("#board", DataTable).row_count == 2
            assert "Benchmark complete" in str(app.query_one("#status", Static).render())

    asyncio.run(scenario())


def test_tui_speed_only_hides_task_table():
    provider = FakeProvider()
    runner = Runner(provider, RunConfig(sample_rss=False, run_quality=False))

    async def scenario():
        app = HomebenchApp(runner, provider.list_models())
        async with app.run_test() as pilot:
            for _ in range(200):
                if app.result is not None:
                    break
                await pilot.pause(0.05)
            assert app.result is not None
            assert not app.query_one("#tasks", DataTable).display
            assert not app.query_one("#task-heading", Static).renderable

    asyncio.run(scenario())
