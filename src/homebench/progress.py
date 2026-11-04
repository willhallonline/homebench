"""Shared live quality-task state for the terminal renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from .runner import EV_PHASE, EV_TASK_DONE, EV_TASK_GENERATED


@dataclass
class LiveTask:
    task_id: str
    category: str
    grading: str
    status: str
    score: Optional[float] = None
    detail: str = ""

    @property
    def score_label(self) -> str:
        return "-" if self.score is None else f"{100 * self.score:.0f}%"

    @property
    def grading_label(self) -> str:
        return "Auto-graded" if self.grading == "auto" else "LLM judge"


@dataclass
class QualityProgress:
    total_tasks: int
    judge_tasks: int = 0
    generated: Set[str] = field(default_factory=set)
    judged: Set[str] = field(default_factory=set)
    tasks: Dict[str, LiveTask] = field(default_factory=dict)

    def update(self, event: str, data: dict) -> None:
        if event == EV_PHASE:
            if data["phase"] == "quality":
                self.total_tasks = data["total_tasks"]
                self.judge_tasks = data["judge_tasks"]
            elif data["phase"] == "judge":
                self.judge_tasks = data["total_tasks"]
        elif event == EV_TASK_GENERATED:
            self.generated.add(data["task_id"])
            self.tasks[data["task_id"]] = LiveTask(
                data["task_id"], data["category"], data["grading"],
                "Awaiting judge" if data["pending_judge"] else "Generated",
            )
        elif event == EV_TASK_DONE:
            error = data.get("generation_error", False)
            if data["grading"] == "judge" and not error:
                self.judged.add(data["task_id"])
            self.tasks[data["task_id"]] = LiveTask(
                data["task_id"], data["category"], data["grading"],
                "Generation error" if error else ("Pass" if data["passed"] else "Fail"),
                data["score"], data.get("detail", ""),
            )

    def label(self, phase: str) -> str:
        if phase == "quality":
            prefix = "1/2 " if self.judge_tasks else ""
            return (f"{prefix}Generate and auto-grade "
                    f"{len(self.generated)}/{self.total_tasks}")
        if phase == "judge":
            return (f"2/2 Judge open-ended responses "
                    f"{len(self.judged)}/{self.judge_tasks}")
        return phase
