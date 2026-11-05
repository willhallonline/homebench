"""The benchmark orchestrator.

Runs, per model: an optional warmup, a repeated speed probe (tokens/sec +
time-to-first-token), a memory reading, then the quality suite. Progress is
reported through a lightweight observer callback so both the TUI and the
plain-CLI renderer can subscribe to the same events.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .metrics.memory import RSSSampler
from .models import (
    BenchmarkResult,
    MemoryMetrics,
    ModelInfo,
    ModelReport,
    SpeedMetrics,
    TaskResult,
)
from .providers.base import Provider, ProviderError
from .quality import LLMJudge, Task, default_suite

# ---- observer events -------------------------------------------------------
EV_RUN_START = "run_start"
EV_MODEL_START = "model_start"
EV_PHASE = "phase"            # data: model, phase, note
EV_TASK_GENERATED = "task_generated"  # data: model, task_id, category, pending_judge
EV_TASK_DONE = "task_done"    # data: model, task_id, category, score, passed, grading
EV_MODEL_DONE = "model_done"  # data: report
EV_RUN_DONE = "run_done"      # data: result

Observer = Optional[Callable[..., None]]


def _emit(observer: Observer, event: str, **data) -> None:
    if observer is not None:
        observer(event, **data)


def _response_for_grading(task: Task, response: str) -> str:
    if task.strip_think:
        return response.rsplit("</think>", 1)[-1]
    return response


DEFAULT_SPEED_PROMPT = (
    "Write three detailed paragraphs explaining how a computer's CPU executes "
    "a program, from source code to running instructions."
)


@dataclass
class RunConfig:
    max_tokens: int = 256
    speed_max_tokens: int = 200
    temperature: float = 0.0
    seed: Optional[int] = 42
    speed_prompt: str = DEFAULT_SPEED_PROMPT
    repeat: int = 1                 # speed-probe repetitions (best is kept)
    warmup: bool = True
    unload_between: bool = True
    sample_rss: bool = True
    timeout: float = 300.0
    include_open: bool = False
    judge_model: Optional[str] = None
    run_quality: bool = True
    run_speed: bool = True
    # Custom task suite (from user packs); None => the built-in suite.
    suite: Optional[List[Task]] = None
    quick: bool = True              # default to the fast curated subset
    use_cache: bool = True          # reuse cached deterministic responses
    refresh_cache: bool = False     # ignore existing cache, overwrite it

    def to_dict(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "speed_max_tokens": self.speed_max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "repeat": self.repeat,
            "warmup": self.warmup,
            "unload_between": self.unload_between,
            "sample_rss": self.sample_rss,
            "include_open": self.include_open,
            "judge_model": self.judge_model,
            "run_quality": self.run_quality,
            "run_speed": self.run_speed,
            "quick": self.quick,
        }


def resolve_suite(config: "RunConfig", judge_present: bool) -> List[Task]:
    """The effective task list for a run.

    Uses the config's custom suite if provided, else the built-in suite
    (the quick subset unless ``config.quick`` is False). Open-ended
    (grader-less) tasks are dropped unless a judge is available.
    """
    if config.suite is not None:
        base = config.suite
    else:
        base = default_suite(include_open=True, quick=config.quick)
    if not (config.include_open and judge_present):
        return [t for t in base if t.grader is not None]
    return list(base)


class Runner:
    def __init__(
        self,
        provider: Provider,
        config: Optional[RunConfig] = None,
        judge: Optional[LLMJudge] = None,
    ):
        self.provider = provider
        self.config = config or RunConfig()
        self.judge = judge
        self.cache = None
        if self.config.use_cache:
            from .cache import ResponseCache
            self.cache = ResponseCache(enabled=True, refresh=self.config.refresh_cache)

    # ------------------------------------------------------------------
    def run(self, models: List[ModelInfo], observer: Observer = None) -> BenchmarkResult:
        result = BenchmarkResult(
            provider=self.provider.name,
            config=self.config.to_dict(),
            environment=self._capture_environment(),
        )
        _emit(observer, EV_RUN_START, models=[m.name for m in models])
        for model in models:
            _emit(observer, EV_MODEL_START, model=model.name)
            report = self._run_model(model, observer)
            result.reports.append(report)
            _emit(observer, EV_MODEL_DONE, report=report)
        if self.cache is not None:
            self.cache.save()
        result.finished_at = time.time()
        _emit(observer, EV_RUN_DONE, result=result)
        return result

    @staticmethod
    def _capture_environment() -> dict:
        """Snapshot the host + homebench version for reproducibility."""
        from . import __version__
        try:
            from .hardware import capture
            hw = capture().to_dict()
        except Exception:
            hw = {}
        return {"homebench_version": __version__, "hardware": hw}

    # ------------------------------------------------------------------
    def _run_model(self, model: ModelInfo, observer: Observer) -> ModelReport:
        report = ModelReport(model=model)
        cfg = self.config
        # Sample RSS across the WHOLE per-model run (load + speed + every quality
        # generation) so the reported peak reflects sustained use, not a single
        # call. Baseline is captured before the model loads.
        sampler = RSSSampler(name_contains=self.provider.process_hint) if cfg.sample_rss else None
        if sampler is not None:
            sampler.__enter__()
        try:
            if cfg.warmup:
                _emit(observer, EV_PHASE, model=model.name, phase="warmup")
                self.provider.warmup(model.name, timeout=cfg.timeout)

            if cfg.run_speed:
                _emit(observer, EV_PHASE, model=model.name, phase="speed")
                report.speed, report.memory = self._measure_speed(model.name)

            if cfg.run_quality:
                report.task_results = self._run_quality(model, observer)

            if cfg.unload_between:
                self.provider.unload(model.name)
        except ProviderError as exc:
            report.error = str(exc)
            _emit(observer, EV_PHASE, model=model.name, phase="error", note=str(exc))
        finally:
            if sampler is not None:
                sampler.__exit__()
                report.memory.rss_peak_bytes = max(report.memory.rss_peak_bytes,
                                                   sampler.delta_bytes)
        return report

    # ------------------------------------------------------------------
    def _measure_speed(self, model: str):
        cfg = self.config
        runs: List[SpeedMetrics] = []
        mem = MemoryMetrics()
        for _ in range(max(1, cfg.repeat)):
            gen = self.provider.generate(
                model,
                cfg.speed_prompt,
                max_tokens=cfg.speed_max_tokens,
                temperature=cfg.temperature,
                seed=cfg.seed,
                timeout=cfg.timeout,
            )
            runs.append(gen.speed)

        # Keep the best (highest tokens/sec) run as the representative sample;
        # report median TTFT to smooth out first-call noise.
        best = max(runs, key=lambda s: s.tokens_per_sec)
        best.ttft_s = statistics.median(r.ttft_s for r in runs)

        # Resident memory: the provider's view (RSS peak is sampled per-model in
        # _run_model, spanning the whole run).
        pmem = self.provider.memory(model)
        mem.size_bytes = pmem.size_bytes
        mem.vram_bytes = pmem.vram_bytes
        return best, mem

    # ------------------------------------------------------------------
    def _run_quality(self, model: ModelInfo, observer: Observer) -> List[TaskResult]:
        suite = resolve_suite(self.config, self.judge is not None)
        results: List[TaskResult] = []
        pending_judgement = []
        _emit(
            observer, EV_PHASE, model=model.name, phase="quality",
            total_tasks=len(suite),
            judge_tasks=sum(task.grader is None for task in suite),
        )
        for task in suite:
            tr = self._run_task(model, task, defer_judge=self.judge is not None)
            results.append(tr)
            pending_judge = tr.detail == "pending judge"
            _emit(
                observer, EV_TASK_GENERATED, model=model.name, task_id=task.id,
                category=task.category, pending_judge=pending_judge,
                grading="auto" if task.grader is not None else "judge",
            )
            if pending_judge:
                pending_judgement.append((task, tr))
                continue
            _emit(
                observer, EV_TASK_DONE, model=model.name, task_id=task.id,
                category=task.category, score=tr.score, passed=tr.passed,
                grading="auto" if task.grader is not None else "judge",
                generation_error=tr.detail.startswith("generation error:"),
                detail=tr.detail,
            )

        if pending_judgement:
            _emit(
                observer, EV_PHASE, model=model.name, phase="judge",
                total_tasks=len(pending_judgement),
                note=f"scoring {len(pending_judgement)} open-ended tasks",
            )
            for task, tr in pending_judgement:
                grade = self.judge.score(
                    task.prompt, _response_for_grading(task, tr.response),
                    task.reference,
                )
                tr.score = grade.score
                tr.passed = grade.passed
                tr.detail = grade.detail
                _emit(
                    observer, EV_TASK_DONE, model=model.name, task_id=task.id,
                    category=task.category, score=tr.score, passed=tr.passed,
                    grading="judge", detail=tr.detail,
                )
        return results

    def _cached_response(self, model: ModelInfo, task: Task):
        """Return (response, output_tokens) from cache, or None. Judge tasks and
        non-deterministic (temperature>0) generations are never cached."""
        cfg = self.config
        if self.cache is None or task.grader is None or cfg.temperature != 0:
            return None
        key = self.cache.key(model.cache_id, task.id, task.prompt,
                             task.max_tokens, cfg.temperature, cfg.seed)
        hit = self.cache.get(key)
        if hit is None:
            return None
        return hit["response"], int(hit.get("output_tokens", 0) or 0)

    def _run_task(self, model: ModelInfo, task: Task,
                  defer_judge: bool = False) -> TaskResult:
        cfg = self.config
        cached = self._cached_response(model, task)
        if cached is not None:
            response, out_tokens = cached
            latency = 0.0
        else:
            start = time.perf_counter()
            try:
                gen = self.provider.generate(
                    model.name, task.prompt,
                    max_tokens=task.max_tokens,
                    temperature=cfg.temperature,
                    seed=cfg.seed,
                    timeout=cfg.timeout,
                )
            except ProviderError as exc:
                return TaskResult(
                    task.id, task.category, 0.0, False,
                    latency_s=time.perf_counter() - start,
                    detail=f"generation error: {exc}",
                )
            latency = time.perf_counter() - start
            response = gen.text
            out_tokens = gen.speed.output_tokens
            if self.cache is not None and task.grader is not None and cfg.temperature == 0:
                key = self.cache.key(model.cache_id, task.id, task.prompt,
                                     task.max_tokens, cfg.temperature, cfg.seed)
                self.cache.set(key, response, out_tokens)

        grade_response = _response_for_grading(task, response)
        if task.grader is not None:
            grade = task.grader(grade_response)
        elif self.judge is not None and not defer_judge:
            grade = self.judge.score(task.prompt, grade_response, task.reference)
        elif self.judge is not None:
            grade = None
        else:
            grade = None  # open-ended task with no judge available

        if grade is None:
            return TaskResult(
                task.id, task.category, 0.0, False, latency_s=latency,
                output_tokens=out_tokens, response=response,
                detail=("pending judge" if defer_judge and self.judge is not None
                        else "skipped (no grader/judge)"),
            )
        return TaskResult(
            task.id, task.category, grade.score, grade.passed,
            latency_s=latency, output_tokens=out_tokens,
            response=response, detail=grade.detail,
        )
