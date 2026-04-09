"""
Purpose: Reusable benchmark execution helper shared by CLI commands and the dashboard run manager.
Input/Output: Loads config and tests, executes the orchestrator, then writes raw and aggregate report artifacts.
Important invariants: Missing tests fail with an actionable message, and the same execution path is reused across UI and CLI.
How to debug: If dashboard-triggered and CLI-triggered runs diverge, compare inputs and emitted progress events here first.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from llm_benchmark.config.loader import filter_test_cases_by_suite, load_config, load_test_cases
from llm_benchmark.config.models import BenchmarkConfig
from llm_benchmark.reporting.builder import ReportArtifacts, build_report_artifacts
from llm_benchmark.reporting.exporters import write_raw_results, write_report_artifacts
from llm_benchmark.runner.orchestrator import BenchmarkOrchestrator, BenchmarkRunSummary

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class RunExecutionArtifacts:
    """Bundle benchmark summary plus generated report artifacts and resolved paths."""

    run_summary: BenchmarkRunSummary
    artifacts: ReportArtifacts
    config: BenchmarkConfig
    tests_dir: Path
    results_dir: Path


def execute_benchmark_run(
    *,
    config_path: Path,
    tests_dir: Path,
    results_dir: Path,
    suite: str | None,
    progress_callback: ProgressCallback | None = None,
) -> RunExecutionArtifacts:
    """
    Execute a benchmark run end-to-end and persist all outputs.

    Why this exists:
    The CLI and dashboard should use the exact same execution path so operators do not have to
    reason about two different benchmark engines or report writers.
    """

    _emit(progress_callback, event="loading_config", stage="preflight", config_path=str(config_path))
    benchmark_config = load_config(config_path)

    _emit(
        progress_callback,
        event="discovering_tests",
        stage="preflight",
        tests_dir=str(tests_dir),
        suite=suite or "all",
    )
    selected_test_cases = filter_test_cases_by_suite(load_test_cases(tests_dir), suite)
    if not selected_test_cases:
        raise ValueError(
            f"No test cases selected in '{tests_dir}' for suite '{suite or 'all'}'. "
            "If you mounted /app/tests from Unraid, copy the fixture tests into that host directory first."
        )

    _emit(
        progress_callback,
        event="test_discovery_completed",
        stage="preflight",
        selected_test_case_count=len(selected_test_cases),
        selected_model_count=len(benchmark_config.enabled_models()),
    )

    orchestrator = BenchmarkOrchestrator(benchmark_config, progress_callback=progress_callback)
    run_summary = asyncio.run(orchestrator.run(test_cases=selected_test_cases, suite=suite))

    _emit(
        progress_callback,
        event="writing_raw_results",
        stage="writing_reports",
        results_dir=str(results_dir),
    )
    write_raw_results(run_summary.results, results_dir)

    _emit(
        progress_callback,
        event="building_reports",
        stage="writing_reports",
        results_dir=str(results_dir),
    )
    artifacts = build_report_artifacts(
        results=run_summary.results,
        benchmark_name=run_summary.benchmark_name,
        benchmark_run_id=run_summary.benchmark_run_id,
        suite=run_summary.suite,
        run_started_at=run_summary.run_started_at,
        run_finished_at=run_summary.run_finished_at,
        environment_info=run_summary.environment_info,
        config=benchmark_config,
        test_cases=run_summary.selected_test_cases,
    )
    write_report_artifacts(artifacts, results_dir)

    _emit(
        progress_callback,
        event="reports_written",
        stage="completed",
        benchmark_run_id=run_summary.benchmark_run_id,
        results_dir=str(results_dir),
    )

    return RunExecutionArtifacts(
        run_summary=run_summary,
        artifacts=artifacts,
        config=benchmark_config,
        tests_dir=tests_dir,
        results_dir=results_dir,
    )


def _emit(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback is None:
        return
    progress_callback(payload)
