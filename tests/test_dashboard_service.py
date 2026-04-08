"""
Purpose: Smoke tests for the read-only dashboard service built on top of benchmark report artifacts.
Input/Output: Generates a tiny report set in a temporary directory and verifies the service view models.
Important invariants: The dashboard must work from persisted files without mutating them or requiring live endpoints.
How to debug: Run `pytest tests/test_dashboard_service.py -q` and inspect the generated temp artifact set.
"""

from __future__ import annotations

from pathlib import Path

from llm_benchmark.config.models import BenchmarkConfig
from llm_benchmark.domain.result import RunResult, ScoreBreakdown
from llm_benchmark.domain.test_case import TestCaseDefinition
from llm_benchmark.dashboard.service import DashboardFilters, DashboardService
from llm_benchmark.reporting.builder import build_report_artifacts
from llm_benchmark.reporting.exporters import write_raw_results, write_report_artifacts

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _build_result() -> RunResult:
    return RunResult(
        benchmark_run_id="dash-run-1",
        run_started_at="2026-04-08T10:00:00Z",
        run_finished_at="2026-04-08T10:00:05Z",
        model_id="openai-reference",
        model_label="OpenAI Reference",
        provider="openai",
        endpoint="https://api.openai.com/v1",
        model_name="gpt-4.1-mini",
        test_case_id="quick-chat-triage",
        category="chat",
        repetition_index=1,
        prompt_hash="abc123",
        prompt_version="1.0",
        duration_ms=980.0,
        ttft_ms=110.0,
        input_tokens=45,
        output_tokens=95,
        tokens_per_second=96.9,
        http_status=200,
        success=True,
        timeout=False,
        retries=0,
        raw_response_text="Rollback the release first and inspect config drift as the likely cause category.",
        parsed_output_json=None,
        validation_passed=True,
        validation_errors=[],
        score_breakdown=ScoreBreakdown(
            quality_score=95.0,
            format_score=100.0,
            latency_score=100.0,
            stability_score=100.0,
            instruction_score=100.0,
            reproducibility_score=100.0,
            total_score=97.25,
            weights={
                "quality": 0.35,
                "format": 0.25,
                "latency": 0.20,
                "stability": 0.10,
                "instruction": 0.00,
                "reproducibility": 0.10,
            },
        ),
        score_total=97.25,
        error_type=None,
        error_message=None,
        metadata={
            "phase": "cold",
            "suites": ["quick_compare"],
            "tags": ["quick_compare", "chat"],
            "expected_format": "text",
        },
    )


def test_dashboard_service_builds_context_from_artifacts(tmp_path: Path) -> None:
    config = BenchmarkConfig.model_validate(
        {
            "models": [
                {
                    "id": "openai-reference",
                    "label": "OpenAI Reference",
                    "provider": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "model_name": "gpt-4.1-mini",
                }
            ]
        }
    )
    test_case = TestCaseDefinition.model_validate(
        {
            "test_case_id": "quick-chat-triage",
            "category": "chat",
            "title": "Quick operational triage answer",
            "description": "Short chat task.",
            "prompt": "Rollback first.",
            "tags": ["quick_compare", "chat"],
            "suites": ["quick_compare"],
        }
    )
    results = [_build_result()]
    artifacts = build_report_artifacts(
        results=results,
        benchmark_name="dashboard-test",
        benchmark_run_id="dash-run-1",
        suite="quick_compare",
        run_started_at="2026-04-08T10:00:00Z",
        run_finished_at="2026-04-08T10:00:05Z",
        environment_info={"python_version": "3.12.0"},
        config=config,
        test_cases=[test_case],
    )

    write_raw_results(results, tmp_path)
    write_report_artifacts(artifacts, tmp_path)

    service = DashboardService(results_dir=tmp_path, tests_dir=FIXTURES_DIR / "tests")
    context = service.build_dashboard_context(DashboardFilters(view="overview"))

    assert context["overview"]["has_results"] is True
    assert context["models"][0]["model_label"] == "OpenAI Reference"
    assert any(item["name"] == "final_report.json" for item in context["downloads"])
    assert service.health()["status"] == "ok"
