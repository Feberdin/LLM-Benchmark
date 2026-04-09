"""
Purpose: Regression tests for scoring, reproducibility updates and final report generation.
Input/Output: Builds small in-memory result sets to keep the tests deterministic and fast.
Important invariants: Weighted totals must stay stable and the final report schema should validate.
How to debug: Run `pytest tests/test_scoring_and_reporting.py -q` and inspect the generated in-memory report payload.
"""

from __future__ import annotations

from llm_benchmark.config.models import BenchmarkConfig, ScoreWeights
from llm_benchmark.domain.result import RunResult, ScoreBreakdown
from llm_benchmark.domain.test_case import TestCaseDefinition
from llm_benchmark.reporting.builder import build_report_artifacts
from llm_benchmark.reporting.exporters import write_raw_results, write_report_artifacts
from llm_benchmark.runner.scoring import ScoreCalculator


def _build_result(*, repetition_index: int, text: str, validation_passed: bool = True) -> RunResult:
    return RunResult(
        benchmark_run_id="run-1",
        run_started_at="2026-04-08T10:00:00Z",
        run_finished_at="2026-04-08T10:00:05Z",
        model_id="mistral-local",
        model_label="Mistral Small 3.2 (Local)",
        provider="ollama",
        endpoint="http://ollama:11434/v1",
        model_name="mistral-small3.2",
        test_case_id="chat-cap-theorem",
        category="chat",
        repetition_index=repetition_index,
        prompt_hash="abc123",
        prompt_version="1.0",
        duration_ms=1200.0 + repetition_index,
        ttft_ms=None,
        input_tokens=40,
        output_tokens=120,
        tokens_per_second=100.0,
        http_status=200,
        success=True,
        timeout=False,
        retries=0,
        raw_response_text=text,
        parsed_output_json=None,
        validation_passed=validation_passed,
        validation_errors=[],
        score_breakdown=ScoreBreakdown(
            quality_score=90.0,
            format_score=100.0,
            latency_score=100.0,
            stability_score=100.0,
            instruction_score=100.0,
            reproducibility_score=100.0,
            total_score=96.5,
            weights={
                "quality": 0.35,
                "format": 0.25,
                "latency": 0.20,
                "stability": 0.10,
                "instruction": 0.00,
                "reproducibility": 0.10,
            },
        ),
        score_total=96.5,
        error_type=None,
        error_message=None,
        metadata={"phase": "cold" if repetition_index == 1 else "warm", "suites": ["core"]},
    )


def test_reproducibility_score_is_recomputed_across_repetitions() -> None:
    calculator = ScoreCalculator(default_weights=ScoreWeights(), latency_target_ms=5000)
    first = _build_result(repetition_index=1, text="CAP theorem balances consistency and availability.")
    second = _build_result(repetition_index=2, text="CAP theorem balances consistency and availability under partitions.")

    calculator.apply_reproducibility_scores([first, second])

    assert first.score_breakdown.reproducibility_score < 100.0
    assert second.score_total == second.score_breakdown.total_score


def test_report_builder_generates_expected_sections() -> None:
    config = BenchmarkConfig.model_validate(
        {
            "models": [
                {
                    "id": "mistral-local",
                    "label": "Mistral Small 3.2 (Local)",
                    "provider": "ollama",
                    "base_url": "http://ollama:11434/v1",
                    "model_name": "mistral-small3.2"
                }
            ]
        }
    )
    test_case = TestCaseDefinition.model_validate(
        {
            "test_case_id": "chat-cap-theorem",
            "category": "chat",
            "title": "Explain CAP theorem tradeoffs",
            "description": "Simple report generation test case.",
            "prompt": "Explain CAP theorem.",
            "suites": ["core"]
        }
    )
    results = [
        _build_result(repetition_index=1, text="CAP theorem balances consistency and availability."),
        _build_result(repetition_index=2, text="CAP theorem balances consistency and availability under partitions.")
    ]

    artifacts = build_report_artifacts(
        results=results,
        benchmark_name="unit-test-benchmark",
        benchmark_run_id="run-1",
        suite="core",
        run_started_at="2026-04-08T10:00:00Z",
        run_finished_at="2026-04-08T10:00:05Z",
        environment_info={"python_version": "3.12.0"},
        config=config,
        test_cases=[test_case],
    )

    assert "benchmark_info" in artifacts.final_report
    assert "rankings" in artifacts.final_report
    assert "analysis_input_json" in artifacts.final_report["raw_artifacts"]
    assert artifacts.analysis_input["purpose"] == "secondary_llm_analysis_input"
    assert artifacts.analysis_input["benchmark_metadata"]["suite"] == "core"
    assert artifacts.final_report["test_suites"][0]["suite_name"] == "core"
    assert "repo_recommendations" in artifacts.final_report
    assert "best_model_for_paperless_kiplus" in artifacts.final_report
    assert "repo_recommendations" in artifacts.analysis_input
    assert "best_model_for_secondbrain" in artifacts.analysis_input
    assert "structured_output_summary" in artifacts.analysis_input
    assert "LLM Benchmark Report" in artifacts.markdown_report
    assert "<html" in artifacts.html_report.lower()


def test_exporters_write_history_snapshots(tmp_path) -> None:
    config = BenchmarkConfig.model_validate(
        {
            "models": [
                {
                    "id": "mistral-local",
                    "label": "Mistral Small 3.2 (Local)",
                    "provider": "ollama",
                    "base_url": "http://ollama:11434/v1",
                    "model_name": "mistral-small3.2",
                }
            ]
        }
    )
    test_case = TestCaseDefinition.model_validate(
        {
            "test_case_id": "chat-cap-theorem",
            "category": "chat",
            "title": "Explain CAP theorem tradeoffs",
            "description": "Simple report generation test case.",
            "prompt": "Explain CAP theorem.",
            "suites": ["core"],
        }
    )
    results = [_build_result(repetition_index=1, text="CAP theorem balances consistency and availability.")]
    artifacts = build_report_artifacts(
        results=results,
        benchmark_name="unit-test-benchmark",
        benchmark_run_id="run-1",
        suite="core",
        run_started_at="2026-04-08T10:00:00Z",
        run_finished_at="2026-04-08T10:00:05Z",
        environment_info={"python_version": "3.12.0"},
        config=config,
        test_cases=[test_case],
    )

    write_raw_results(results, tmp_path)
    write_report_artifacts(artifacts, tmp_path)

    history_dir = tmp_path / "history" / "run-1"
    assert (history_dir / "raw_runs.jsonl").exists()
    assert (history_dir / "raw_runs.csv").exists()
    assert (history_dir / "final_report.json").exists()
    assert (history_dir / "analysis_input.json").exists()
