"""
Purpose: Aggregate raw benchmark runs into rankings, summaries and machine-readable final report structures.
Input/Output: Consumes `RunResult` records and returns CSV-ready frames plus Markdown, HTML and JSON report content.
Important invariants: Warmup runs stay visible in raw outputs but are excluded from the primary benchmark ranking.
How to debug: If a ranking or recommendation looks wrong, inspect the intermediate summary dataframes in this module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from importlib import resources
from typing import Any

import jinja2
import jsonschema
import orjson
import pandas as pd

from llm_benchmark.config.models import BenchmarkConfig
from llm_benchmark.domain.result import RunResult
from llm_benchmark.domain.test_case import TestCaseDefinition
from llm_benchmark.utils import isoformat_utc

REPO_SUITE_LABELS = {
    "secondbrain": "SecondBrain",
    "voice_gateway": "secondbrain-voice-gateway",
    "paperless_kiplus": "Paperless-KIplus",
}

REPO_SUITE_FOCUS_AREAS = {
    "secondbrain": ["RAG / knowledge queries", "Citations", "Security obedience", "Context extraction"],
    "voice_gateway": ["Routing", "Voice answers", "Backend selection", "Home Assistant safety"],
    "paperless_kiplus": ["Document classification", "JSON/YAML replacement", "Tax enrichment", "Review flags"],
}


@dataclass(slots=True)
class ReportArtifacts:
    """Container for everything that gets written into the results directory."""

    raw_runs: pd.DataFrame
    summary_by_model: pd.DataFrame
    summary_by_category: pd.DataFrame
    final_report: dict[str, Any]
    analysis_input: dict[str, Any]
    markdown_report: str
    html_report: str


def build_report_artifacts(
    *,
    results: list[RunResult],
    benchmark_name: str,
    benchmark_run_id: str,
    suite: str | None,
    run_started_at: str,
    run_finished_at: str,
    environment_info: dict[str, Any],
    config: BenchmarkConfig | None = None,
    test_cases: list[TestCaseDefinition] | None = None,
) -> ReportArtifacts:
    """Build report artifacts from raw results and validate the final JSON schema."""

    raw_runs = _results_to_dataframe(results)
    measured_runs = raw_runs[raw_runs["phase"] != "warmup"].copy() if not raw_runs.empty else raw_runs.copy()

    summary_by_model = _summarize_by_model(measured_runs)
    summary_by_category = _summarize_by_category(measured_runs)
    summary_by_test_case = _summarize_by_test_case(measured_runs)
    rankings = _build_rankings(summary_by_model)
    strengths, weaknesses = _build_strengths_and_weaknesses(summary_by_category)
    anomalies = _detect_anomalies(summary_by_model, summary_by_category)
    failed_runs = _extract_failed_runs(measured_runs)
    recommendations = _build_recommendations(summary_by_model, summary_by_category, rankings)
    suite_records = _build_suite_records(test_cases, results)
    model_records = _build_model_records(config, summary_by_model, results)
    repo_recommendations = _build_repo_recommendations(
        _enrich_runs_with_test_metadata(measured_runs, _build_test_metadata_lookup(test_cases, results))
    )

    final_report = {
        "schema_version": "1.0",
        "benchmark_info": {
            "benchmark_run_id": benchmark_run_id,
            "benchmark_name": benchmark_name,
            "suite": suite or "all",
            "run_started_at": run_started_at,
            "run_finished_at": run_finished_at,
            "generated_at": isoformat_utc(),
            "total_records": int(len(raw_runs)),
            "measured_records": int(len(measured_runs)),
            "warmup_records": int((raw_runs["phase"] == "warmup").sum()) if not raw_runs.empty else 0,
        },
        "environment_info": environment_info,
        "models": model_records,
        "test_suites": suite_records,
        "aggregate_scores": {
            "overall_average_total_score": round(
                float(measured_runs["score_total"].mean()) if not measured_runs.empty else 0.0, 2
            ),
            "by_model": _frame_to_records(summary_by_model),
            "by_category": _frame_to_records(summary_by_category),
            "by_test_case": _frame_to_records(summary_by_test_case),
        },
        "rankings": rankings,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "anomalies": anomalies,
        "failed_runs": failed_runs,
        "recommendations": recommendations,
        "repo_recommendations": repo_recommendations,
        "best_model_for_secondbrain": repo_recommendations["secondbrain"]["best_model"],
        "best_model_for_voice_gateway": repo_recommendations["voice_gateway"]["best_model"],
        "best_model_for_paperless_kiplus": repo_recommendations["paperless_kiplus"]["best_model"],
        "fairness_notes": [
            "All models receive the same prompt text, prompt hash and declared max token budget for a given test case.",
            "Cold and warm phases are tracked separately so cache effects are not hidden in the raw data.",
            "Streaming-based TTFT is optional and only populated when the backend supports comparable stream events.",
            "External provider-side prompt caching or hidden request shaping can still influence results and is surfaced as a limitation.",
            "Missing token usage metrics do not fail the run; affected throughput fields remain null instead of guessed.",
        ],
        "raw_artifacts": {
            "raw_runs_jsonl": "results/raw_runs.jsonl",
            "raw_runs_csv": "results/raw_runs.csv",
            "summary_by_model_csv": "results/summary_by_model.csv",
            "summary_by_category_csv": "results/summary_by_category.csv",
            "final_report_json": "results/final_report.json",
            "analysis_input_json": "results/analysis_input.json",
            "final_report_md": "results/final_report.md",
            "final_report_html": "results/final_report.html",
        },
    }

    _validate_final_report(final_report)
    analysis_input = _build_analysis_input(
        benchmark_name=benchmark_name,
        benchmark_run_id=benchmark_run_id,
        suite=suite,
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
        environment_info=environment_info,
        config=config,
        results=results,
        test_cases=test_cases,
        raw_runs=raw_runs,
        measured_runs=measured_runs,
        summary_by_model=summary_by_model,
        summary_by_category=summary_by_category,
        summary_by_test_case=summary_by_test_case,
        rankings=rankings,
        anomalies=anomalies,
        failed_runs=failed_runs,
    )
    _validate_analysis_input(analysis_input)
    markdown_report = _render_template("report.md.j2", final_report)
    html_report = _render_template("report.html.j2", final_report)

    return ReportArtifacts(
        raw_runs=raw_runs,
        summary_by_model=summary_by_model,
        summary_by_category=summary_by_category,
        final_report=final_report,
        analysis_input=analysis_input,
        markdown_report=markdown_report,
        html_report=html_report,
    )


def _results_to_dataframe(results: list[RunResult]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for result in results:
        record = result.model_dump(mode="json")
        record["phase"] = result.metadata.get("phase", "measured")
        record["suite_list"] = list(result.metadata.get("suites", []))
        record["suite_names"] = ",".join(record["suite_list"])
        for score_key, score_value in result.score_breakdown.model_dump(mode="json").items():
            record[score_key] = score_value
        records.append(record)
    return pd.DataFrame.from_records(records)


def _summarize_by_model(measured_runs: pd.DataFrame) -> pd.DataFrame:
    if measured_runs.empty:
        return pd.DataFrame()

    grouped = measured_runs.groupby(["model_id", "model_label", "provider", "model_name"], dropna=False)
    summary = grouped.agg(
        total_runs=("test_case_id", "count"),
        success_rate=("success", "mean"),
        validation_pass_rate=("validation_passed", "mean"),
        avg_duration_ms=("duration_ms", "mean"),
        avg_ttft_ms=("ttft_ms", "mean"),
        avg_tokens_per_second=("tokens_per_second", "mean"),
        avg_quality_score=("quality_score", "mean"),
        avg_format_score=("format_score", "mean"),
        avg_latency_score=("latency_score", "mean"),
        avg_stability_score=("stability_score", "mean"),
        avg_instruction_score=("instruction_score", "mean"),
        avg_reproducibility_score=("reproducibility_score", "mean"),
        avg_total_score=("score_total", "mean"),
    ).reset_index()
    return _round_numeric(summary)


def _summarize_by_category(measured_runs: pd.DataFrame) -> pd.DataFrame:
    if measured_runs.empty:
        return pd.DataFrame()

    grouped = measured_runs.groupby(["model_id", "model_label", "category"], dropna=False)
    summary = grouped.agg(
        total_runs=("test_case_id", "count"),
        success_rate=("success", "mean"),
        validation_pass_rate=("validation_passed", "mean"),
        avg_duration_ms=("duration_ms", "mean"),
        avg_total_score=("score_total", "mean"),
        avg_format_score=("format_score", "mean"),
        avg_quality_score=("quality_score", "mean"),
    ).reset_index()
    return _round_numeric(summary)


def _summarize_by_test_case(measured_runs: pd.DataFrame) -> pd.DataFrame:
    if measured_runs.empty:
        return pd.DataFrame()

    grouped = measured_runs.groupby(["model_id", "model_label", "test_case_id", "category"], dropna=False)
    summary = grouped.agg(
        total_runs=("test_case_id", "count"),
        success_rate=("success", "mean"),
        validation_pass_rate=("validation_passed", "mean"),
        avg_duration_ms=("duration_ms", "mean"),
        avg_total_score=("score_total", "mean"),
    ).reset_index()
    return _round_numeric(summary)


def _build_rankings(summary_by_model: pd.DataFrame) -> list[dict[str, Any]]:
    if summary_by_model.empty:
        return []
    ranking_frame = summary_by_model.sort_values(
        by=["avg_total_score", "validation_pass_rate", "success_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranking_frame["rank"] = ranking_frame.index + 1
    ordered_columns = [
        "rank",
        "model_id",
        "model_label",
        "provider",
        "model_name",
        "avg_total_score",
        "success_rate",
        "validation_pass_rate",
        "avg_duration_ms",
    ]
    return _frame_to_records(ranking_frame[ordered_columns])


def _build_strengths_and_weaknesses(summary_by_category: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if summary_by_category.empty:
        return [], []

    strengths: list[dict[str, Any]] = []
    weaknesses: list[dict[str, Any]] = []
    for model_id, group in summary_by_category.groupby("model_id"):
        best = group.sort_values("avg_total_score", ascending=False).head(2)
        worst = group.sort_values("avg_total_score", ascending=True).head(2)
        for _, row in best.iterrows():
            strengths.append(
                {
                    "model_id": model_id,
                    "category": row["category"],
                    "average_score": round(float(row["avg_total_score"]), 2),
                    "summary": (
                        f"{row['model_label']} performs strongly in {row['category']} "
                        f"with an average score of {row['avg_total_score']:.2f}."
                    ),
                }
            )
        for _, row in worst.iterrows():
            weaknesses.append(
                {
                    "model_id": model_id,
                    "category": row["category"],
                    "average_score": round(float(row["avg_total_score"]), 2),
                    "summary": (
                        f"{row['model_label']} struggles most in {row['category']} "
                        f"with an average score of {row['avg_total_score']:.2f}."
                    ),
                }
            )
    return strengths, weaknesses


def _detect_anomalies(summary_by_model: pd.DataFrame, summary_by_category: pd.DataFrame) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    if not summary_by_model.empty:
        overall_duration = float(summary_by_model["avg_duration_ms"].median()) if len(summary_by_model) else 0.0
        for _, row in summary_by_model.iterrows():
            failure_rate = 1.0 - float(row["success_rate"])
            if failure_rate >= 0.25:
                anomalies.append(
                    {
                        "type": "high_failure_rate",
                        "model_id": row["model_id"],
                        "summary": (
                            f"{row['model_label']} has a failure rate of {failure_rate:.0%}, "
                            "which is high enough to question endpoint stability."
                        ),
                    }
                )
            if overall_duration and float(row["avg_duration_ms"]) > overall_duration * 2:
                anomalies.append(
                    {
                        "type": "high_latency",
                        "model_id": row["model_id"],
                        "summary": (
                            f"{row['model_label']} is more than twice as slow as the median measured model latency."
                        ),
                    }
                )
            if float(row["avg_reproducibility_score"]) < 60:
                anomalies.append(
                    {
                        "type": "low_reproducibility",
                        "model_id": row["model_id"],
                        "summary": (
                            f"{row['model_label']} shows low reproducibility with an average score below 60."
                        ),
                    }
                )

    if not summary_by_category.empty:
        weak_format = summary_by_category[summary_by_category["validation_pass_rate"] < 0.70]
        for _, row in weak_format.iterrows():
            anomalies.append(
                {
                    "type": "category_validation_gap",
                    "model_id": row["model_id"],
                    "summary": (
                        f"{row['model_label']} validates poorly in {row['category']} "
                        f"with only {row['validation_pass_rate']:.0%} passing runs."
                    ),
                }
            )
    return anomalies


def _extract_failed_runs(measured_runs: pd.DataFrame) -> list[dict[str, Any]]:
    if measured_runs.empty:
        return []
    failed = measured_runs[(measured_runs["success"] == False) | (measured_runs["validation_passed"] == False)].copy()
    if failed.empty:
        return []
    columns = [
        "model_id",
        "model_label",
        "test_case_id",
        "category",
        "repetition_index",
        "phase",
        "success",
        "validation_passed",
        "http_status",
        "error_type",
        "error_message",
        "duration_ms",
        "score_total",
    ]
    return _frame_to_records(failed[columns].head(50))


def _build_recommendations(
    summary_by_model: pd.DataFrame,
    summary_by_category: pd.DataFrame,
    rankings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    if rankings:
        best = rankings[0]
        recommendations.append(
            {
                "priority": "high",
                "summary": (
                    f"Use {best['model_label']} as the default candidate for this suite because it leads the ranking "
                    f"with an average total score of {best['avg_total_score']:.2f}."
                ),
            }
        )

    if not summary_by_model.empty:
        slowest = summary_by_model.sort_values("avg_duration_ms", ascending=False).iloc[0]
        recommendations.append(
            {
                "priority": "medium",
                "summary": (
                    f"Batch or background workloads are a better fit for {slowest['model_label']} "
                    "if latency matters more than peak quality."
                ),
            }
        )

        least_stable = summary_by_model.sort_values("success_rate", ascending=True).iloc[0]
        if float(least_stable["success_rate"]) < 0.90:
            recommendations.append(
                {
                    "priority": "high",
                    "summary": (
                        f"Stabilize the endpoint behind {least_stable['model_label']} before treating it as a "
                        "drop-in production replacement."
                    ),
                }
            )

    if not summary_by_category.empty:
        best_format = summary_by_category.sort_values("avg_format_score", ascending=False).iloc[0]
        recommendations.append(
            {
                "priority": "medium",
                "summary": (
                    f"Prefer {best_format['model_label']} for structured workflows in category "
                    f"{best_format['category']} because it currently shows the strongest format adherence."
                ),
            }
        )

    return recommendations[:6]


def _build_suite_records(test_cases: list[TestCaseDefinition] | None, results: list[RunResult]) -> list[dict[str, Any]]:
    suite_map: dict[str, list[TestCaseDefinition | RunResult]] = defaultdict(list)
    if test_cases:
        for case in test_cases:
            for suite_name in case.suites:
                suite_map[suite_name].append(case)
        return [
            {
                "suite_name": suite_name,
                "test_count": len(case_items),
                "test_case_ids": [case.test_case_id for case in case_items if isinstance(case, TestCaseDefinition)],
                "categories": sorted(
                    {
                        case.category
                        for case in case_items
                        if isinstance(case, TestCaseDefinition)
                    }
                ),
            }
            for suite_name, case_items in sorted(suite_map.items())
        ]

    for result in results:
        for suite_name in result.metadata.get("suites", []):
            suite_map[suite_name].append(result)
    return [
        {
            "suite_name": suite_name,
            "test_count": len({item.test_case_id for item in items if isinstance(item, RunResult)}),
            "test_case_ids": sorted({item.test_case_id for item in items if isinstance(item, RunResult)}),
            "categories": sorted({item.category for item in items if isinstance(item, RunResult)}),
        }
        for suite_name, items in sorted(suite_map.items())
    ]


def _build_model_records(
    config: BenchmarkConfig | None,
    summary_by_model: pd.DataFrame,
    results: list[RunResult],
) -> list[dict[str, Any]]:
    summary_lookup = {
        row["model_id"]: row
        for row in _frame_to_records(summary_by_model)
    }
    if config:
        records: list[dict[str, Any]] = []
        for model in config.enabled_models():
            summary = summary_lookup.get(model.id, {})
            records.append(
                {
                    "model_id": model.id,
                    "model_label": model.label,
                    "provider": model.provider,
                    "base_url": model.base_url,
                    "model_name": model.model_name,
                    "supports_streaming": model.supports_streaming,
                    "supports_tools": model.supports_tools,
                    "supports_structured_output": model.supports_structured_output,
                    "summary": summary,
                }
            )
        return records

    discovered: dict[str, dict[str, Any]] = {}
    for result in results:
        discovered.setdefault(
            result.model_id,
            {
                "model_id": result.model_id,
                "model_label": result.model_label,
                "provider": result.provider,
                "base_url": result.endpoint,
                "model_name": result.model_name,
                "summary": summary_lookup.get(result.model_id, {}),
            },
        )
    return list(discovered.values())


def _render_template(template_name: str, report_payload: dict[str, Any]) -> str:
    environment = jinja2.Environment(
        loader=jinja2.PackageLoader("llm_benchmark", "templates"),
        autoescape=jinja2.select_autoescape(enabled_extensions=("html", "xml")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template(template_name)
    return template.render(report=report_payload)


def _validate_final_report(final_report: dict[str, Any]) -> None:
    schema_bytes = resources.files("llm_benchmark.reporting").joinpath("final_report_schema.json").read_bytes()
    schema = orjson.loads(schema_bytes)
    jsonschema.Draft202012Validator(schema).validate(final_report)


def _validate_analysis_input(analysis_input: dict[str, Any]) -> None:
    schema_bytes = resources.files("llm_benchmark.reporting").joinpath("analysis_input_schema.json").read_bytes()
    schema = orjson.loads(schema_bytes)
    jsonschema.Draft202012Validator(schema).validate(analysis_input)


def _build_analysis_input(
    *,
    benchmark_name: str,
    benchmark_run_id: str,
    suite: str | None,
    run_started_at: str,
    run_finished_at: str,
    environment_info: dict[str, Any],
    config: BenchmarkConfig | None,
    results: list[RunResult],
    test_cases: list[TestCaseDefinition] | None,
    raw_runs: pd.DataFrame,
    measured_runs: pd.DataFrame,
    summary_by_model: pd.DataFrame,
    summary_by_category: pd.DataFrame,
    summary_by_test_case: pd.DataFrame,
    rankings: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    failed_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    test_metadata = _build_test_metadata_lookup(test_cases, results)
    enriched_measured_runs = _enrich_runs_with_test_metadata(measured_runs, test_metadata)
    fastest_model = _pick_model_by_speed(summary_by_model, fastest=True)
    slowest_model = _pick_model_by_speed(summary_by_model, fastest=False)
    representative_success_examples = _build_representative_success_examples(enriched_measured_runs)
    representative_failure_examples = failed_runs[:10]
    repo_recommendations = _build_repo_recommendations(enriched_measured_runs)
    security_behavior_summary = _build_security_behavior_summary(enriched_measured_runs)
    structured_output_summary = _build_structured_output_summary(enriched_measured_runs)
    voice_response_summary = _build_voice_response_summary(enriched_measured_runs)
    tax_enrichment_summary = _build_tax_enrichment_summary(enriched_measured_runs)

    return {
        "schema_version": "1.0",
        "purpose": "secondary_llm_analysis_input",
        "benchmark_metadata": {
            "benchmark_run_id": benchmark_run_id,
            "benchmark_name": benchmark_name,
            "suite": suite or "all",
            "run_started_at": run_started_at,
            "run_finished_at": run_finished_at,
            "generated_at": isoformat_utc(),
            "total_records": int(len(raw_runs)),
            "measured_records": int(len(measured_runs)),
            "warmup_records": int((raw_runs["phase"] == "warmup").sum()) if not raw_runs.empty else 0,
        },
        "hardware_and_environment_hints": {
            "environment_info": environment_info,
            "configured_execution_profile": config.run_defaults.model_dump(mode="json") if config else {},
            "interpretation_hints": [
                "Local CPU-bound models may show significantly higher latency than cloud APIs.",
                "Missing token metrics should be treated as unavailable provider telemetry, not zero usage.",
                "Warmup runs are excluded from rankings but remain useful when investigating cache effects.",
            ],
        },
        "model_rankings": rankings,
        "speed_summary": {
            "fastest_model_by_avg_duration": fastest_model,
            "slowest_model_by_avg_duration": slowest_model,
        },
        "category_findings": {
            "best_category_rows": _frame_to_records(
                summary_by_category.sort_values("avg_total_score", ascending=False).head(10)
                if not summary_by_category.empty
                else pd.DataFrame()
            ),
            "worst_category_rows": _frame_to_records(
                summary_by_category.sort_values("avg_total_score", ascending=True).head(10)
                if not summary_by_category.empty
                else pd.DataFrame()
            ),
            "best_test_case_rows": _frame_to_records(
                summary_by_test_case.sort_values("avg_total_score", ascending=False).head(15)
                if not summary_by_test_case.empty
                else pd.DataFrame()
            ),
            "worst_test_case_rows": _frame_to_records(
                summary_by_test_case.sort_values("avg_total_score", ascending=True).head(15)
                if not summary_by_test_case.empty
                else pd.DataFrame()
            ),
        },
        "aggregate_scores": {
            "overall_average_total_score": round(
                float(measured_runs["score_total"].mean()) if not measured_runs.empty else 0.0, 2
            ),
            "by_model": _frame_to_records(summary_by_model),
            "by_category": _frame_to_records(summary_by_category),
            "by_test_case": _frame_to_records(summary_by_test_case),
        },
        "anomalies": anomalies,
        "representative_failures": representative_failure_examples,
        "representative_success_examples": representative_success_examples,
        "model_cards": _build_analysis_model_cards(summary_by_model, summary_by_category),
        "repo_recommendations": repo_recommendations,
        "best_model_for_secondbrain": repo_recommendations["secondbrain"]["best_model"],
        "best_model_for_voice_gateway": repo_recommendations["voice_gateway"]["best_model"],
        "best_model_for_paperless_kiplus": repo_recommendations["paperless_kiplus"]["best_model"],
        "security_behavior_summary": security_behavior_summary,
        "structured_output_summary": structured_output_summary,
        "voice_response_summary": voice_response_summary,
        "tax_enrichment_summary": tax_enrichment_summary,
        "follow_up_recommendations": [
            "Compare the top-ranked model against the fastest model if latency matters more than absolute quality.",
            "Inspect representative failures before promoting a local endpoint into production workflows.",
            "Re-run the quick_compare suite after changing local model quantization, runtime flags or hardware placement.",
            "Use the repo-specific suites before selecting a default model for SecondBrain, secondbrain-voice-gateway or Paperless-KIplus.",
        ],
    }


def _pick_model_by_speed(summary_by_model: pd.DataFrame, *, fastest: bool) -> dict[str, Any] | None:
    if summary_by_model.empty:
        return None
    ordered = summary_by_model.sort_values("avg_duration_ms", ascending=fastest)
    return _frame_to_records(ordered.head(1))[0]


def _build_representative_success_examples(measured_runs: pd.DataFrame) -> list[dict[str, Any]]:
    if measured_runs.empty:
        return []

    successful = measured_runs[(measured_runs["success"] == True) & (measured_runs["validation_passed"] == True)].copy()
    if successful.empty:
        return []

    examples: list[dict[str, Any]] = []
    for model_id, group in successful.groupby("model_id"):
        best_row = group.sort_values(["score_total", "duration_ms"], ascending=[False, True]).iloc[0]
        examples.append(
            {
                "model_id": model_id,
                "model_label": best_row["model_label"],
                "test_case_id": best_row["test_case_id"],
                "category": best_row["category"],
                "phase": best_row["phase"],
                "score_total": best_row["score_total"],
                "duration_ms": best_row["duration_ms"],
                "validation_passed": bool(best_row["validation_passed"]),
                "response_excerpt": _truncate_text(best_row.get("raw_response_text")),
                "parsed_output_json_excerpt": _truncate_json(best_row.get("parsed_output_json")),
            }
        )
    return sorted(examples, key=lambda item: (-float(item["score_total"]), float(item["duration_ms"])))[:10]


def _build_analysis_model_cards(summary_by_model: pd.DataFrame, summary_by_category: pd.DataFrame) -> list[dict[str, Any]]:
    if summary_by_model.empty:
        return []

    cards: list[dict[str, Any]] = []
    category_records = _frame_to_records(summary_by_category)
    category_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in category_records:
        category_by_model[str(record["model_id"])].append(record)

    for model_record in _frame_to_records(summary_by_model):
        model_id = str(model_record["model_id"])
        categories = category_by_model.get(model_id, [])
        best_category = max(categories, key=lambda item: item["avg_total_score"], default=None)
        worst_category = min(categories, key=lambda item: item["avg_total_score"], default=None)
        cards.append(
            {
                "model_id": model_id,
                "model_label": model_record["model_label"],
                "provider": model_record["provider"],
                "summary": model_record,
                "best_category": best_category,
                "worst_category": worst_category,
            }
        )
    return cards


def _build_test_metadata_lookup(
    test_cases: list[TestCaseDefinition] | None,
    results: list[RunResult],
) -> dict[str, dict[str, Any]]:
    """
    Build stable per-test metadata for suite-, tag- and format-specific analysis.

    Why this exists:
    The dashboard and `analysis_input.json` should stay useful even when reports are rebuilt
    from JSONL only, so we fall back to persisted run metadata when test definitions are absent.
    """

    metadata: dict[str, dict[str, Any]] = {}
    if test_cases:
        for case in test_cases:
            metadata[case.test_case_id] = {
                "suites": list(case.suites),
                "tags": list(case.tags),
                "expected_format": case.expected_format,
                "title": case.title,
                "category": case.category,
            }

    for result in results:
        metadata.setdefault(
            result.test_case_id,
            {
                "suites": list(result.metadata.get("suites", [])),
                "tags": list(result.metadata.get("tags", [])),
                "expected_format": result.metadata.get("expected_format"),
                "title": result.metadata.get("title", result.test_case_id),
                "category": result.category,
            },
        )
    return metadata


def _enrich_runs_with_test_metadata(
    frame: pd.DataFrame,
    test_metadata: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Attach suite, tag and expected-format metadata to run rows for subset summaries."""

    if frame.empty:
        return frame.copy()

    enriched = frame.copy()

    def _lookup(test_case_id: Any, key: str, default: Any) -> Any:
        return test_metadata.get(str(test_case_id), {}).get(key, default)

    enriched["suite_list"] = enriched.apply(
        lambda row: _normalize_string_list(row.get("suite_list"))
        or _normalize_string_list(_lookup(row["test_case_id"], "suites", [])),
        axis=1,
    )
    enriched["test_tags"] = enriched["test_case_id"].map(
        lambda value: _normalize_string_list(_lookup(value, "tags", []))
    )
    enriched["expected_format"] = enriched["test_case_id"].map(
        lambda value: _lookup(value, "expected_format", None)
    )
    enriched["test_title"] = enriched["test_case_id"].map(
        lambda value: _lookup(value, "title", str(value))
    )
    return enriched


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _filter_runs_by_suite(frame: pd.DataFrame, suite_name: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = frame["suite_list"].apply(lambda suites: suite_name in _normalize_string_list(suites))
    return frame[mask].copy()


def _filter_runs_by_tags(frame: pd.DataFrame, required_tags: set[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = frame["test_tags"].apply(lambda tags: bool(required_tags.intersection(_normalize_string_list(tags))))
    return frame[mask].copy()


def _filter_runs_by_expected_formats(frame: pd.DataFrame, expected_formats: set[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = frame["expected_format"].apply(lambda value: str(value) in expected_formats)
    return frame[mask].copy()


def _summarize_subset_by_model(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    grouped = frame.groupby(["model_id", "model_label", "provider", "model_name"], dropna=False)
    summary = grouped.agg(
        total_runs=("test_case_id", "count"),
        success_rate=("success", "mean"),
        validation_pass_rate=("validation_passed", "mean"),
        avg_duration_ms=("duration_ms", "mean"),
        avg_quality_score=("quality_score", "mean"),
        avg_format_score=("format_score", "mean"),
        avg_instruction_score=("instruction_score", "mean"),
        avg_total_score=("score_total", "mean"),
    ).reset_index()
    return _round_numeric(summary)


def _summarize_subset_by_category(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    grouped = frame.groupby(["model_id", "model_label", "category"], dropna=False)
    summary = grouped.agg(
        total_runs=("test_case_id", "count"),
        success_rate=("success", "mean"),
        validation_pass_rate=("validation_passed", "mean"),
        avg_duration_ms=("duration_ms", "mean"),
        avg_total_score=("score_total", "mean"),
    ).reset_index()
    return _round_numeric(summary)


def _build_repo_recommendations(enriched_measured_runs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    recommendations: dict[str, dict[str, Any]] = {}
    for suite_name, project_label in REPO_SUITE_LABELS.items():
        suite_runs = _filter_runs_by_suite(enriched_measured_runs, suite_name)
        suite_summary = _summarize_subset_by_model(suite_runs)
        suite_categories = _summarize_subset_by_category(suite_runs)
        ranking = _build_rankings(suite_summary)
        best_model = ranking[0] if ranking else None
        strengths = _frame_to_records(
            suite_categories.sort_values("avg_total_score", ascending=False).head(3)
            if not suite_categories.empty
            else pd.DataFrame()
        )
        weaknesses = _frame_to_records(
            suite_categories.sort_values("avg_total_score", ascending=True).head(3)
            if not suite_categories.empty
            else pd.DataFrame()
        )
        failures = _extract_failed_runs(suite_runs)[:5]

        summary_text = (
            f"{best_model['model_label']} currently leads the {project_label} suite with an average total score "
            f"of {best_model['avg_total_score']:.2f}."
            if best_model
            else f"No measured runs are available yet for the {project_label} suite."
        )

        recommendations[suite_name] = {
            "suite_name": suite_name,
            "project_label": project_label,
            "focus_areas": REPO_SUITE_FOCUS_AREAS[suite_name],
            "benchmark_fit_summary": summary_text,
            "best_model": best_model,
            "ranking": ranking[:5],
            "project_strengths": strengths,
            "project_weaknesses": weaknesses,
            "notable_failures": failures,
        }
    return recommendations


def _build_security_behavior_summary(enriched_measured_runs: pd.DataFrame) -> dict[str, Any]:
    security_runs = _filter_runs_by_tags(enriched_measured_runs, {"security", "prompt_injection", "safety"})
    summary = _summarize_subset_by_model(security_runs)
    ranking = _build_rankings(summary)

    return {
        "covered_test_case_ids": sorted(security_runs["test_case_id"].unique().tolist()) if not security_runs.empty else [],
        "overall_validation_pass_rate": round(
            float(security_runs["validation_passed"].mean()) if not security_runs.empty else 0.0,
            2,
        ),
        "top_model": ranking[0] if ranking else None,
        "summary_by_model": _frame_to_records(summary),
        "representative_failures": _extract_failed_runs(security_runs)[:5],
    }


def _build_structured_output_summary(enriched_measured_runs: pd.DataFrame) -> dict[str, Any]:
    structured_runs = _filter_runs_by_expected_formats(enriched_measured_runs, {"json", "yaml", "tool_call"})
    summary = _summarize_subset_by_model(structured_runs)
    ranking = _build_rankings(summary)

    by_format: dict[str, dict[str, Any]] = {}
    for format_name in ("json", "yaml", "tool_call"):
        subset = (
            structured_runs[structured_runs["expected_format"] == format_name].copy()
            if not structured_runs.empty
            else pd.DataFrame()
        )
        format_summary = _summarize_subset_by_model(subset)
        format_ranking = _build_rankings(format_summary)
        by_format[format_name] = {
            "run_count": int(len(subset)),
            "validation_pass_rate": round(float(subset["validation_passed"].mean()) if not subset.empty else 0.0, 2),
            "top_model": format_ranking[0] if format_ranking else None,
        }

    return {
        "covered_test_case_ids": (
            sorted(structured_runs["test_case_id"].unique().tolist()) if not structured_runs.empty else []
        ),
        "overall_validation_pass_rate": round(
            float(structured_runs["validation_passed"].mean()) if not structured_runs.empty else 0.0,
            2,
        ),
        "top_model": ranking[0] if ranking else None,
        "summary_by_model": _frame_to_records(summary),
        "by_expected_format": by_format,
    }


def _build_voice_response_summary(enriched_measured_runs: pd.DataFrame) -> dict[str, Any]:
    voice_suite_runs = _filter_runs_by_suite(enriched_measured_runs, "voice_gateway")
    voice_runs = (
        voice_suite_runs[
            voice_suite_runs["expected_format"].eq("text")
            | voice_suite_runs["test_tags"].apply(
                lambda tags: bool({"speech", "alexa"}.intersection(_normalize_string_list(tags)))
            )
        ].copy()
        if not voice_suite_runs.empty
        else pd.DataFrame()
    )
    summary = _summarize_subset_by_model(voice_runs)
    ranking = _build_rankings(summary)

    return {
        "covered_test_case_ids": sorted(voice_runs["test_case_id"].unique().tolist()) if not voice_runs.empty else [],
        "overall_success_rate": round(float(voice_runs["success"].mean()) if not voice_runs.empty else 0.0, 2),
        "top_model": ranking[0] if ranking else None,
        "summary_by_model": _frame_to_records(summary),
        "representative_failures": _extract_failed_runs(voice_runs)[:5],
    }


def _build_tax_enrichment_summary(enriched_measured_runs: pd.DataFrame) -> dict[str, Any]:
    tax_runs = _filter_runs_by_tags(enriched_measured_runs, {"tax"})
    summary = _summarize_subset_by_model(tax_runs)
    ranking = _build_rankings(summary)

    return {
        "covered_test_case_ids": sorted(tax_runs["test_case_id"].unique().tolist()) if not tax_runs.empty else [],
        "overall_validation_pass_rate": round(
            float(tax_runs["validation_passed"].mean()) if not tax_runs.empty else 0.0,
            2,
        ),
        "top_model": ranking[0] if ranking else None,
        "summary_by_model": _frame_to_records(summary),
        "representative_failures": _extract_failed_runs(tax_runs)[:5],
    }


def _truncate_text(value: Any, *, max_chars: int = 320) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _truncate_json(value: Any, *, max_chars: int = 320) -> str | None:
    if value is None:
        return None
    try:
        serialized = orjson.dumps(value).decode("utf-8")
    except TypeError:
        serialized = str(value)
    return _truncate_text(serialized, max_chars=max_chars)


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    sanitized = frame.where(frame.notna(), None)
    return sanitized.to_dict(orient="records")


def _round_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    rounded = frame.copy()
    numeric_columns = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric_columns] = rounded[numeric_columns].round(2)
    return rounded
