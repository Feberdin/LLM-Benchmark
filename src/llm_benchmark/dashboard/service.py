"""
Purpose: Build a cached, read-only dashboard view model from benchmark result artifacts and test metadata.
Input/Output: Reads JSON/JSONL report files plus optional test definitions and returns HTML/API friendly dictionaries.
Important invariants: The service never changes benchmark artifacts; it only enriches them for browsing and filtering.
How to debug: If dashboard cards or rankings look off, inspect `_build_runs_frame` and the filter logic in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
import pandas as pd

from llm_benchmark.config.loader import load_test_cases
from llm_benchmark.domain.result import RunResult
from llm_benchmark.domain.test_case import TestCaseDefinition
from llm_benchmark.reporting.exporters import load_results_from_jsonl
from llm_benchmark.utils import isoformat_utc

REPORT_FILES = [
    "raw_runs.jsonl",
    "raw_runs.csv",
    "summary_by_model.csv",
    "summary_by_category.csv",
    "final_report.md",
    "final_report.html",
    "final_report.json",
    "analysis_input.json",
]

VIEW_OPTIONS = [
    ("overview", "Overview"),
    ("models", "Models"),
    ("categories", "Categories"),
    ("tests", "Tests"),
    ("failures", "Failures"),
    ("domains", "Domains"),
    ("downloads", "Downloads"),
]

@dataclass(slots=True)
class DashboardFilters:
    """Simple container for server-side dashboard filters."""

    view: str = "overview"
    model: str | None = None
    category: str | None = None
    suite: str | None = None
    status: str = "all"
    error_type: str | None = None
    search: str | None = None


class DashboardService:
    """Load benchmark artifacts once and expose filtered dashboard-ready sections."""

    def __init__(self, *, results_dir: Path, tests_dir: Path | None) -> None:
        self.results_dir = results_dir
        self.tests_dir = tests_dir
        self._cache_signature: tuple[int, ...] | None = None
        self._cache: dict[str, Any] | None = None

    def health(self) -> dict[str, Any]:
        """Return a compact health payload that is useful for container probes and operators."""

        context = self._load_context()
        overview = context["overview"]
        return {
            "status": "ok",
            "has_results": overview["has_results"],
            "results_dir": str(self.results_dir),
            "tests_dir": str(self.tests_dir) if self.tests_dir else None,
            "latest_report_generated_at": overview.get("generated_at"),
            "available_downloads": [item["name"] for item in context["downloads"]],
        }

    def build_dashboard_context(self, filters: DashboardFilters) -> dict[str, Any]:
        """Build the HTML view model including filtered raw runs and precomputed summaries."""

        context = self._load_context()
        frame = context["runs_frame"]
        filtered = self._apply_filters(frame, filters)

        tests_lookup: dict[str, dict[str, Any]] = context["tests_lookup"]
        filtered_rows = self._build_test_rows(filtered, tests_lookup)
        filtered_categories = self._build_category_sections(filtered)
        filtered_models = self._build_model_cards(filtered)
        filtered_failures = self._build_failure_rows(filtered)
        domain_sections = self._build_domain_sections(
            filtered,
            analysis_input=context["analysis_input"],
            tests_lookup=tests_lookup,
        )

        return {
            "page_title": "LLM Benchmark Dashboard",
            "generated_at": isoformat_utc(),
            "view_options": VIEW_OPTIONS,
            "filters": filters,
            "filter_options": context["filter_options"],
            "overview": context["overview"],
            "models": filtered_models or context["models"],
            "categories": filtered_categories or context["categories"],
            "tests": filtered_rows,
            "failures": {
                "rows": filtered_failures,
                "anomalies": context["anomalies"],
            },
            "downloads": context["downloads"],
            "domains": domain_sections,
            "environment_info": context["environment_info"],
            "analysis_input": context["analysis_input"],
            "final_report": context["final_report"],
        }

    def api_summary(self) -> dict[str, Any]:
        context = self._load_context()
        return {
            "overview": context["overview"],
            "environment_info": context["environment_info"],
            "downloads": context["downloads"],
        }

    def api_models(self) -> list[dict[str, Any]]:
        return self._load_context()["models"]

    def api_categories(self) -> list[dict[str, Any]]:
        return self._load_context()["categories"]

    def api_failures(self) -> dict[str, Any]:
        context = self._load_context()
        return {"anomalies": context["anomalies"], "rows": self._build_failure_rows(context["runs_frame"])}

    def api_domains(self) -> dict[str, Any]:
        context = self._load_context()
        return self._build_domain_sections(
            context["runs_frame"],
            analysis_input=context["analysis_input"],
            tests_lookup=context["tests_lookup"],
        )

    def api_tests(self, filters: DashboardFilters) -> list[dict[str, Any]]:
        context = self._load_context()
        filtered = self._apply_filters(context["runs_frame"], filters)
        return self._build_test_rows(filtered, context["tests_lookup"])

    def available_download(self, filename: str) -> Path | None:
        candidate = self.results_dir / filename
        if filename not in REPORT_FILES or not candidate.exists():
            return None
        return candidate

    def _load_context(self) -> dict[str, Any]:
        signature = self._build_signature()
        if self._cache is not None and self._cache_signature == signature:
            return self._cache

        final_report = self._load_json(self.results_dir / "final_report.json")
        analysis_input = self._load_json(self.results_dir / "analysis_input.json")
        test_cases = self._load_test_cases()
        tests_lookup = self._build_tests_lookup(test_cases)
        results = self._load_results()
        runs_frame = self._build_runs_frame(results, tests_lookup)
        models = self._build_model_cards(runs_frame)
        categories = self._build_category_sections(runs_frame)
        downloads = self._build_downloads()
        anomalies = final_report.get("anomalies", [])
        overview = self._build_overview(final_report, analysis_input, models, downloads, anomalies)

        self._cache_signature = signature
        self._cache = {
            "final_report": final_report,
            "analysis_input": analysis_input,
            "environment_info": final_report.get("environment_info", analysis_input.get("hardware_and_environment_hints", {})),
            "overview": overview,
            "models": models,
            "categories": categories,
            "downloads": downloads,
            "anomalies": anomalies,
            "runs_frame": runs_frame,
            "tests_lookup": tests_lookup,
            "filter_options": self._build_filter_options(runs_frame),
        }
        return self._cache

    def _build_signature(self) -> tuple[int, ...]:
        tracked_paths = [
            self.results_dir / "final_report.json",
            self.results_dir / "analysis_input.json",
            self.results_dir / "raw_runs.jsonl",
        ]
        signature: list[int] = []
        for path in tracked_paths:
            signature.append(path.stat().st_mtime_ns if path.exists() else 0)
        signature.append(self._latest_tests_mtime())
        return tuple(signature)

    def _latest_tests_mtime(self) -> int:
        if not self.tests_dir or not self.tests_dir.exists():
            return 0
        latest = 0
        for path in self.tests_dir.rglob("*"):
            if path.is_file():
                latest = max(latest, path.stat().st_mtime_ns)
        return latest

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return orjson.loads(path.read_bytes())

    def _load_results(self) -> list[RunResult]:
        path = self.results_dir / "raw_runs.jsonl"
        if not path.exists():
            return []
        return load_results_from_jsonl(path)

    def _load_test_cases(self) -> list[TestCaseDefinition]:
        if not self.tests_dir or not self.tests_dir.exists():
            return []
        return load_test_cases(self.tests_dir)

    def _build_tests_lookup(self, test_cases: list[TestCaseDefinition]) -> dict[str, dict[str, Any]]:
        lookup: dict[str, dict[str, Any]] = {}
        for case in test_cases:
            lookup[case.test_case_id] = {
                "title": case.title,
                "description": case.description,
                "prompt_excerpt": self._truncate_text(case.prompt, 560),
                "system_prompt_excerpt": self._truncate_text(case.system_prompt, 320),
                "expected_format": case.expected_format,
                "suites": case.suites,
                "tags": case.tags,
                "category": case.category,
                "max_output_tokens": case.max_output_tokens,
            }
        return lookup

    def _build_runs_frame(
        self,
        results: list[RunResult],
        tests_lookup: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Flatten raw results into a DataFrame that is easy to filter and summarize.

        Example output columns:
        - model_label, test_case_id, suite_names, expected_format
        - quality_score, total_score, validation_error_codes
        - response_excerpt, parsed_output_excerpt
        """

        records: list[dict[str, Any]] = []
        for result in results:
            test_meta = tests_lookup.get(result.test_case_id, {})
            issue_codes = [str(issue.get("code")) for issue in result.validation_errors]
            suites = result.metadata.get("suites") or test_meta.get("suites", [])
            tags = result.metadata.get("tags") or test_meta.get("tags", [])
            expected_format = result.metadata.get("expected_format") or test_meta.get("expected_format")
            record = {
                "benchmark_run_id": result.benchmark_run_id,
                "run_started_at": result.run_started_at,
                "run_finished_at": result.run_finished_at,
                "model_id": result.model_id,
                "model_label": result.model_label,
                "provider": result.provider,
                "model_name": result.model_name,
                "endpoint": result.endpoint,
                "test_case_id": result.test_case_id,
                "test_title": test_meta.get("title", result.test_case_id),
                "category": result.category,
                "phase": result.metadata.get("phase", "measured"),
                "suite_list": list(suites),
                "suite_names": ", ".join(suites),
                "tags": list(tags),
                "expected_format": expected_format,
                "repetition_index": result.repetition_index,
                "duration_ms": result.duration_ms,
                "ttft_ms": result.ttft_ms,
                "tokens_per_second": result.tokens_per_second,
                "http_status": result.http_status,
                "success": result.success,
                "timeout": result.timeout,
                "retries": result.retries,
                "validation_passed": result.validation_passed,
                "error_type": result.error_type,
                "error_message": result.error_message,
                "validation_error_codes": issue_codes,
                "validation_error_summary": ", ".join(issue_codes),
                "quality_score": result.score_breakdown.quality_score,
                "format_score": result.score_breakdown.format_score,
                "latency_score": result.score_breakdown.latency_score,
                "stability_score": result.score_breakdown.stability_score,
                "instruction_score": result.score_breakdown.instruction_score,
                "reproducibility_score": result.score_breakdown.reproducibility_score,
                "score_total": result.score_total,
                "response_excerpt": self._truncate_text(result.raw_response_text, 360),
                "parsed_output_excerpt": self._truncate_json(result.parsed_output_json, 360),
                "prompt_excerpt": test_meta.get("prompt_excerpt"),
                "description": test_meta.get("description"),
            }
            records.append(record)
        return pd.DataFrame.from_records(records)

    def _build_overview(
        self,
        final_report: dict[str, Any],
        analysis_input: dict[str, Any],
        models: list[dict[str, Any]],
        downloads: list[dict[str, Any]],
        anomalies: list[dict[str, Any]],
    ) -> dict[str, Any]:
        benchmark_info = final_report.get("benchmark_info", analysis_input.get("benchmark_metadata", {}))
        model_rankings = analysis_input.get("model_rankings", final_report.get("rankings", []))
        fastest = analysis_input.get("speed_summary", {}).get("fastest_model_by_avg_duration")
        best_quality = max(models, key=lambda item: item["avg_quality_score"], default=None)
        best_openai_replacement = next(
            (item for item in sorted(models, key=lambda item: item["avg_total_score"], reverse=True) if item["provider"] != "openai"),
            None,
        )

        return {
            "has_results": bool(benchmark_info),
            "benchmark_name": benchmark_info.get("benchmark_name"),
            "suite": benchmark_info.get("suite"),
            "run_started_at": benchmark_info.get("run_started_at"),
            "run_finished_at": benchmark_info.get("run_finished_at"),
            "generated_at": benchmark_info.get("generated_at"),
            "total_records": benchmark_info.get("total_records", 0),
            "measured_records": benchmark_info.get("measured_records", 0),
            "overall_ranking": model_rankings[:5],
            "fastest_model": fastest,
            "best_quality_model": best_quality,
            "best_openai_replacement": best_openai_replacement,
            "recommendations": analysis_input.get("follow_up_recommendations", final_report.get("recommendations", [])),
            "anomalies": anomalies[:6],
            "downloads": downloads[:6],
            "repo_cards": [
                {
                    "title": "Bestes Modell fuer SecondBrain",
                    "payload": analysis_input.get("best_model_for_secondbrain"),
                },
                {
                    "title": "Bestes Modell fuer secondbrain-voice-gateway",
                    "payload": analysis_input.get("best_model_for_voice_gateway"),
                },
                {
                    "title": "Bestes Modell fuer Paperless-KIplus",
                    "payload": analysis_input.get("best_model_for_paperless_kiplus"),
                }
            ],
        }

    def _build_model_cards(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []

        cards: list[dict[str, Any]] = []
        for _, group in frame.groupby(["model_id", "model_label", "provider", "model_name"], dropna=False):
            first = group.iloc[0]
            category_scores = (
                group.groupby("category", dropna=False)["score_total"].mean().sort_values(ascending=False)
            )
            structured = group[group["expected_format"].isin(["json", "yaml"])]
            tool_runs = group[group["expected_format"] == "tool_call"]
            cards.append(
                {
                    "model_id": first["model_id"],
                    "model_label": first["model_label"],
                    "provider": first["provider"],
                    "model_name": first["model_name"],
                    "avg_total_score": round(float(group["score_total"].mean()), 2),
                    "avg_quality_score": round(float(group["quality_score"].mean()), 2),
                    "avg_format_score": round(float(group["format_score"].mean()), 2),
                    "avg_latency_score": round(float(group["latency_score"].mean()), 2),
                    "avg_stability_score": round(float(group["stability_score"].mean()), 2),
                    "avg_instruction_score": round(float(group["instruction_score"].mean()), 2),
                    "avg_duration_ms": round(float(group["duration_ms"].mean()), 2),
                    "median_duration_ms": round(float(group["duration_ms"].median()), 2),
                    "p95_duration_ms": round(float(group["duration_ms"].quantile(0.95)), 2),
                    "success_rate": round(float(group["success"].mean()) * 100, 1),
                    "validation_pass_rate": round(float(group["validation_passed"].mean()) * 100, 1),
                    "json_validity_rate": round(float(structured["validation_passed"].mean()) * 100, 1)
                    if not structured.empty
                    else None,
                    "tool_call_validity_rate": round(float(tool_runs["validation_passed"].mean()) * 100, 1)
                    if not tool_runs.empty
                    else None,
                    "strengths": category_scores.head(2).index.tolist(),
                    "weaknesses": list(reversed(category_scores.tail(2).index.tolist())),
                }
            )
        return sorted(cards, key=lambda item: item["avg_total_score"], reverse=True)

    def _build_category_sections(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []

        sections: list[dict[str, Any]] = []
        for category, category_group in frame.groupby("category", dropna=False):
            models: list[dict[str, Any]] = []
            for _, model_group in category_group.groupby(["model_id", "model_label"], dropna=False):
                first = model_group.iloc[0]
                models.append(
                    {
                        "model_id": first["model_id"],
                        "model_label": first["model_label"],
                        "avg_total_score": round(float(model_group["score_total"].mean()), 2),
                        "avg_duration_ms": round(float(model_group["duration_ms"].mean()), 2),
                        "success_rate": round(float(model_group["success"].mean()) * 100, 1),
                        "validation_pass_rate": round(float(model_group["validation_passed"].mean()) * 100, 1),
                        "test_case_ids": sorted(model_group["test_case_id"].unique().tolist()),
                    }
                )
            sections.append(
                {
                    "category": category,
                    "models": sorted(models, key=lambda item: item["avg_total_score"], reverse=True),
                }
            )
        return sorted(sections, key=lambda item: item["category"])

    def _build_test_rows(
        self,
        frame: pd.DataFrame,
        tests_lookup: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if frame.empty:
            return []

        ordered = frame.sort_values(
            by=["run_finished_at", "score_total", "duration_ms"],
            ascending=[False, False, True],
        ).head(160)
        rows: list[dict[str, Any]] = []
        for _, row in ordered.iterrows():
            test_meta = tests_lookup.get(str(row["test_case_id"]), {})
            rows.append(
                {
                    "model_label": row["model_label"],
                    "provider": row["provider"],
                    "test_case_id": row["test_case_id"],
                    "test_title": row["test_title"],
                    "category": row["category"],
                    "suite_names": row["suite_names"],
                    "phase": row["phase"],
                    "status": "ok" if row["success"] and row["validation_passed"] else "problem",
                    "duration_ms": round(float(row["duration_ms"]), 2),
                    "score_total": round(float(row["score_total"]), 2),
                    "validation_passed": bool(row["validation_passed"]),
                    "error_type": row["error_type"],
                    "error_message": row["error_message"],
                    "validation_error_summary": row["validation_error_summary"],
                    "response_excerpt": row["response_excerpt"],
                    "parsed_output_excerpt": row["parsed_output_excerpt"],
                    "prompt_excerpt": test_meta.get("prompt_excerpt") or row["prompt_excerpt"],
                    "description": test_meta.get("description") or row["description"],
                    "expected_format": row["expected_format"],
                    "tags": row["tags"],
                }
            )
        return rows

    def _build_failure_rows(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        failed = frame[(frame["success"] == False) | (frame["validation_passed"] == False)].copy()  # noqa: E712
        if failed.empty:
            return []
        ordered = failed.sort_values(by=["run_finished_at", "duration_ms"], ascending=[False, False]).head(80)
        rows: list[dict[str, Any]] = []
        for _, row in ordered.iterrows():
            rows.append(
                {
                    "model_label": row["model_label"],
                    "test_case_id": row["test_case_id"],
                    "test_title": row["test_title"],
                    "category": row["category"],
                    "suite_names": row["suite_names"],
                    "error_type": row["error_type"] or "validation_failure",
                    "error_message": row["error_message"] or row["validation_error_summary"],
                    "http_status": row["http_status"],
                    "timeout": bool(row["timeout"]),
                    "retries": int(row["retries"]),
                    "duration_ms": round(float(row["duration_ms"]), 2),
                    "score_total": round(float(row["score_total"]), 2),
                    "response_excerpt": row["response_excerpt"],
                }
            )
        return rows

    def _build_domain_sections(
        self,
        frame: pd.DataFrame,
        *,
        analysis_input: dict[str, Any],
        tests_lookup: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        repo_recommendations = analysis_input.get("repo_recommendations", {})
        focus_rankings = self._build_focus_area_rankings(frame)
        return {
            "repo_recommendations": repo_recommendations,
            "recommendation_cards": [
                {
                    "title": "Bestes Modell fuer SecondBrain",
                    "payload": analysis_input.get("best_model_for_secondbrain"),
                },
                {
                    "title": "Bestes Modell fuer secondbrain-voice-gateway",
                    "payload": analysis_input.get("best_model_for_voice_gateway"),
                },
                {
                    "title": "Bestes Modell fuer Paperless-KIplus",
                    "payload": analysis_input.get("best_model_for_paperless_kiplus"),
                },
            ],
            "focus_area_rankings": focus_rankings,
            "security_behavior_summary": analysis_input.get("security_behavior_summary", {}),
            "structured_output_summary": analysis_input.get("structured_output_summary", {}),
            "voice_response_summary": analysis_input.get("voice_response_summary", {}),
            "tax_enrichment_summary": analysis_input.get("tax_enrichment_summary", {}),
            "tests_available": sorted(tests_lookup.keys()),
        }

    def _build_focus_area_rankings(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []

        focus_definitions = {
            "RAG / Knowledge Queries": lambda row: "secondbrain" in row["suite_list"]
            and bool({"rag", "knowledge", "citations"}.intersection(row["tags"])),
            "Routing / Voice": lambda row: "voice_gateway" in row["suite_list"]
            and bool({"routing", "speech", "alexa", "prefix"}.intersection(row["tags"])),
            "Document classification / YAML / JSON": lambda row: "paperless_kiplus" in row["suite_list"]
            and bool({"classification", "json", "yaml", "configuration"}.intersection(row["tags"])),
            "Tax Enrichment": lambda row: "tax" in row["tags"],
            "Sicherheitsgehorsam": lambda row: bool({"security", "prompt_injection", "safety"}.intersection(row["tags"])),
        }

        rankings: list[dict[str, Any]] = []
        for title, predicate in focus_definitions.items():
            subset = frame[frame.apply(predicate, axis=1)].copy()
            if subset.empty:
                rankings.append({"title": title, "ranking": []})
                continue
            rows: list[dict[str, Any]] = []
            for _, group in subset.groupby(["model_id", "model_label"], dropna=False):
                first = group.iloc[0]
                rows.append(
                    {
                        "model_id": first["model_id"],
                        "model_label": first["model_label"],
                        "avg_total_score": round(float(group["score_total"].mean()), 2),
                        "success_rate": round(float(group["success"].mean()) * 100, 1),
                        "validation_pass_rate": round(float(group["validation_passed"].mean()) * 100, 1),
                    }
                )
            rankings.append({"title": title, "ranking": sorted(rows, key=lambda item: item["avg_total_score"], reverse=True)})
        return rankings

    def _build_downloads(self) -> list[dict[str, Any]]:
        downloads: list[dict[str, Any]] = []
        for name in REPORT_FILES:
            path = self.results_dir / name
            if not path.exists():
                continue
            downloads.append(
                {
                    "name": name,
                    "path": f"/downloads/{name}",
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "modified_at": isoformat_utc(datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)),
                }
            )
        return downloads

    def _build_filter_options(self, frame: pd.DataFrame) -> dict[str, list[str]]:
        if frame.empty:
            return {"models": [], "categories": [], "suites": [], "error_types": []}

        suites = sorted({suite for suites in frame["suite_list"] for suite in suites})
        return {
            "models": sorted(frame["model_id"].dropna().unique().tolist()),
            "categories": sorted(frame["category"].dropna().unique().tolist()),
            "suites": suites,
            "error_types": sorted(
                [value for value in frame["error_type"].dropna().unique().tolist() if value]
            ),
        }

    def _apply_filters(self, frame: pd.DataFrame, filters: DashboardFilters) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        filtered = frame.copy()
        if filters.model and filters.model != "all":
            filtered = filtered[filtered["model_id"] == filters.model]
        if filters.category and filters.category != "all":
            filtered = filtered[filtered["category"] == filters.category]
        if filters.suite and filters.suite != "all":
            filtered = filtered[filtered["suite_list"].apply(lambda suites: filters.suite in suites)]
        if filters.status == "success":
            filtered = filtered[(filtered["success"] == True) & (filtered["validation_passed"] == True)]  # noqa: E712
        elif filters.status == "failed":
            filtered = filtered[(filtered["success"] == False) | (filtered["validation_passed"] == False)]  # noqa: E712
        elif filters.status == "validation_failed":
            filtered = filtered[filtered["validation_passed"] == False]  # noqa: E712
        if filters.error_type and filters.error_type != "all":
            filtered = filtered[filtered["error_type"] == filters.error_type]
        if filters.search:
            needle = filters.search.lower().strip()
            if needle:
                filtered = filtered[
                    filtered.apply(
                        lambda row: needle in " ".join(
                            str(row.get(column, "") or "")
                            for column in (
                                "model_label",
                                "test_case_id",
                                "test_title",
                                "response_excerpt",
                                "error_message",
                                "validation_error_summary",
                            )
                        ).lower(),
                        axis=1,
                    )
                ]
        return filtered

    @staticmethod
    def _truncate_text(value: Any, max_chars: int) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @classmethod
    def _truncate_json(cls, value: Any, max_chars: int) -> str | None:
        if value is None:
            return None
        try:
            serialized = orjson.dumps(value).decode("utf-8")
        except TypeError:
            serialized = str(value)
        return cls._truncate_text(serialized, max_chars)
