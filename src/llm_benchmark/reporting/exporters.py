"""
Purpose: Persist raw benchmark outputs and aggregated reports to disk in analysis-friendly formats.
Input/Output: Writes JSONL, CSV, Markdown, HTML and final JSON files into the configured results directory.
Important invariants: Raw run order is preserved and complex nested fields stay JSON-encoded in CSV exports.
How to debug: If a file is missing or malformed, inspect this module before changing the report builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from llm_benchmark.domain.result import RunResult
from llm_benchmark.reporting.builder import ReportArtifacts
from llm_benchmark.utils import ensure_directory, json_dump_to_path, json_dumps

HISTORY_DIR_NAME = "history"


def write_raw_results(results: list[RunResult], output_dir: Path) -> tuple[Path, Path]:
    """Write normalized raw runs to JSONL and CSV."""

    ensure_directory(output_dir)
    jsonl_path = output_dir / "raw_runs.jsonl"
    csv_path = output_dir / "raw_runs.csv"

    with jsonl_path.open("wb") as handle:
        for result in results:
            handle.write(json_dumps(result.model_dump(mode="json")))
            handle.write(b"\n")

    frame = pd.DataFrame.from_records([_result_to_csv_record(result) for result in results])
    frame.to_csv(csv_path, index=False)

    benchmark_run_id = _infer_benchmark_run_id_from_results(results)
    if benchmark_run_id:
        history_dir = _history_run_dir(output_dir, benchmark_run_id)
        history_jsonl_path = history_dir / "raw_runs.jsonl"
        history_csv_path = history_dir / "raw_runs.csv"
        history_jsonl_path.write_bytes(jsonl_path.read_bytes())
        history_csv_path.write_bytes(csv_path.read_bytes())

    return jsonl_path, csv_path


def write_report_artifacts(artifacts: ReportArtifacts, output_dir: Path) -> dict[str, Path]:
    """Write aggregate CSV, JSON, Markdown and HTML artifacts."""

    ensure_directory(output_dir)
    summary_by_model_path = output_dir / "summary_by_model.csv"
    summary_by_category_path = output_dir / "summary_by_category.csv"
    final_report_json_path = output_dir / "final_report.json"
    analysis_input_json_path = output_dir / "analysis_input.json"
    final_report_md_path = output_dir / "final_report.md"
    final_report_html_path = output_dir / "final_report.html"

    artifacts.summary_by_model.to_csv(summary_by_model_path, index=False)
    artifacts.summary_by_category.to_csv(summary_by_category_path, index=False)
    json_dump_to_path(final_report_json_path, artifacts.final_report, pretty=True)
    json_dump_to_path(analysis_input_json_path, artifacts.analysis_input, pretty=True)
    final_report_md_path.write_text(artifacts.markdown_report, encoding="utf-8")
    final_report_html_path.write_text(artifacts.html_report, encoding="utf-8")

    benchmark_run_id = _infer_benchmark_run_id_from_artifacts(artifacts)
    if benchmark_run_id:
        history_dir = _history_run_dir(output_dir, benchmark_run_id)
        (history_dir / "summary_by_model.csv").write_bytes(summary_by_model_path.read_bytes())
        (history_dir / "summary_by_category.csv").write_bytes(summary_by_category_path.read_bytes())
        (history_dir / "final_report.json").write_bytes(final_report_json_path.read_bytes())
        (history_dir / "analysis_input.json").write_bytes(analysis_input_json_path.read_bytes())
        (history_dir / "final_report.md").write_text(artifacts.markdown_report, encoding="utf-8")
        (history_dir / "final_report.html").write_text(artifacts.html_report, encoding="utf-8")

    return {
        "summary_by_model_csv": summary_by_model_path,
        "summary_by_category_csv": summary_by_category_path,
        "final_report_json": final_report_json_path,
        "analysis_input_json": analysis_input_json_path,
        "final_report_md": final_report_md_path,
        "final_report_html": final_report_html_path,
    }


def load_results_from_jsonl(path: Path) -> list[RunResult]:
    """Load raw runs back from the JSONL export."""

    results: list[RunResult] = []
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        results.append(RunResult.model_validate_json(line))
    return results


def _result_to_csv_record(result: RunResult) -> dict[str, Any]:
    record = result.model_dump(mode="json")
    record["parsed_output_json"] = _stringify_json(record.get("parsed_output_json"))
    record["validation_errors"] = _stringify_json(record.get("validation_errors"))
    record["score_breakdown"] = _stringify_json(record.get("score_breakdown"))
    record["metadata"] = _stringify_json(record.get("metadata"))
    return record


def _stringify_json(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json_dumps(value).decode("utf-8")


def _history_run_dir(output_dir: Path, benchmark_run_id: str) -> Path:
    """Return the history directory for one persisted benchmark run."""

    return ensure_directory(output_dir / HISTORY_DIR_NAME / benchmark_run_id)


def _infer_benchmark_run_id_from_results(results: list[RunResult]) -> str | None:
    """Extract the run id from the first raw result when available."""

    if not results:
        return None
    return results[0].benchmark_run_id


def _infer_benchmark_run_id_from_artifacts(artifacts: ReportArtifacts) -> str | None:
    """Extract the run id from the final report payload for history snapshots."""

    benchmark_info = artifacts.final_report.get("benchmark_info", {})
    benchmark_run_id = benchmark_info.get("benchmark_run_id")
    return str(benchmark_run_id) if benchmark_run_id else None
