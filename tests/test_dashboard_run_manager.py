"""
Purpose: Verify dashboard-triggered benchmark execution and actionable preflight feedback.
Input/Output: Uses temporary config/test directories plus a fake benchmark execution to validate run manager behavior.
Important invariants: Empty Unraid test mounts must fail clearly, and background dashboard runs must expose status updates.
How to debug: Run `pytest tests/test_dashboard_run_manager.py -q` and inspect the temp results directory when a state assertion fails.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time

from fastapi.testclient import TestClient

from llm_benchmark.dashboard.app import create_dashboard_app
from llm_benchmark.dashboard.run_manager import ConnectivityResult, DashboardRunManager


def _write_config(path: Path) -> None:
    path.write_text(
        """
version: "1"
benchmark_name: "dashboard-test"
models:
  - id: "mistral_local"
    label: "Mistral Local"
    provider: "local"
    base_url: "http://127.0.0.1:11434/v1"
    api_type: "openai_compatible"
    model_name: "mistral-small3.2"
    enabled: true
""".strip(),
        encoding="utf-8",
    )


def _write_test_case(path: Path) -> None:
    path.write_text(
        """
test_case_id: "quick-chat-triage"
category: "chat"
title: "Quick operational triage answer"
description: "Short dashboard smoke test."
prompt: "Antworte kurz mit genau einem Satz."
expected_format: "text"
tags:
  - "quick_compare"
  - "chat"
suites:
  - "quick_compare"
""".strip(),
        encoding="utf-8",
    )


def test_run_manager_preflight_detects_empty_unraid_tests_mount(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    tests_dir = tmp_path / "tests"
    results_dir = tmp_path / "results"
    tests_dir.mkdir()
    results_dir.mkdir()
    _write_config(config_path)

    manager = DashboardRunManager(config_path=config_path, tests_dir=tests_dir, results_dir=results_dir)
    preflight = manager.preflight(suite="quick_compare")

    assert preflight["errors"]
    assert "leeres Unraid-Mount" in " ".join(preflight["errors"])


def test_dashboard_api_can_start_and_track_background_run(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    tests_dir = tmp_path / "tests"
    results_dir = tmp_path / "results"
    tests_dir.mkdir()
    results_dir.mkdir()
    _write_config(config_path)
    _write_test_case(tests_dir / "quick_compare.yaml")

    def fake_execute_benchmark_run(*, config_path, tests_dir, results_dir, suite, progress_callback):
        assert suite == "quick_compare"
        progress_callback({"event": "loading_config", "stage": "preflight"})
        progress_callback(
            {
                "event": "test_discovery_completed",
                "stage": "preflight",
                "selected_test_case_count": 1,
                "selected_model_count": 1,
            }
        )
        progress_callback(
            {
                "event": "run_started",
                "stage": "running",
                "benchmark_run_id": "bench-123",
                "suite": "quick_compare",
                "model_count": 1,
                "test_case_count": 1,
                "total_planned_records": 1,
            }
        )
        progress_callback(
            {
                "event": "run_step_started",
                "stage": "running",
                "model_label": "Mistral Local",
                "test_case_id": "quick-chat-triage",
                "phase": "cold",
                "repetition_index": 1,
            }
        )
        progress_callback(
            {
                "event": "run_step_finished",
                "stage": "running",
                "model_id": "mistral_local",
                "model_label": "Mistral Local",
                "test_case_id": "quick-chat-triage",
                "category": "chat",
                "phase": "cold",
                "repetition_index": 1,
                "success": True,
                "validation_passed": True,
                "duration_ms": 123.0,
                "score_total": 98.5,
            }
        )
        progress_callback(
            {
                "event": "run_finished",
                "stage": "running",
                "benchmark_run_id": "bench-123",
                "total_records": 1,
                "successful_records": 1,
                "failed_records": 0,
            }
        )
        progress_callback({"event": "writing_raw_results", "stage": "writing_reports"})
        (results_dir / "raw_runs.jsonl").write_text("{}", encoding="utf-8")
        progress_callback({"event": "building_reports", "stage": "writing_reports"})
        (results_dir / "final_report.json").write_text("{}", encoding="utf-8")
        (results_dir / "analysis_input.json").write_text("{}", encoding="utf-8")
        history_dir = results_dir / "history" / "bench-123"
        history_dir.mkdir(parents=True, exist_ok=True)
        (history_dir / "raw_runs.jsonl").write_text("{}", encoding="utf-8")
        (history_dir / "final_report.json").write_text("{}", encoding="utf-8")
        (history_dir / "analysis_input.json").write_text("{}", encoding="utf-8")
        progress_callback(
            {
                "event": "reports_written",
                "stage": "completed",
                "benchmark_run_id": "bench-123",
                "results_dir": str(results_dir),
            }
        )
        return SimpleNamespace(run_summary=SimpleNamespace(benchmark_run_id="bench-123"))

    monkeypatch.setattr(
        "llm_benchmark.dashboard.run_manager.execute_benchmark_run",
        fake_execute_benchmark_run,
    )

    app = create_dashboard_app(config_path=config_path, results_dir=results_dir, tests_dir=tests_dir)

    with TestClient(app) as client:
        start_response = client.post("/api/dashboard/run/start?suite=quick_compare")
        assert start_response.status_code == 202

        deadline = time.time() + 3.0
        current_payload = None
        while time.time() < deadline:
            current_response = client.get("/api/dashboard/run/current?suite=quick_compare")
            assert current_response.status_code == 200
            current_payload = current_response.json()
            if current_payload["state"]["status"] == "succeeded":
                break
            time.sleep(0.05)

        assert current_payload is not None
        assert current_payload["state"]["status"] == "succeeded"
        assert current_payload["state"]["benchmark_run_id"] == "bench-123"
        assert "final_report.json" in current_payload["state"]["generated_files"]
        assert current_payload["state"]["generated_downloads"][0]["path"].startswith("/downloads/history/bench-123/")
        assert current_payload["history"][0]["status"] == "succeeded"
        assert current_payload["history"][0]["artifact_downloads"][0]["path"].startswith("/downloads/history/bench-123/")

        download_response = client.get("/downloads/history/bench-123/analysis_input.json")
        assert download_response.status_code == 200
        assert download_response.text == "{}"


def test_dashboard_api_can_run_connectivity_check(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    tests_dir = tmp_path / "tests"
    results_dir = tmp_path / "results"
    tests_dir.mkdir()
    results_dir.mkdir()
    _write_config(config_path)
    _write_test_case(tests_dir / "quick_compare.yaml")

    def fake_probe(self, *, model_config, default_timeout_seconds):
        assert model_config.id == "mistral_local"
        return ConnectivityResult(
            model_id=model_config.id,
            model_label=model_config.label,
            provider=model_config.provider,
            endpoint=model_config.base_url,
            model_name=model_config.model_name,
            ok=True,
            status="reachable",
            message="Endpoint und Modell sind erreichbar.",
            http_status=200,
            duration_ms=42.0,
            model_listed=True,
            listed_models_preview=[model_config.model_name],
        )

    monkeypatch.setattr(DashboardRunManager, "_probe_model_endpoint", fake_probe)

    app = create_dashboard_app(config_path=config_path, results_dir=results_dir, tests_dir=tests_dir)

    with TestClient(app) as client:
        start_response = client.post("/api/dashboard/connectivity/check")
        assert start_response.status_code == 202

        deadline = time.time() + 3.0
        current_payload = None
        while time.time() < deadline:
            current_response = client.get("/api/dashboard/connectivity/current")
            assert current_response.status_code == 200
            current_payload = current_response.json()
            if current_payload["status"] == "succeeded":
                break
            time.sleep(0.05)

        assert current_payload is not None
        assert current_payload["status"] == "succeeded"
        assert current_payload["reachable_models"] == 1
        assert current_payload["failed_models"] == 0
        assert current_payload["results"][0]["status"] == "reachable"
