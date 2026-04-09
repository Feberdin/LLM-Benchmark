"""
Purpose: Verify the interactive live compare manager, API wiring and history persistence.
Input/Output: Uses a temporary config plus a fake compare executor so the dashboard can be tested without real LLM endpoints.
Important invariants: Live compare requests must stay isolated per model, persist completed runs and remain queryable through the API.
How to debug: Run `pytest tests/test_live_compare_manager.py -q` and inspect the temporary `live_compare` directory when assertions fail.
"""

from __future__ import annotations

from pathlib import Path
import time

from fastapi.testclient import TestClient

from llm_benchmark.dashboard.app import create_dashboard_app
from llm_benchmark.domain.live_compare import LiveCompareModelResult, LiveCompareRunRecord, LiveCompareSummary


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
    model_name: "mistral-small3.2:latest"
    enabled: true
  - id: "qwen_local"
    label: "Qwen Local"
    provider: "local"
    base_url: "http://127.0.0.1:11434/v1"
    api_type: "openai_compatible"
    model_name: "qwen3.5:35b-a3b"
    enabled: true
  - id: "openai_reference"
    label: "OpenAI Reference"
    provider: "openai"
    base_url: "https://api.openai.com/v1"
    api_type: "openai"
    model_name: "gpt-4.1-mini"
    enabled: true
""".strip(),
        encoding="utf-8",
    )


def test_live_compare_api_can_start_track_and_reload_history(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    tests_dir = tmp_path / "tests"
    results_dir = tmp_path / "results"
    tests_dir.mkdir()
    results_dir.mkdir()
    _write_config(config_path)

    def fake_execute(*, config, request, run_id, progress_callback):
        progress_callback(
            {
                "event": "compare_started",
                "run_id": run_id,
                "mode": request.mode,
                "model_ids": request.models,
            }
        )
        for model_id, model_label in [
            ("mistral_local", "Mistral Local"),
            ("qwen_local", "Qwen Local"),
            ("openai_reference", "OpenAI Reference"),
        ]:
            progress_callback(
                {
                    "event": "compare_model_started",
                    "run_id": run_id,
                    "model_id": model_id,
                    "model_label": model_label,
                    "provider": "local" if model_id != "openai_reference" else "openai",
                    "model_name": model_id,
                    "endpoint": "http://example.test/v1",
                    "started_at": "2026-04-09T20:00:00Z",
                }
            )
        results = [
            LiveCompareModelResult(
                model_id="mistral_local",
                model_label="Mistral Local",
                provider="local",
                model_name="mistral-small3.2:latest",
                endpoint="http://example.test/v1",
                status="success",
                success=True,
                started_at="2026-04-09T20:00:00Z",
                finished_at="2026-04-09T20:00:10Z",
                duration_ms=10000.0,
                duration_human="10.00 s",
                response_text="Mistral answer",
                http_status=200,
            ),
            LiveCompareModelResult(
                model_id="qwen_local",
                model_label="Qwen Local",
                provider="local",
                model_name="qwen3.5:35b-a3b",
                endpoint="http://example.test/v1",
                status="success",
                success=True,
                started_at="2026-04-09T20:00:00Z",
                finished_at="2026-04-09T20:00:08Z",
                duration_ms=8000.0,
                duration_human="8.00 s",
                response_text="Qwen answer",
                http_status=200,
            ),
            LiveCompareModelResult(
                model_id="openai_reference",
                model_label="OpenAI Reference",
                provider="openai",
                model_name="gpt-4.1-mini",
                endpoint="https://api.openai.com/v1",
                status="success",
                success=True,
                started_at="2026-04-09T20:00:00Z",
                finished_at="2026-04-09T20:00:02Z",
                duration_ms=2000.0,
                duration_human="2.00 s",
                response_text="OpenAI answer",
                http_status=200,
            ),
        ]
        for result in results:
            progress_callback(
                {
                    "event": "compare_model_finished",
                    "run_id": run_id,
                    "result": result.model_dump(mode="json"),
                }
            )
        progress_callback(
            {
                "event": "compare_finished",
                "run_id": run_id,
                "status": "succeeded",
                "summary": {
                    "fastest_model_id": "openai_reference",
                    "fastest_model_label": "OpenAI Reference",
                    "longest_response_model_id": "mistral_local",
                    "longest_response_model_label": "Mistral Local",
                    "all_successful": True,
                    "successful_model_count": 3,
                    "failed_model_count": 0,
                },
            }
        )
        return LiveCompareRunRecord(
            run_id=run_id,
            status="succeeded",
            created_at="2026-04-09T20:00:00Z",
            finished_at="2026-04-09T20:00:10Z",
            question=request.question,
            system_prompt=request.system_prompt,
            mode=request.mode,
            selected_models=request.models,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            results=results,
            summary=LiveCompareSummary(
                fastest_model_id="openai_reference",
                fastest_model_label="OpenAI Reference",
                longest_response_model_id="mistral_local",
                longest_response_model_label="Mistral Local",
                all_successful=True,
                successful_model_count=3,
                failed_model_count=0,
            ),
        )

    monkeypatch.setattr(
        "llm_benchmark.dashboard.live_compare_manager.execute_live_compare_sync",
        fake_execute,
    )

    app = create_dashboard_app(config_path=config_path, results_dir=results_dir, tests_dir=tests_dir)

    with TestClient(app) as client:
        page = client.get("/live-compare")
        assert page.status_code == 200
        assert "Live Compare" in page.text

        start_response = client.post(
            "/api/dashboard/live-compare",
            json={
                "question": "Vergleiche diese Antwort bitte kurz.",
                "models": ["mistral_local", "qwen_local", "openai_reference"],
                "mode": "chat",
                "max_tokens": 300,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        )
        assert start_response.status_code == 202

        deadline = time.time() + 3.0
        current_payload = None
        while time.time() < deadline:
            current_response = client.get("/api/dashboard/live-compare/current")
            assert current_response.status_code == 200
            current_payload = current_response.json()
            if current_payload["status"] == "succeeded":
                break
            time.sleep(0.05)

        assert current_payload is not None
        assert current_payload["status"] == "succeeded"
        assert current_payload["summary"]["fastest_model_id"] == "openai_reference"
        assert len(current_payload["results"]) == 3

        history_response = client.get("/api/dashboard/live-compare/history")
        assert history_response.status_code == 200
        history_payload = history_response.json()
        assert history_payload[0]["run_id"] == current_payload["run_id"]

        run_response = client.get(f"/api/dashboard/live-compare/{current_payload['run_id']}")
        assert run_response.status_code == 200
        run_payload = run_response.json()
        assert run_payload["question"] == "Vergleiche diese Antwort bitte kurz."
        assert run_payload["results"][1]["model_id"] == "qwen_local"
