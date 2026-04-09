"""
Purpose: Verify fair serial execution and stable timing metadata for the live compare runner.
Input/Output: Uses a patched compare worker to observe ordering without touching real endpoints.
Important invariants: Serial mode must execute models strictly one after another, while keeping execution order explicit in the final record.
How to debug: Run `pytest tests/test_live_compare_runner.py -q` and inspect the recorded call trace when the order assertion fails.
"""

from __future__ import annotations

import asyncio

from llm_benchmark.config.models import BenchmarkConfig
from llm_benchmark.domain.live_compare import LiveCompareModelResult, LiveCompareRequest
from llm_benchmark.runner import live_compare as live_compare_module


def _build_config() -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "models": [
                {
                    "id": "mistral_local",
                    "label": "Mistral Local",
                    "provider": "local",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "api_type": "openai_compatible",
                    "model_name": "mistral-small3.2:latest",
                },
                {
                    "id": "qwen_local",
                    "label": "Qwen Local",
                    "provider": "local",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "api_type": "openai_compatible",
                    "model_name": "qwen3.5:35b-a3b",
                },
                {
                    "id": "openai_reference",
                    "label": "OpenAI Reference",
                    "provider": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "api_type": "openai",
                    "model_name": "gpt-4.1-mini",
                },
            ]
        }
    )


def test_live_compare_serial_execution_preserves_fair_order(monkeypatch) -> None:
    config = _build_config()
    request = LiveCompareRequest.model_validate(
        {
            "question": "Vergleiche bitte die Antwortqualitaet.",
            "models": ["openai_reference", "qwen_local", "mistral_local"],
            "mode": "chat",
            "execution_mode": "serial",
            "max_tokens": 200,
        }
    )
    call_trace: list[tuple[str, str]] = []

    async def fake_execute_single_compare(
        *,
        client,
        config,
        model_config,
        request,
        progress_callback,
        run_id,
        run_started_monotonic,
    ):
        call_trace.append(("start", model_config.id))
        await asyncio.sleep(0.01)
        call_trace.append(("finish", model_config.id))
        return LiveCompareModelResult(
            model_id=model_config.id,
            model_label=model_config.label,
            provider=model_config.provider,
            model_name=model_config.model_name,
            endpoint=model_config.base_url,
            status="finished",
            success=True,
            execution_start_at="2026-04-09T20:00:00Z",
            execution_end_at="2026-04-09T20:00:01Z",
            isolated_duration_ms=1000.0,
            total_elapsed_since_run_start_ms=1000.0,
            duration_ms=1000.0,
            duration_human="1.00 s",
            response_text=f"Antwort von {model_config.label}",
        )

    monkeypatch.setattr(live_compare_module, "_execute_single_compare", fake_execute_single_compare)

    record = live_compare_module.execute_live_compare_sync(config=config, request=request, run_id="livecmp_test")

    assert record.execution_mode == "serial"
    assert record.execution_order == ["mistral_local", "qwen_local", "openai_reference"]
    assert call_trace == [
        ("start", "mistral_local"),
        ("finish", "mistral_local"),
        ("start", "qwen_local"),
        ("finish", "qwen_local"),
        ("start", "openai_reference"),
        ("finish", "openai_reference"),
    ]
