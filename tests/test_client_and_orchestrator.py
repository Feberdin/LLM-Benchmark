"""
Purpose: Regression tests for client normalization edge cases and benchmark timing behavior.
Input/Output: Uses tiny in-memory payloads plus a deterministic async stub to keep tests fast and readable.
Important invariants: Empty assistant content must not look like a successful answer, and duration_ms must measure request time rather than semaphore wait time.
How to debug: Run `pytest tests/test_client_and_orchestrator.py -q` and inspect the failing assertion together with the provider payload or queue wait metadata.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from llm_benchmark.clients.openai_compatible import BenchmarkClientError, OpenAICompatibleClient
from llm_benchmark.config.models import BenchmarkConfig
from llm_benchmark.domain.result import UnifiedResponse
from llm_benchmark.domain.test_case import TestCaseDefinition
from llm_benchmark.runner.orchestrator import BenchmarkOrchestrator


def test_standard_client_rejects_reasoning_only_payload() -> None:
    """
    Why this exists:
    Local thinking-capable models can answer with empty assistant content while putting all emitted tokens
    into a vendor-specific reasoning field. We want to classify that explicitly instead of pretending the
    benchmark received a usable answer.

    Example payload:
    - content: ""
    - reasoning: "Thinking Process: ..."
    - finish_reason: "length"
    """

    client = OpenAICompatibleClient(max_retries=0, retry_backoff_seconds=0.1, user_agent="unit-test")
    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning": "Thinking Process: the model spent its budget before the final answer.",
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 32},
        },
    )

    with pytest.raises(BenchmarkClientError) as exc_info:
        client._normalize_standard_response(response)

    assert exc_info.value.error_type == "empty_assistant_content"
    assert "reasoning text" in str(exc_info.value)
    assert exc_info.value.payload is not None
    assert exc_info.value.payload["reasoning_excerpt"].startswith("Thinking Process:")


def test_orchestrator_duration_excludes_semaphore_wait_time() -> None:
    """
    Why this exists:
    The benchmark runner limits concurrency with a semaphore. Duration_ms must describe only the actual
    backend request duration, while queue_wait_ms keeps the waiting time visible for debugging.

    Example:
    - request A sleeps for ~120 ms
    - request B waits behind A and then also sleeps for ~120 ms
    - both duration_ms values should stay near 120 ms, while only B gets a notable queue_wait_ms
    """

    config = BenchmarkConfig.model_validate(
        {
            "models": [
                {
                    "id": "local-model",
                    "label": "Local Model",
                    "provider": "local",
                    "base_url": "http://localhost:11434/v1",
                    "model_name": "dummy",
                }
            ],
            "run_defaults": {
                "concurrency": 1,
                "warmup_runs": 0,
                "default_repetitions": 1,
                "capture_raw_response_text": True,
            },
        }
    )
    test_case = TestCaseDefinition.model_validate(
        {
            "test_case_id": "timing-check",
            "category": "chat",
            "title": "Timing check",
            "description": "Minimal text case that keeps validation neutral for duration tests.",
            "prompt": "Reply with OK.",
            "expected_format": "text",
        }
    )

    class SlowStubClient:
        async def execute(self, **_: object) -> tuple[UnifiedResponse, int]:
            await asyncio.sleep(0.12)
            return (
                UnifiedResponse(
                    http_status=200,
                    raw_payload={"choices": []},
                    raw_response_text="OK",
                    finish_reason="stop",
                    input_tokens=5,
                    output_tokens=1,
                ),
                0,
            )

    async def run_pair() -> list:
        orchestrator = BenchmarkOrchestrator(config)
        semaphore = asyncio.Semaphore(1)
        client = SlowStubClient()
        model_config = config.models[0]
        first = asyncio.create_task(
            orchestrator._execute_single_run(
                client=client,
                semaphore=semaphore,
                benchmark_run_id="run-1",
                model_config=model_config,
                test_case=test_case,
                repetition_index=1,
                timeout_seconds=5.0,
                phase="cold",
            )
        )
        second = asyncio.create_task(
            orchestrator._execute_single_run(
                client=client,
                semaphore=semaphore,
                benchmark_run_id="run-1",
                model_config=model_config,
                test_case=test_case,
                repetition_index=2,
                timeout_seconds=5.0,
                phase="warm",
            )
        )
        return sorted(await asyncio.gather(first, second), key=lambda item: item.repetition_index)

    first_result, second_result = asyncio.run(run_pair())

    assert first_result.success is True
    assert second_result.success is True
    assert first_result.duration_ms < 170.0
    assert second_result.duration_ms < 170.0
    assert first_result.metadata.get("queue_wait_ms", 0.0) < 40.0
    assert second_result.metadata.get("queue_wait_ms", 0.0) >= 60.0

