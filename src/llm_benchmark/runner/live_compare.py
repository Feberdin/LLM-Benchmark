"""
Purpose: Execute one interactive live-compare prompt against multiple configured models in parallel.
Input/Output: Consumes the existing benchmark config plus a live compare request and returns one structured result per selected model.
Important invariants: The same OpenAI-compatible clients are reused, one failed model never aborts the others and persisted outputs stay machine-readable.
How to debug: Start with the per-model `error_type`, `http_status` and `technical_details` fields before changing any dashboard UI.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from time import perf_counter
from typing import Any, Callable

import httpx
import orjson

from llm_benchmark.clients.openai_compatible import BenchmarkClientError, OpenAICompatibleClient
from llm_benchmark.config.models import BenchmarkConfig, BenchmarkModelConfig
from llm_benchmark.domain.live_compare import (
    LiveCompareEvent,
    LiveCompareModelResult,
    LiveCompareRequest,
    LiveCompareRunRecord,
    LiveCompareSummary,
)
from llm_benchmark.domain.test_case import TestCaseDefinition, ValidationRules
from llm_benchmark.utils import isoformat_utc, utcnow
from llm_benchmark.validation.service import ResponseValidator

LOGGER = logging.getLogger(__name__)
LiveCompareProgressCallback = Callable[[dict[str, Any]], None]


def execute_live_compare_sync(
    *,
    config: BenchmarkConfig,
    request: LiveCompareRequest,
    run_id: str | None = None,
    progress_callback: LiveCompareProgressCallback | None = None,
) -> LiveCompareRunRecord:
    """Synchronous wrapper for thread-based dashboard execution."""

    return asyncio.run(
        execute_live_compare(config=config, request=request, run_id=run_id, progress_callback=progress_callback)
    )


async def execute_live_compare(
    *,
    config: BenchmarkConfig,
    request: LiveCompareRequest,
    run_id: str | None = None,
    progress_callback: LiveCompareProgressCallback | None = None,
) -> LiveCompareRunRecord:
    """
    Run the same prompt against multiple models in parallel.

    Why this exists:
    The dashboard should compare real model behaviour with the same transport layer and config handling as the
    benchmark engine, but without forcing the operator through a full test suite.
    """

    enabled_models = {model.id: model for model in config.enabled_models()}
    if not enabled_models:
        raise ValueError("No enabled models are available for live compare.")

    selected_models = _select_models(enabled_models, request.models)
    effective_run_id = run_id or f"livecmp_{uuid.uuid4().hex[:12]}"
    created_at = isoformat_utc()

    _emit(
        progress_callback,
        event="compare_started",
        run_id=effective_run_id,
        created_at=created_at,
        mode=request.mode,
        model_ids=[model.id for model in selected_models],
    )

    async with OpenAICompatibleClient(
        max_retries=config.run_defaults.max_retries,
        retry_backoff_seconds=config.run_defaults.retry_backoff_seconds,
        user_agent=f"{config.benchmark_name}/0.1.0",
    ) as client:
        tasks = [
            asyncio.create_task(
                _execute_single_compare(
                    client=client,
                    config=config,
                    model_config=model_config,
                    request=request,
                    progress_callback=progress_callback,
                    run_id=effective_run_id,
                )
            )
            for model_config in selected_models
        ]
        results = await asyncio.gather(*tasks)

    summary = _build_summary(results)
    finished_at = isoformat_utc()
    status = "succeeded" if summary.all_successful else "partial"
    events = [
        LiveCompareEvent(
            timestamp=finished_at,
            event="compare_finished",
            level="success" if summary.all_successful else "warning",
            summary=(
                "Live Compare erfolgreich abgeschlossen."
                if summary.all_successful
                else "Live Compare mit gemischten Ergebnissen abgeschlossen."
            ),
            details=summary.model_dump(mode="json"),
        )
    ]
    _emit(
        progress_callback,
        event="compare_finished",
        run_id=effective_run_id,
        finished_at=finished_at,
        status=status,
        summary=summary.model_dump(mode="json"),
    )

    return LiveCompareRunRecord(
        run_id=effective_run_id,
        status=status,
        created_at=created_at,
        finished_at=finished_at,
        question=request.question,
        system_prompt=request.system_prompt,
        mode=request.mode,
        selected_models=[model.id for model in selected_models],
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        results=results,
        summary=summary,
        latest_error=next((row.error_message for row in results if row.error_message), None),
        events=events,
    )


async def _execute_single_compare(
    *,
    client: OpenAICompatibleClient,
    config: BenchmarkConfig,
    model_config: BenchmarkModelConfig,
    request: LiveCompareRequest,
    progress_callback: LiveCompareProgressCallback | None,
    run_id: str,
) -> LiveCompareModelResult:
    """Execute one live compare request for one model and normalize the response."""

    test_case = _build_live_compare_test_case(request)
    validator = ResponseValidator()
    started_at = isoformat_utc()
    started_monotonic = perf_counter()

    _emit(
        progress_callback,
        event="compare_model_started",
        run_id=run_id,
        model_id=model_config.id,
        model_label=model_config.label,
        provider=model_config.provider,
        model_name=model_config.model_name,
        endpoint=model_config.base_url,
        started_at=started_at,
    )

    try:
        response, retries_used = await client.execute(
            model_config=model_config,
            test_case=test_case,
            timeout_seconds=model_config.effective_timeout(config.run_defaults.default_timeout_seconds),
            stream_for_ttft=config.run_defaults.stream_for_ttft,
        )
        duration_ms = round((perf_counter() - started_monotonic) * 1000, 2)
        validation_passed, response_json = _validate_live_compare_response(
            request=request,
            test_case=test_case,
            validator=validator,
            response=response,
        )
        result = LiveCompareModelResult(
            model_id=model_config.id,
            model_label=model_config.label,
            provider=model_config.provider,
            model_name=model_config.model_name,
            endpoint=model_config.base_url,
            status="success",
            success=True,
            started_at=started_at,
            finished_at=isoformat_utc(),
            duration_ms=duration_ms,
            duration_human=_humanize_duration_ms(duration_ms),
            ttft_ms=response.ttft_ms,
            tokens_per_second=_calculate_tokens_per_second(response.output_tokens, duration_ms),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            response_text=response.raw_response_text,
            response_json=response_json,
            validation_passed=validation_passed,
            http_status=response.http_status,
            retries=retries_used,
            finish_reason=response.finish_reason,
        )
        _emit(
            progress_callback,
            event="compare_model_finished",
            run_id=run_id,
            result=result.model_dump(mode="json"),
        )
        return result
    except httpx.TimeoutException:
        result = _build_failed_result(
            model_config=model_config,
            started_at=started_at,
            started_monotonic=started_monotonic,
            error_type="timeout",
            error_message=(
                "Modellantwort hat das konfigurierte Timeout ueberschritten. "
                "Erhoehe den Timeout oder pruefe die lokale CPU-/Modelllast."
            ),
            http_status=None,
        )
    except BenchmarkClientError as exc:
        result = _build_failed_result(
            model_config=model_config,
            started_at=started_at,
            started_monotonic=started_monotonic,
            error_type=exc.error_type,
            error_message=str(exc),
            http_status=exc.http_status,
            technical_details=exc.payload or {},
        )
    except httpx.TransportError as exc:
        result = _build_failed_result(
            model_config=model_config,
            started_at=started_at,
            started_monotonic=started_monotonic,
            error_type="network_error",
            error_message=f"Netzwerkfehler beim Aufruf von {model_config.base_url}: {exc}",
            http_status=None,
        )
    except Exception as exc:  # pragma: no cover - guard rail for unexpected provider behaviour
        LOGGER.exception("Live compare failed for model=%s", model_config.id)
        result = _build_failed_result(
            model_config=model_config,
            started_at=started_at,
            started_monotonic=started_monotonic,
            error_type="unexpected_error",
            error_message=str(exc),
            http_status=None,
        )

    _emit(
        progress_callback,
        event="compare_model_finished",
        run_id=run_id,
        result=result.model_dump(mode="json"),
    )
    return result


def _build_live_compare_test_case(request: LiveCompareRequest) -> TestCaseDefinition:
    """Translate the UI request into the same test model used by the benchmark runner."""

    mode_to_category = {
        "chat": "chat",
        "json": "extraction_json",
        "technical": "coding",
        "summarization": "summarization",
    }
    mode_system_prefix = {
        "chat": "Answer the user directly and stay concise.",
        "json": "Return valid JSON only. Do not wrap the answer in Markdown.",
        "technical": "Provide a technically precise answer. Use code blocks only when they help the comparison.",
        "summarization": "Provide a compact summary that focuses on the most important information.",
    }
    combined_system_prompt = "\n\n".join(
        [text for text in [mode_system_prefix[request.mode], request.system_prompt] if text]
    )
    validation_rules = ValidationRules()
    expected_format = "text"
    if request.mode == "json":
        expected_format = "json"

    return TestCaseDefinition(
        test_case_id=f"live-compare-{request.mode}",
        category=mode_to_category[request.mode],  # type: ignore[arg-type]
        title="Live Compare Prompt",
        description="Interactive dashboard compare request.",
        prompt=request.question,
        system_prompt=combined_system_prompt or None,
        expected_format=expected_format,  # type: ignore[arg-type]
        validation_rules=validation_rules,
        tags=["live_compare", request.mode],
        suites=["live_compare"],
        max_output_tokens=request.max_tokens,
        repetitions_default=1,
        request_overrides={
            "temperature": request.temperature,
            "top_p": request.top_p,
        },
    )


def _validate_live_compare_response(
    *,
    request: LiveCompareRequest,
    test_case: TestCaseDefinition,
    validator: ResponseValidator,
    response: Any,
) -> tuple[bool | None, Any | None]:
    """Apply lightweight validation only when the selected compare mode benefits from it."""

    if request.mode == "json":
        outcome = validator.validate(test_case, response)
        return outcome.passed, outcome.parsed_output_json
    return None, _attempt_parse_json(response.raw_response_text)


def _attempt_parse_json(text: str | None) -> Any | None:
    """Best-effort JSON extraction for the optional JSON preview button in the UI."""

    if not text:
        return None
    stripped = text.strip()
    candidates = [stripped]
    if "```" in stripped:
        fence_start = stripped.find("```")
        fence_end = stripped.rfind("```")
        if fence_start != -1 and fence_end > fence_start:
            fenced = stripped[fence_start + 3 : fence_end].strip()
            if fenced.startswith("json"):
                fenced = fenced[4:].strip()
            candidates.append(fenced)
    brace_start = stripped.find("{")
    brace_end = stripped.rfind("}")
    if 0 <= brace_start < brace_end:
        candidates.append(stripped[brace_start : brace_end + 1])
    bracket_start = stripped.find("[")
    bracket_end = stripped.rfind("]")
    if 0 <= bracket_start < bracket_end:
        candidates.append(stripped[bracket_start : bracket_end + 1])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen or not candidate:
            continue
        seen.add(candidate)
        try:
            return orjson.loads(candidate)
        except orjson.JSONDecodeError:
            continue
    return None


def _build_failed_result(
    *,
    model_config: BenchmarkModelConfig,
    started_at: str,
    started_monotonic: float,
    error_type: str,
    error_message: str,
    http_status: int | None,
    technical_details: dict[str, Any] | None = None,
) -> LiveCompareModelResult:
    duration_ms = round((perf_counter() - started_monotonic) * 1000, 2)
    return LiveCompareModelResult(
        model_id=model_config.id,
        model_label=model_config.label,
        provider=model_config.provider,
        model_name=model_config.model_name,
        endpoint=model_config.base_url,
        status="failed",
        success=False,
        started_at=started_at,
        finished_at=isoformat_utc(),
        duration_ms=duration_ms,
        duration_human=_humanize_duration_ms(duration_ms),
        validation_passed=None,
        http_status=http_status,
        error_type=error_type,
        error_message=error_message,
        technical_details=technical_details or {},
    )


def _select_models(
    enabled_models: dict[str, BenchmarkModelConfig],
    requested_model_ids: list[str],
) -> list[BenchmarkModelConfig]:
    """Resolve selected models from ids and keep the original request order."""

    if not requested_model_ids:
        preferred_order = ["mistral_local", "qwen_local", "openai_reference"]
        requested_model_ids = [model_id for model_id in preferred_order if model_id in enabled_models]
        if len(requested_model_ids) < 2:
            requested_model_ids = list(enabled_models.keys())[:3]

    missing = [model_id for model_id in requested_model_ids if model_id not in enabled_models]
    if missing:
        raise ValueError(
            "Unknown or disabled live compare model ids: " + ", ".join(sorted(missing))
        )

    if not 2 <= len(requested_model_ids) <= 4:
        raise ValueError("Live compare currently supports between 2 and 4 models per request.")

    return [enabled_models[model_id] for model_id in requested_model_ids]


def _build_summary(results: list[LiveCompareModelResult]) -> LiveCompareSummary:
    """Generate a compact summary that is easy to surface above the comparison cards."""

    successful = [row for row in results if row.success]
    failed = [row for row in results if row.success is False]
    fastest = min(
        [row for row in successful if row.duration_ms is not None],
        key=lambda row: row.duration_ms or float("inf"),
        default=None,
    )
    longest = max(successful, key=lambda row: len(row.response_text or ""), default=None)
    return LiveCompareSummary(
        fastest_model_id=fastest.model_id if fastest else None,
        fastest_model_label=fastest.model_label if fastest else None,
        longest_response_model_id=longest.model_id if longest else None,
        longest_response_model_label=longest.model_label if longest else None,
        all_successful=len(successful) == len(results),
        successful_model_count=len(successful),
        failed_model_count=len(failed),
    )


def _calculate_tokens_per_second(output_tokens: int | None, duration_ms: float | None) -> float | None:
    if not output_tokens or not duration_ms or duration_ms <= 0:
        return None
    return round(output_tokens / (duration_ms / 1000.0), 2)


def _humanize_duration_ms(duration_ms: float | None) -> str | None:
    """Return compact duration strings that fit well into small dashboard cards."""

    if duration_ms is None:
        return None
    seconds = duration_ms / 1000.0
    if seconds < 1:
        return f"{duration_ms:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"{minutes}m {remainder:.1f}s"


def _emit(progress_callback: LiveCompareProgressCallback | None, **payload: Any) -> None:
    if progress_callback is None:
        return
    progress_callback(payload)
