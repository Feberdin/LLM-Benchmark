"""
Purpose: Execute one interactive live-compare prompt against multiple configured models with fair serial execution by default.
Input/Output: Consumes the existing benchmark config plus a live compare request and returns one structured result per selected model.
Important invariants: The same OpenAI-compatible clients are reused, CPU-bound local fairness prefers serial execution and one failed model never aborts the others.
How to debug: Start with `execution_mode`, `queue_wait_ms`, `isolated_duration_ms` and the per-model `technical_details` fields before changing the UI.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from time import perf_counter
from typing import Any, Callable

import httpx
import orjson
import yaml

from llm_benchmark.clients.openai_compatible import BenchmarkClientError, OpenAICompatibleClient
from llm_benchmark.config.models import BenchmarkConfig, BenchmarkModelConfig
from llm_benchmark.domain.live_compare import (
    CompareExecutionMode,
    LiveCompareEvent,
    LiveCompareModelResult,
    LiveCompareRequest,
    LiveCompareRunRecord,
    LiveCompareSummary,
)
from llm_benchmark.domain.test_case import TestCaseDefinition, ValidationRules
from llm_benchmark.utils import isoformat_utc
from llm_benchmark.validation.service import ResponseValidator

LOGGER = logging.getLogger(__name__)
LiveCompareProgressCallback = Callable[[dict[str, Any]], None]
DEFAULT_FAIR_ORDER = ["mistral_local", "qwen_local", "openai_reference"]
UNCERTAINTY_HINT_PATTERNS = [
    "nicht sicher",
    "unsicher",
    "ich weiss nicht",
    "ich weiß nicht",
    "soweit bekannt",
    "kann ich nicht bestaetigen",
    "kann ich nicht bestätigen",
    "ohne weitere quelle",
    "ohne weitere quelle",
]


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
    Run the same prompt against multiple models and preserve fair timing metadata.

    Why this exists:
    The dashboard should compare real model behaviour with the same transport layer and config handling as the
    benchmark engine. On CPU-bound local servers, fair latency comparisons require serial execution by default.
    """

    enabled_models = {model.id: model for model in config.enabled_models()}
    if not enabled_models:
        raise ValueError("No enabled models are available for live compare.")

    selected_models = _select_models(enabled_models, request.models)
    execution_order = [model.id for model in selected_models]
    effective_run_id = run_id or f"livecmp_{uuid.uuid4().hex[:12]}"
    created_at = isoformat_utc()
    run_started_monotonic = perf_counter()

    _emit(
        progress_callback,
        event="compare_started",
        run_id=effective_run_id,
        created_at=created_at,
        mode=request.mode,
        execution_mode=request.execution_mode,
        execution_order=execution_order,
        model_ids=execution_order,
    )

    async with OpenAICompatibleClient(
        max_retries=config.run_defaults.max_retries,
        retry_backoff_seconds=config.run_defaults.retry_backoff_seconds,
        user_agent=f"{config.benchmark_name}/0.1.0",
    ) as client:
        if request.execution_mode == "parallel":
            tasks = [
                asyncio.create_task(
                    _execute_single_compare(
                        client=client,
                        config=config,
                        model_config=model_config,
                        request=request,
                        progress_callback=progress_callback,
                        run_id=effective_run_id,
                        run_started_monotonic=run_started_monotonic,
                    )
                )
                for model_config in selected_models
            ]
            results = await asyncio.gather(*tasks)
        else:
            results = []
            for model_config in selected_models:
                results.append(
                    await _execute_single_compare(
                        client=client,
                        config=config,
                        model_config=model_config,
                        request=request,
                        progress_callback=progress_callback,
                        run_id=effective_run_id,
                        run_started_monotonic=run_started_monotonic,
                    )
                )

    summary = _build_summary(results, execution_mode=request.execution_mode, execution_order=execution_order)
    _apply_quick_badges(results, summary=summary, request=request)
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
        execution_mode=request.execution_mode,
        execution_order=execution_order,
        selected_models=execution_order,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        manual_note=request.manual_note,
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
    run_started_monotonic: float,
) -> LiveCompareModelResult:
    """Execute one live compare request for one model and normalize the response."""

    test_case = _build_live_compare_test_case(request)
    validator = ResponseValidator()
    queue_wait_ms = round((perf_counter() - run_started_monotonic) * 1000, 2)
    execution_start_at = isoformat_utc()
    request_started_monotonic = perf_counter()

    _emit(
        progress_callback,
        event="compare_model_started",
        run_id=run_id,
        model_id=model_config.id,
        model_label=model_config.label,
        provider=model_config.provider,
        model_name=model_config.model_name,
        endpoint=model_config.base_url,
        execution_start_at=execution_start_at,
        queue_wait_ms=queue_wait_ms,
    )

    try:
        response, retries_used = await client.execute(
            model_config=model_config,
            test_case=test_case,
            timeout_seconds=model_config.effective_timeout(config.run_defaults.default_timeout_seconds),
            stream_for_ttft=config.run_defaults.stream_for_ttft,
        )
        execution_end_at = isoformat_utc()
        isolated_duration_ms = round((perf_counter() - request_started_monotonic) * 1000, 2)
        total_elapsed_since_run_start_ms = round((perf_counter() - run_started_monotonic) * 1000, 2)
        validation_passed, response_json = _validate_live_compare_response(
            request=request,
            test_case=test_case,
            validator=validator,
            response=response,
        )
        response_text = response.raw_response_text
        detected_content_types = _detect_content_types(response_text, response_json)
        result = LiveCompareModelResult(
            model_id=model_config.id,
            model_label=model_config.label,
            provider=model_config.provider,
            model_name=model_config.model_name,
            endpoint=model_config.base_url,
            status="finished",
            success=True,
            started_at=execution_start_at,
            finished_at=execution_end_at,
            duration_ms=isolated_duration_ms,
            duration_human=_humanize_duration_ms(isolated_duration_ms),
            queue_wait_ms=queue_wait_ms,
            execution_start_at=execution_start_at,
            execution_end_at=execution_end_at,
            isolated_duration_ms=isolated_duration_ms,
            total_elapsed_since_run_start_ms=total_elapsed_since_run_start_ms,
            total_elapsed_human=_humanize_duration_ms(total_elapsed_since_run_start_ms),
            ttft_ms=response.ttft_ms,
            tokens_per_second=_calculate_tokens_per_second(response.output_tokens, isolated_duration_ms),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            response_text=response_text,
            response_json=response_json,
            validation_passed=validation_passed,
            http_status=response.http_status,
            retries=retries_used,
            finish_reason=response.finish_reason,
            detected_content_types=detected_content_types,
            uncertainty_marked=_detect_uncertainty_marker(request.question, response_text),
        )
        result.quick_badges = _build_intrinsic_badges(result)
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
            queue_wait_ms=queue_wait_ms,
            execution_start_at=execution_start_at,
            run_started_monotonic=run_started_monotonic,
            request_started_monotonic=request_started_monotonic,
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
            queue_wait_ms=queue_wait_ms,
            execution_start_at=execution_start_at,
            run_started_monotonic=run_started_monotonic,
            request_started_monotonic=request_started_monotonic,
            error_type=exc.error_type,
            error_message=str(exc),
            http_status=exc.http_status,
            technical_details=exc.payload or {},
        )
    except httpx.TransportError as exc:
        result = _build_failed_result(
            model_config=model_config,
            queue_wait_ms=queue_wait_ms,
            execution_start_at=execution_start_at,
            run_started_monotonic=run_started_monotonic,
            request_started_monotonic=request_started_monotonic,
            error_type="network_error",
            error_message=f"Netzwerkfehler beim Aufruf von {model_config.base_url}: {exc}",
            http_status=None,
        )
    except Exception as exc:  # pragma: no cover - guard rail for unexpected provider behaviour
        LOGGER.exception("Live compare failed for model=%s", model_config.id)
        result = _build_failed_result(
            model_config=model_config,
            queue_wait_ms=queue_wait_ms,
            execution_start_at=execution_start_at,
            run_started_monotonic=run_started_monotonic,
            request_started_monotonic=request_started_monotonic,
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
    queue_wait_ms: float,
    execution_start_at: str,
    run_started_monotonic: float,
    request_started_monotonic: float,
    error_type: str,
    error_message: str,
    http_status: int | None,
    technical_details: dict[str, Any] | None = None,
) -> LiveCompareModelResult:
    isolated_duration_ms = round((perf_counter() - request_started_monotonic) * 1000, 2)
    total_elapsed_since_run_start_ms = round((perf_counter() - run_started_monotonic) * 1000, 2)
    result = LiveCompareModelResult(
        model_id=model_config.id,
        model_label=model_config.label,
        provider=model_config.provider,
        model_name=model_config.model_name,
        endpoint=model_config.base_url,
        status="failed",
        success=False,
        started_at=execution_start_at,
        finished_at=isoformat_utc(),
        duration_ms=isolated_duration_ms,
        duration_human=_humanize_duration_ms(isolated_duration_ms),
        queue_wait_ms=queue_wait_ms,
        execution_start_at=execution_start_at,
        execution_end_at=isoformat_utc(),
        isolated_duration_ms=isolated_duration_ms,
        total_elapsed_since_run_start_ms=total_elapsed_since_run_start_ms,
        total_elapsed_human=_humanize_duration_ms(total_elapsed_since_run_start_ms),
        validation_passed=None,
        http_status=http_status,
        error_type=error_type,
        error_message=error_message,
        technical_details=technical_details or {},
    )
    result.quick_badges = _build_intrinsic_badges(result)
    return result


def _select_models(
    enabled_models: dict[str, BenchmarkModelConfig],
    requested_model_ids: list[str],
) -> list[BenchmarkModelConfig]:
    """Resolve selected models from ids and keep a stable fair execution order."""

    if not requested_model_ids:
        requested_model_ids = [model_id for model_id in DEFAULT_FAIR_ORDER if model_id in enabled_models]
        if len(requested_model_ids) < 2:
            requested_model_ids = list(enabled_models.keys())[:3]

    missing = [model_id for model_id in requested_model_ids if model_id not in enabled_models]
    if missing:
        raise ValueError(
            "Unknown or disabled live compare model ids: " + ", ".join(sorted(missing))
        )

    if not 2 <= len(requested_model_ids) <= 4:
        raise ValueError("Live compare currently supports between 2 and 4 models per request.")

    preferred_index = {model_id: index for index, model_id in enumerate(DEFAULT_FAIR_ORDER)}
    ordered_ids = sorted(
        requested_model_ids,
        key=lambda model_id: (preferred_index.get(model_id, len(DEFAULT_FAIR_ORDER)), requested_model_ids.index(model_id)),
    )
    return [enabled_models[model_id] for model_id in ordered_ids]


def _build_summary(
    results: list[LiveCompareModelResult],
    *,
    execution_mode: CompareExecutionMode,
    execution_order: list[str],
) -> LiveCompareSummary:
    """Generate a compact summary that is easy to surface above the comparison cards."""

    successful = [row for row in results if row.success]
    failed = [row for row in results if row.success is False]
    fastest = min(
        [row for row in successful if row.isolated_duration_ms is not None],
        key=lambda row: row.isolated_duration_ms or float("inf"),
        default=None,
    )
    longest = max(successful, key=lambda row: len(row.response_text or ""), default=None)
    shortest = min(
        [row for row in successful if row.response_text],
        key=lambda row: len(row.response_text or ""),
        default=None,
    )
    advisory = (
        "Serieller Fair-Compare: Modelle werden nacheinander ausgefuehrt, damit CPU-lastige lokale Hardware die Laufzeiten nicht verfälscht."
        if execution_mode == "serial"
        else "Parallel Compare: Lokale CPU-Modelle koennen sich gegenseitig ausbremsen; Latenzen sind dadurch nur eingeschraenkt vergleichbar."
    )
    return LiveCompareSummary(
        fastest_model_id=fastest.model_id if fastest else None,
        fastest_model_label=fastest.model_label if fastest else None,
        shortest_response_model_id=shortest.model_id if shortest else None,
        shortest_response_model_label=shortest.model_label if shortest else None,
        longest_response_model_id=longest.model_id if longest else None,
        longest_response_model_label=longest.model_label if longest else None,
        execution_mode=execution_mode,
        execution_order=execution_order,
        advisory=advisory,
        all_successful=len(successful) == len(results),
        successful_model_count=len(successful),
        failed_model_count=len(failed),
    )


def _apply_quick_badges(
    results: list[LiveCompareModelResult],
    *,
    summary: LiveCompareSummary,
    request: LiveCompareRequest,
) -> None:
    """Attach operator-friendly badges after all model responses are known."""

    for result in results:
        badges = list(dict.fromkeys(result.quick_badges))
        if result.model_id == summary.fastest_model_id:
            badges.append("Schnellstes Modell")
        if result.model_id == summary.shortest_response_model_id:
            badges.append("Kuerzeste Antwort")
        if result.model_id == summary.longest_response_model_id:
            badges.append("Laengste Antwort")
        if _detect_uncertainty_marker(request.question, result.response_text):
            badges.append("Unsicherheit markiert")
            result.uncertainty_marked = True
        elif _question_calls_for_uncertainty(request.question):
            result.uncertainty_marked = False
        result.quick_badges = badges


def _build_intrinsic_badges(result: LiveCompareModelResult) -> list[str]:
    badges: list[str] = []
    if result.success is True:
        badges.append("Erfolgreich")
    elif result.success is False:
        badges.append("Fehlgeschlagen")
    if "json" in result.detected_content_types:
        badges.append("JSON erkannt")
    if "yaml" in result.detected_content_types:
        badges.append("YAML erkannt")
    if "code" in result.detected_content_types:
        badges.append("Code erkannt")
    return badges


def _detect_content_types(response_text: str | None, response_json: Any | None) -> list[str]:
    """Detect response patterns for quick dashboard badges."""

    content_types: list[str] = []
    if response_json is not None:
        content_types.append("json")

    if response_text:
        stripped = response_text.strip()
        if _attempt_parse_yaml(stripped) is not None and response_json is None:
            content_types.append("yaml")
        code_patterns = [
            "```",
            r"\bdef\s+\w+\(",
            r"\bclass\s+\w+",
            r"\bpytest\b",
            r"\bimport\s+\w+",
        ]
        if any(re.search(pattern, stripped, flags=re.MULTILINE) for pattern in code_patterns):
            content_types.append("code")
    return content_types


def _attempt_parse_yaml(text: str) -> Any | None:
    if not text:
        return None
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if isinstance(parsed, (dict, list)) and ":" in text:
        return parsed
    return None


def _question_calls_for_uncertainty(question: str) -> bool:
    lowered = question.lower()
    return any(
        pattern in lowered
        for pattern in ["gesicherten informationen", "kennzeichne die unsicherheit", "erfinde nichts", "wenn du dir nicht sicher"]
    )


def _detect_uncertainty_marker(question: str, response_text: str | None) -> bool:
    if not response_text or not _question_calls_for_uncertainty(question):
        return False
    lowered = response_text.lower()
    return any(pattern in lowered for pattern in UNCERTAINTY_HINT_PATTERNS)


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
