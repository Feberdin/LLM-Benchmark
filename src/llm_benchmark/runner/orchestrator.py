"""
Purpose: Benchmark orchestration that executes all model/test combinations, isolates failures and records stable results.
Input/Output: Consumes validated config and test cases, then returns enriched run results plus benchmark metadata.
Important invariants: One broken endpoint must not stop other model runs, and warmup versus measured phases stay explicit.
How to debug: Start here when execution order, timeout handling or retry isolation behaves unexpectedly.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import httpx

from llm_benchmark.clients.openai_compatible import BenchmarkClientError, OpenAICompatibleClient
from llm_benchmark.config.models import BenchmarkConfig
from llm_benchmark.domain.result import RunResult
from llm_benchmark.domain.test_case import TestCaseDefinition
from llm_benchmark.runner.scoring import ScoreCalculator
from llm_benchmark.utils import build_environment_info, isoformat_utc, sha256_text, utcnow
from llm_benchmark.validation.service import ResponseValidator

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class BenchmarkRunSummary:
    """Bundle benchmark run metadata with the generated raw results."""

    benchmark_run_id: str
    benchmark_name: str
    suite: str | None
    run_started_at: str
    run_finished_at: str
    environment_info: dict[str, Any]
    results: list[RunResult]
    selected_models: list[str]
    selected_test_cases: list[TestCaseDefinition]


class BenchmarkOrchestrator:
    """Run benchmark suites against all enabled models with bounded concurrency."""

    def __init__(self, config: BenchmarkConfig, progress_callback: ProgressCallback | None = None) -> None:
        self.config = config
        self.validator = ResponseValidator()
        self.scorer = ScoreCalculator(
            default_weights=config.run_defaults.scoring_weights,
            latency_target_ms=config.run_defaults.latency_target_ms,
        )
        self.progress_callback = progress_callback

    async def run(self, *, test_cases: list[TestCaseDefinition], suite: str | None) -> BenchmarkRunSummary:
        """Execute the selected suite and return normalized results."""

        enabled_models = self.config.enabled_models()
        if not enabled_models:
            raise ValueError("No enabled models found in the configuration.")
        if not test_cases:
            raise ValueError("No test cases selected for execution.")

        benchmark_run_id = str(uuid.uuid4())
        benchmark_started_at = utcnow()
        semaphore = asyncio.Semaphore(self.config.run_defaults.concurrency)

        LOGGER.info(
            "Starting benchmark run %s with %s model(s) and %s test case(s).",
            benchmark_run_id,
            len(enabled_models),
            len(test_cases),
        )
        self._emit_progress(
            event="run_started",
            stage="running",
            benchmark_run_id=benchmark_run_id,
            suite=suite or "all",
            model_count=len(enabled_models),
            test_case_count=len(test_cases),
            total_planned_records=self._estimate_total_records(enabled_models, test_cases),
        )

        async with OpenAICompatibleClient(
            max_retries=self.config.run_defaults.max_retries,
            retry_backoff_seconds=self.config.run_defaults.retry_backoff_seconds,
            user_agent=f"{self.config.benchmark_name}/0.1.0",
        ) as client:
            tasks = [
                asyncio.create_task(
                    self._run_test_case_for_model(
                        client=client,
                        semaphore=semaphore,
                        benchmark_run_id=benchmark_run_id,
                        model_config=model_config,
                        test_case=test_case,
                    )
                )
                for model_config in enabled_models
                for test_case in test_cases
            ]
            grouped_results = await asyncio.gather(*tasks)

        results = [result for group in grouped_results for result in group]
        self.scorer.apply_reproducibility_scores(results)

        benchmark_finished_at = utcnow()
        LOGGER.info(
            "Finished benchmark run %s with %s total execution records.",
            benchmark_run_id,
            len(results),
        )
        success_count = sum(1 for result in results if result.success and result.validation_passed)
        failure_count = sum(1 for result in results if not result.success or not result.validation_passed)
        self._emit_progress(
            event="run_finished",
            stage="running",
            benchmark_run_id=benchmark_run_id,
            total_records=len(results),
            successful_records=success_count,
            failed_records=failure_count,
        )

        return BenchmarkRunSummary(
            benchmark_run_id=benchmark_run_id,
            benchmark_name=self.config.benchmark_name,
            suite=suite,
            run_started_at=isoformat_utc(benchmark_started_at),
            run_finished_at=isoformat_utc(benchmark_finished_at),
            environment_info=build_environment_info(),
            results=results,
            selected_models=[model.id for model in enabled_models],
            selected_test_cases=test_cases,
        )

    async def _run_test_case_for_model(
        self,
        *,
        client: OpenAICompatibleClient,
        semaphore: asyncio.Semaphore,
        benchmark_run_id: str,
        model_config: Any,
        test_case: TestCaseDefinition,
    ) -> list[RunResult]:
        """Run warmup plus measured repetitions sequentially for one model/test pair."""

        results: list[RunResult] = []
        timeout_seconds = model_config.effective_timeout(self.config.run_defaults.default_timeout_seconds)

        for warmup_index in range(1, self.config.run_defaults.warmup_runs + 1):
            self._emit_progress(
                event="run_step_started",
                stage="running",
                model_id=model_config.id,
                model_label=model_config.label,
                test_case_id=test_case.test_case_id,
                test_title=test_case.title,
                phase="warmup",
                repetition_index=warmup_index,
            )
            warmup_result = await self._execute_single_run(
                client=client,
                semaphore=semaphore,
                benchmark_run_id=benchmark_run_id,
                model_config=model_config,
                test_case=test_case,
                repetition_index=warmup_index,
                timeout_seconds=timeout_seconds,
                phase="warmup",
            )
            if self.config.run_defaults.include_warmup_in_raw_outputs:
                results.append(warmup_result)
            self._emit_result_progress(warmup_result)

        measured_repetitions = test_case.effective_repetitions(self.config.run_defaults.default_repetitions)
        for repetition_index in range(1, measured_repetitions + 1):
            phase = "warm" if self.config.run_defaults.warmup_runs > 0 or repetition_index > 1 else "cold"
            self._emit_progress(
                event="run_step_started",
                stage="running",
                model_id=model_config.id,
                model_label=model_config.label,
                test_case_id=test_case.test_case_id,
                test_title=test_case.title,
                phase=phase,
                repetition_index=repetition_index,
            )
            measured_result = await self._execute_single_run(
                client=client,
                semaphore=semaphore,
                benchmark_run_id=benchmark_run_id,
                model_config=model_config,
                test_case=test_case,
                repetition_index=repetition_index,
                timeout_seconds=timeout_seconds,
                phase=phase,
            )
            results.append(measured_result)
            self._emit_result_progress(measured_result)

        return results

    async def _execute_single_run(
        self,
        *,
        client: OpenAICompatibleClient,
        semaphore: asyncio.Semaphore,
        benchmark_run_id: str,
        model_config: Any,
        test_case: TestCaseDefinition,
        repetition_index: int,
        timeout_seconds: float,
        phase: str,
    ) -> RunResult:
        """Execute one request, validate the response and keep all errors local to this run."""

        prompt_hash = sha256_text(f"{test_case.system_prompt or ''}\n\n{test_case.prompt}")
        run_started_at = utcnow()
        started_monotonic = perf_counter()
        metadata: dict[str, Any] = {
            "phase": phase,
            "suites": test_case.suites,
            "tags": test_case.tags,
            "expected_format": test_case.expected_format,
            "request_parameters": model_config.default_parameters | test_case.request_overrides,
            "supports_streaming": model_config.supports_streaming,
            "supports_tools": model_config.supports_tools,
            "supports_structured_output": model_config.supports_structured_output,
        }

        try:
            async with semaphore:
                response, retries_used = await client.execute(
                    model_config=model_config,
                    test_case=test_case,
                    timeout_seconds=timeout_seconds,
                    stream_for_ttft=self.config.run_defaults.stream_for_ttft,
                )

            validation_outcome = self.validator.validate(test_case, response)
            metadata["validation_metrics"] = validation_outcome.metrics
            metadata["finish_reason"] = response.finish_reason
            if response.tool_calls:
                metadata["tool_calls"] = response.tool_calls

            duration_ms = round((perf_counter() - started_monotonic) * 1000, 2)
            tokens_per_second = self._calculate_tokens_per_second(response.output_tokens, duration_ms)
            raw_response_text = response.raw_response_text if self.config.run_defaults.capture_raw_response_text else None

            result = RunResult(
                benchmark_run_id=benchmark_run_id,
                run_started_at=isoformat_utc(run_started_at),
                run_finished_at=isoformat_utc(),
                model_id=model_config.id,
                model_label=model_config.label,
                provider=model_config.provider,
                endpoint=model_config.base_url,
                model_name=model_config.model_name,
                test_case_id=test_case.test_case_id,
                category=test_case.category,
                repetition_index=repetition_index,
                prompt_hash=prompt_hash,
                prompt_version=test_case.prompt_version or self.config.run_defaults.prompt_version,
                duration_ms=duration_ms,
                ttft_ms=response.ttft_ms,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                tokens_per_second=tokens_per_second,
                http_status=response.http_status,
                success=True,
                timeout=False,
                retries=retries_used,
                raw_response_text=raw_response_text,
                parsed_output_json=validation_outcome.parsed_output_json,
                validation_passed=validation_outcome.passed,
                validation_errors=[issue.model_dump(mode="json") for issue in validation_outcome.issues],
                error_type=None,
                error_message=None,
                metadata=metadata,
            )
            result.score_breakdown = self.scorer.calculate_preliminary_score(
                result=result,
                test_case=test_case,
                validation_metrics=validation_outcome.metrics,
            )
            result.score_total = result.score_breakdown.total_score
            return result
        except httpx.TimeoutException:
            return self._build_failed_result(
                benchmark_run_id=benchmark_run_id,
                run_started_at=run_started_at,
                started_monotonic=started_monotonic,
                model_config=model_config,
                test_case=test_case,
                repetition_index=repetition_index,
                prompt_hash=prompt_hash,
                metadata=metadata,
                error_type="timeout",
                error_message=(
                    f"Request timed out after {timeout_seconds} seconds. "
                    "Increase the model timeout or inspect backend latency."
                ),
                http_status=None,
                timeout=True,
            )
        except BenchmarkClientError as exc:
            if exc.payload:
                metadata["provider_error_payload"] = exc.payload
            return self._build_failed_result(
                benchmark_run_id=benchmark_run_id,
                run_started_at=run_started_at,
                started_monotonic=started_monotonic,
                model_config=model_config,
                test_case=test_case,
                repetition_index=repetition_index,
                prompt_hash=prompt_hash,
                metadata=metadata,
                error_type=exc.error_type,
                error_message=str(exc),
                http_status=exc.http_status,
                timeout=False,
            )
        except httpx.TransportError as exc:
            return self._build_failed_result(
                benchmark_run_id=benchmark_run_id,
                run_started_at=run_started_at,
                started_monotonic=started_monotonic,
                model_config=model_config,
                test_case=test_case,
                repetition_index=repetition_index,
                prompt_hash=prompt_hash,
                metadata=metadata,
                error_type="network_error",
                error_message=(
                    f"Network communication with '{model_config.base_url}' failed: {exc}. "
                    "Check DNS, routing, firewall rules and whether the endpoint is online."
                ),
                http_status=None,
                timeout=False,
            )
        except Exception as exc:  # pragma: no cover - guard rail for production robustness.
            metadata["traceback"] = traceback.format_exc(limit=10)
            LOGGER.exception(
                "Unexpected error during model=%s test_case=%s repetition=%s",
                model_config.id,
                test_case.test_case_id,
                repetition_index,
            )
            return self._build_failed_result(
                benchmark_run_id=benchmark_run_id,
                run_started_at=run_started_at,
                started_monotonic=started_monotonic,
                model_config=model_config,
                test_case=test_case,
                repetition_index=repetition_index,
                prompt_hash=prompt_hash,
                metadata=metadata,
                error_type="unexpected_error",
                error_message=str(exc),
                http_status=None,
                timeout=False,
            )

    def _build_failed_result(
        self,
        *,
        benchmark_run_id: str,
        run_started_at: Any,
        started_monotonic: float,
        model_config: Any,
        test_case: TestCaseDefinition,
        repetition_index: int,
        prompt_hash: str,
        metadata: dict[str, Any],
        error_type: str,
        error_message: str,
        http_status: int | None,
        timeout: bool,
    ) -> RunResult:
        duration_ms = round((perf_counter() - started_monotonic) * 1000, 2)
        result = RunResult(
            benchmark_run_id=benchmark_run_id,
            run_started_at=isoformat_utc(run_started_at),
            run_finished_at=isoformat_utc(),
            model_id=model_config.id,
            model_label=model_config.label,
            provider=model_config.provider,
            endpoint=model_config.base_url,
            model_name=model_config.model_name,
            test_case_id=test_case.test_case_id,
            category=test_case.category,
            repetition_index=repetition_index,
            prompt_hash=prompt_hash,
            prompt_version=test_case.prompt_version or self.config.run_defaults.prompt_version,
            duration_ms=duration_ms,
            ttft_ms=None,
            input_tokens=None,
            output_tokens=None,
            tokens_per_second=None,
            http_status=http_status,
            success=False,
            timeout=timeout,
            retries=0,
            raw_response_text=None,
            parsed_output_json=None,
            validation_passed=False,
            validation_errors=[],
            error_type=error_type,
            error_message=error_message,
            metadata=metadata,
        )
        result.score_breakdown = self.scorer.calculate_preliminary_score(
            result=result,
            test_case=test_case,
            validation_metrics={},
        )
        result.score_total = result.score_breakdown.total_score
        return result

    @staticmethod
    def _calculate_tokens_per_second(output_tokens: int | None, duration_ms: float) -> float | None:
        if not output_tokens or duration_ms <= 0:
            return None
        return round(output_tokens / (duration_ms / 1000.0), 2)

    def _emit_result_progress(self, result: RunResult) -> None:
        self._emit_progress(
            event="run_step_finished",
            stage="running",
            model_id=result.model_id,
            model_label=result.model_label,
            test_case_id=result.test_case_id,
            category=result.category,
            phase=result.metadata.get("phase", "measured"),
            repetition_index=result.repetition_index,
            success=result.success,
            validation_passed=result.validation_passed,
            duration_ms=result.duration_ms,
            error_type=result.error_type,
            error_message=result.error_message,
            score_total=result.score_total,
        )

    def _estimate_total_records(
        self,
        enabled_models: list[Any],
        test_cases: list[TestCaseDefinition],
    ) -> int:
        total = 0
        for test_case in test_cases:
            total += self.config.run_defaults.warmup_runs + test_case.effective_repetitions(
                self.config.run_defaults.default_repetitions
            )
        return total * len(enabled_models)

    def _emit_progress(self, **payload: Any) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(payload)
