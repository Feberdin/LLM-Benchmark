"""
Purpose: Deterministic scoring for latency, format adherence, stability, instruction following and reproducibility.
Input/Output: Consumes run results plus validation counters and enriches them with per-run score breakdowns.
Important invariants: Scores stay on a 0-100 scale and are transparent enough to explain in reports.
How to debug: If rankings look surprising, inspect the normalized weights and validation counters in this module.
"""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher

from llm_benchmark.config.models import ScoreWeights
from llm_benchmark.domain.result import RunResult, ScoreBreakdown
from llm_benchmark.domain.test_case import TestCaseDefinition


class ScoreCalculator:
    """Calculate transparent scores and apply reproducibility once all runs are available."""

    def __init__(self, *, default_weights: ScoreWeights, latency_target_ms: int) -> None:
        self.default_weights = default_weights
        self.latency_target_ms = latency_target_ms

    def calculate_preliminary_score(
        self,
        *,
        result: RunResult,
        test_case: TestCaseDefinition,
        validation_metrics: dict[str, float],
    ) -> ScoreBreakdown:
        """Score a single run without reproducibility, which needs sibling runs later."""

        weights = self.default_weights.merged(test_case.scoring_weights).normalized()

        quality_score = self._score_ratio(
            validation_metrics.get("quality_checks_passed", 0),
            validation_metrics.get("quality_checks_total", 0),
            fallback=result.success,
        )
        format_score = self._score_ratio(
            validation_metrics.get("format_checks_passed", 0),
            validation_metrics.get("format_checks_total", 0),
            fallback=result.success,
        )
        instruction_score = self._score_ratio(
            validation_metrics.get("instruction_checks_passed", 0),
            validation_metrics.get("instruction_checks_total", 0),
            fallback=result.success,
        )
        latency_score = self._score_latency(result.duration_ms, success=result.success)
        stability_score = self._score_stability(result)
        reproducibility_score = 100.0 if result.success else 0.0

        total = self._weighted_total(
            weights=weights,
            quality_score=quality_score,
            format_score=format_score,
            latency_score=latency_score,
            stability_score=stability_score,
            instruction_score=instruction_score,
            reproducibility_score=reproducibility_score,
        )
        return ScoreBreakdown(
            quality_score=round(quality_score, 2),
            format_score=round(format_score, 2),
            latency_score=round(latency_score, 2),
            stability_score=round(stability_score, 2),
            instruction_score=round(instruction_score, 2),
            reproducibility_score=round(reproducibility_score, 2),
            total_score=round(total, 2),
            weights=weights,
        )

    def apply_reproducibility_scores(self, results: list[RunResult]) -> None:
        """Score consistency across repeated runs and recompute the weighted totals."""

        grouped_results: dict[tuple[str, str], list[RunResult]] = defaultdict(list)
        for result in results:
            grouped_results[(result.model_id, result.test_case_id)].append(result)

        for group in grouped_results.values():
            measured_runs = [item for item in group if item.metadata.get("phase") != "warmup"]
            target_runs = measured_runs or group
            reproducibility_score = self._calculate_group_reproducibility(target_runs)
            for result in group:
                result.score_breakdown.reproducibility_score = round(reproducibility_score, 2)
                result.score_breakdown.total_score = round(
                    self._weighted_total(
                        weights=result.score_breakdown.weights,
                        quality_score=result.score_breakdown.quality_score,
                        format_score=result.score_breakdown.format_score,
                        latency_score=result.score_breakdown.latency_score,
                        stability_score=result.score_breakdown.stability_score,
                        instruction_score=result.score_breakdown.instruction_score,
                        reproducibility_score=result.score_breakdown.reproducibility_score,
                    ),
                    2,
                )
                result.score_total = result.score_breakdown.total_score

    def _calculate_group_reproducibility(self, results: list[RunResult]) -> float:
        if not results:
            return 0.0

        success_ratio = sum(1 for item in results if item.success) / len(results)
        validation_ratio = sum(1 for item in results if item.validation_passed) / len(results)
        successful_texts = [self._normalize_text(item.raw_response_text) for item in results if item.success]

        if len(successful_texts) <= 1:
            similarity_ratio = 1.0 if len(results) == 1 and successful_texts else success_ratio
        else:
            pairwise_scores: list[float] = []
            for left_index, left_text in enumerate(successful_texts):
                for right_text in successful_texts[left_index + 1 :]:
                    pairwise_scores.append(SequenceMatcher(None, left_text, right_text).ratio())
            similarity_ratio = sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else 0.0

        reproducibility_score = (success_ratio * 40.0) + (validation_ratio * 20.0) + (similarity_ratio * 40.0)
        return max(0.0, min(100.0, reproducibility_score))

    @staticmethod
    def _score_ratio(passed: float, total: float, *, fallback: bool) -> float:
        if total <= 0:
            return 100.0 if fallback else 0.0
        return max(0.0, min(100.0, (passed / total) * 100.0))

    def _score_latency(self, duration_ms: float, *, success: bool) -> float:
        if not success or duration_ms <= 0:
            return 0.0
        target = max(1.0, float(self.latency_target_ms))
        if duration_ms <= target:
            return 100.0
        ratio = target / duration_ms
        return max(5.0, min(100.0, ratio * 100.0))

    @staticmethod
    def _score_stability(result: RunResult) -> float:
        if not result.success:
            return 0.0
        score = 100.0 - (result.retries * 15.0)
        if result.timeout:
            score -= 20.0
        if result.http_status and result.http_status >= 500:
            score -= 10.0
        return max(0.0, min(100.0, score))

    @staticmethod
    def _normalize_text(text: str | None) -> str:
        if not text:
            return ""
        return " ".join(text.lower().split())

    @staticmethod
    def _weighted_total(
        *,
        weights: dict[str, float],
        quality_score: float,
        format_score: float,
        latency_score: float,
        stability_score: float,
        instruction_score: float,
        reproducibility_score: float,
    ) -> float:
        return (
            weights.get("quality", 0.0) * quality_score
            + weights.get("format", 0.0) * format_score
            + weights.get("latency", 0.0) * latency_score
            + weights.get("stability", 0.0) * stability_score
            + weights.get("instruction", 0.0) * instruction_score
            + weights.get("reproducibility", 0.0) * reproducibility_score
        )
