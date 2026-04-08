"""
Purpose: Stable result and response models used for persistence, scoring and report generation.
Input/Output: Created by the runner and exported as JSONL, CSV and aggregate reports.
Important invariants: Field names are intentionally explicit because downstream analytics depend on them.
How to debug: Inspect these models when export fields disappear or the final report schema changes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UnifiedResponse(BaseModel):
    """Backend-agnostic representation of a single model response."""

    model_config = ConfigDict(extra="forbid")

    http_status: int | None = None
    raw_payload: dict[str, Any] | None = None
    raw_response_text: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_ms: float | None = None


class ScoreBreakdown(BaseModel):
    """Per-run scores on a 0-100 scale plus the weighted total."""

    model_config = ConfigDict(extra="forbid")

    quality_score: float = 0.0
    format_score: float = 0.0
    latency_score: float = 0.0
    stability_score: float = 0.0
    instruction_score: float = 0.0
    reproducibility_score: float = 0.0
    total_score: float = 0.0
    weights: dict[str, float] = Field(default_factory=dict)


class RunResult(BaseModel):
    """Canonical persisted benchmark record for one test repetition against one model."""

    model_config = ConfigDict(extra="forbid")

    benchmark_run_id: str
    run_started_at: str
    run_finished_at: str
    model_id: str
    model_label: str
    provider: str
    endpoint: str
    model_name: str
    test_case_id: str
    category: str
    repetition_index: int
    prompt_hash: str
    prompt_version: str
    duration_ms: float
    ttft_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tokens_per_second: float | None = None
    http_status: int | None = None
    success: bool
    timeout: bool = False
    retries: int = 0
    raw_response_text: str | None = None
    parsed_output_json: Any | None = None
    validation_passed: bool = False
    validation_errors: list[dict[str, Any]] = Field(default_factory=list)
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    score_total: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
