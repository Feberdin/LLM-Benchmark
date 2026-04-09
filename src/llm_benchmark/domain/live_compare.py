"""
Purpose: Stable request and result models for the interactive live-compare dashboard view.
Input/Output: Accepts a single user prompt plus selected model ids and returns one structured result object per model.
Important invariants: The live compare API stays machine-readable, persists historical runs and remains compatible with later 2-4 model comparisons.
How to debug: Inspect the persisted JSON files under `/app/results/live_compare` and compare them with these models when fields are missing in the UI.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CompareMode = Literal["chat", "json", "technical", "summarization"]
CompareExecutionMode = Literal["serial", "parallel"]
CompareResultStatus = Literal["waiting", "running", "finished", "failed"]
CompareRunStatus = Literal["idle", "queued", "running", "succeeded", "partial", "failed", "interrupted"]


class LiveComparePreset(BaseModel):
    """One reusable live compare prompt preset shown directly in the dashboard UI."""

    model_config = ConfigDict(extra="forbid")

    preset_id: str = Field(min_length=2, max_length=100)
    title: str = Field(min_length=3, max_length=200)
    goal: str = Field(min_length=10, max_length=500)
    prompt: str = Field(min_length=1, max_length=32000)
    system_prompt: str | None = Field(default=None, max_length=16000)
    mode: CompareMode = "chat"


class LiveCompareRequest(BaseModel):
    """Incoming request payload for a single live compare execution."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=32000)
    system_prompt: str | None = Field(default=None, max_length=16000)
    models: list[str] = Field(default_factory=list, min_length=2, max_length=4)
    mode: CompareMode = "chat"
    execution_mode: CompareExecutionMode = "serial"
    max_tokens: int = Field(default=600, ge=16, le=32000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    manual_note: str | None = Field(default=None, max_length=2000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("The live compare question must not be empty.")
        return text

    @field_validator("system_prompt")
    @classmethod
    def normalize_system_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("manual_note")
    @classmethod
    def normalize_manual_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("models")
    @classmethod
    def ensure_unique_models(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item and item.strip()]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Live compare model ids must be unique.")
        return normalized


class LiveCompareModelResult(BaseModel):
    """Result or progress snapshot for one model column inside the live compare view."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    model_label: str
    provider: str
    model_name: str
    endpoint: str | None = None
    status: CompareResultStatus = "waiting"
    success: bool | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: float | None = None
    duration_human: str | None = None
    queue_wait_ms: float | None = None
    execution_start_at: str | None = None
    execution_end_at: str | None = None
    isolated_duration_ms: float | None = None
    total_elapsed_since_run_start_ms: float | None = None
    total_elapsed_human: str | None = None
    ttft_ms: float | None = None
    tokens_per_second: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_text: str | None = None
    response_json: Any | None = None
    validation_passed: bool | None = None
    http_status: int | None = None
    retries: int = 0
    finish_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    quick_badges: list[str] = Field(default_factory=list)
    detected_content_types: list[str] = Field(default_factory=list)
    uncertainty_marked: bool | None = None
    technical_details: dict[str, Any] = Field(default_factory=dict)


class LiveCompareSummary(BaseModel):
    """Compact compare-wide summary for UI badges and history rows."""

    model_config = ConfigDict(extra="forbid")

    fastest_model_id: str | None = None
    fastest_model_label: str | None = None
    shortest_response_model_id: str | None = None
    shortest_response_model_label: str | None = None
    longest_response_model_id: str | None = None
    longest_response_model_label: str | None = None
    execution_mode: CompareExecutionMode = "serial"
    execution_order: list[str] = Field(default_factory=list)
    advisory: str | None = None
    all_successful: bool = False
    successful_model_count: int = 0
    failed_model_count: int = 0


class LiveCompareEvent(BaseModel):
    """Operator-facing timeline event for the live compare run history."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str
    event: str
    level: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class LiveCompareRunRecord(BaseModel):
    """Persisted machine-readable live compare run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: CompareRunStatus = "succeeded"
    created_at: str
    finished_at: str | None = None
    question: str
    system_prompt: str | None = None
    mode: CompareMode = "chat"
    execution_mode: CompareExecutionMode = "serial"
    execution_order: list[str] = Field(default_factory=list)
    selected_models: list[str] = Field(default_factory=list)
    max_tokens: int = 600
    temperature: float = 0.0
    top_p: float = 1.0
    manual_note: str | None = None
    results: list[LiveCompareModelResult] = Field(default_factory=list)
    summary: LiveCompareSummary = Field(default_factory=LiveCompareSummary)
    latest_error: str | None = None
    events: list[LiveCompareEvent] = Field(default_factory=list)
