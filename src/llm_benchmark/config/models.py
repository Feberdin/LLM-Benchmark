"""
Purpose: Typed configuration models for benchmark settings, model endpoints and scoring defaults.
Input/Output: Loader code parses YAML or JSON into these models before execution starts.
Important invariants: Extra fields are rejected to prevent hidden typos in production configs.
How to debug: Use `benchmark validate-config` to inspect validation failures raised by these models.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ScoreWeights(BaseModel):
    """Scoring weights with a stable default that can be overridden per test case."""

    model_config = ConfigDict(extra="forbid")

    quality: float = 0.35
    format: float = 0.25
    latency: float = 0.20
    stability: float = 0.10
    reproducibility: float = 0.10
    instruction: float = 0.00

    @model_validator(mode="after")
    def validate_non_negative(self) -> "ScoreWeights":
        if any(value < 0 for value in self.model_dump().values()):
            raise ValueError("Scoring weights must be zero or positive.")
        total = sum(self.model_dump().values())
        if total <= 0:
            raise ValueError("At least one scoring weight must be greater than zero.")
        return self

    def normalized(self) -> dict[str, float]:
        """Normalize weights so weighted totals stay on a 0-100 scale."""

        values = self.model_dump()
        total = sum(values.values())
        return {key: value / total for key, value in values.items()}

    def merged(self, override: "ScoreWeights | None") -> "ScoreWeights":
        """Return a new weight object where provided override values replace the defaults."""

        if override is None:
            return self
        return ScoreWeights(**(self.model_dump() | override.model_dump(exclude_unset=True)))


class RunDefaults(BaseModel):
    """Global execution defaults that keep benchmark runs fair and reproducible."""

    model_config = ConfigDict(extra="forbid")

    concurrency: int = Field(default=1, ge=1, le=32)
    warmup_runs: int = Field(default=0, ge=0, le=10)
    default_repetitions: int = Field(default=2, ge=1, le=20)
    max_retries: int = Field(default=2, ge=0, le=10)
    retry_backoff_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    default_timeout_seconds: float = Field(default=120.0, ge=1.0, le=3600.0)
    stream_for_ttft: bool = False
    capture_raw_response_text: bool = True
    include_warmup_in_raw_outputs: bool = True
    prompt_version: str = "1.0"
    latency_target_ms: int = Field(default=5000, ge=100)
    scoring_weights: ScoreWeights = Field(default_factory=ScoreWeights)
    tests_dir: str = "/app/tests"
    output_dir: str = "/app/results"


class BenchmarkModelConfig(BaseModel):
    """Single model endpoint definition for an OpenAI-compatible backend."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=2, max_length=100)
    label: str = Field(min_length=2, max_length=200)
    provider: str = Field(min_length=2, max_length=100)
    base_url: str = Field(min_length=8)
    api_type: Literal["openai_chat_completions", "openai_compatible", "openai"] = "openai_chat_completions"
    model_name: str = Field(min_length=1, max_length=200)
    api_key_env: str | None = None
    enabled: bool = True
    default_parameters: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float | None = Field(default=None, ge=1.0, le=3600.0)
    supports_streaming: bool | None = None
    supports_tools: bool | None = None
    supports_structured_output: bool | None = None

    @field_validator("base_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("Model `base_url` must start with http:// or https://.")
        return cleaned

    @field_validator("api_key_env")
    @classmethod
    def normalize_api_key_env(cls, value: str | None) -> str | None:
        return value.strip() if value else value

    @field_validator("api_type")
    @classmethod
    def normalize_api_type(cls, value: str) -> str:
        """
        Accept user-facing aliases without forcing operators to learn the internal transport name.

        Supported examples:
        - `openai`
        - `openai_compatible`
        - `openai_chat_completions`
        """

        cleaned = value.strip().lower()
        if cleaned not in {"openai", "openai_compatible", "openai_chat_completions"}:
            raise ValueError(
                "Unsupported api_type. Use `openai`, `openai_compatible`, or `openai_chat_completions`."
            )
        return cleaned

    def effective_timeout(self, default_timeout_seconds: float) -> float:
        """Resolve the per-model timeout while keeping the config definition compact."""

        return self.timeout_seconds or default_timeout_seconds

    def has_reasoning_effort_control(self) -> bool:
        """
        Return whether the model already declares an explicit OpenAI-compatible reasoning control.

        Why this exists:
        Some local thinking models on Ollama can emit only reasoning text when no reasoning budget
        policy is configured. We keep the detection explicit so doctor/preflight can guide operators.
        """

        if "reasoning_effort" in self.default_parameters:
            return True
        reasoning = self.default_parameters.get("reasoning")
        return isinstance(reasoning, dict) and reasoning.get("effort") is not None

    def needs_reasoning_control_hint(self) -> bool:
        """
        Flag common local thinking-model setups that benefit from explicit reasoning control.

        Example:
        - `qwen3.5:35b-a3b` behind Ollama `/v1/chat/completions`
        - no `reasoning_effort` configured
        """

        if self.has_reasoning_effort_control():
            return False
        if self.api_type not in {"openai", "openai_compatible", "openai_chat_completions"}:
            return False
        model_name = self.model_name.lower()
        return "qwen" in model_name


class BenchmarkConfig(BaseModel):
    """Top-level benchmark configuration file."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1"
    benchmark_name: str = "llm-benchmark"
    description: str = "Benchmark OpenAI-compatible LLM endpoints with reproducible test suites."
    models: list[BenchmarkModelConfig] = Field(min_length=1)
    run_defaults: RunDefaults = Field(default_factory=RunDefaults)

    @model_validator(mode="after")
    def validate_unique_model_ids(self) -> "BenchmarkConfig":
        model_ids = [model.id for model in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("Model ids must be unique inside the benchmark configuration.")
        return self

    def enabled_models(self) -> list[BenchmarkModelConfig]:
        """Return only the active model endpoints."""

        return [model for model in self.models if model.enabled]
