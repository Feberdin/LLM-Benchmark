"""
Purpose: Typed test case definitions used by the loader, validator, runner and report builders.
Input/Output: Parsed from YAML or JSON files under `/app/tests` or another configured directory.
Important invariants: Test ids stay unique and validation rules remain explicit instead of implicit.
How to debug: If a suite or validation rule behaves strangely, inspect the loaded model from this module.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_benchmark.config.models import ScoreWeights

CategoryName = Literal[
    "chat",
    "summarization",
    "extraction_json",
    "classification",
    "function_calling",
    "long_context",
    "instruction_following",
    "coding",
    "robustness",
]
ExpectedFormat = Literal["text", "markdown", "json", "yaml", "tool_call"]


class ValidationRules(BaseModel):
    """Rule-based validation that keeps results explainable for humans and machines."""

    model_config = ConfigDict(extra="forbid")

    json_schema: dict[str, Any] | None = None
    json_schema_file: str | None = None
    required_fields: list[str] = Field(default_factory=list)
    required_json_paths: list[str] = Field(default_factory=list)
    expected_json_values: dict[str, Any] = Field(default_factory=dict)
    contains_all: list[str] = Field(default_factory=list)
    contains_none: list[str] = Field(default_factory=list)
    reference_keywords_any: list[str] = Field(default_factory=list)
    regex_must_match: list[str] = Field(default_factory=list)
    regex_must_not_match: list[str] = Field(default_factory=list)
    min_length_chars: int | None = Field(default=None, ge=0)
    max_length_chars: int | None = Field(default=None, ge=1)
    tool_call_required: bool = False
    expected_tool_name: str | None = None
    reference_keywords: list[str] = Field(default_factory=list)


class TestCaseDefinition(BaseModel):
    """Complete benchmark test case including prompt, validation and optional API extras."""

    model_config = ConfigDict(extra="forbid")
    __test__ = False

    test_case_id: str = Field(min_length=2, max_length=100)
    category: CategoryName
    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=10)
    prompt: str = Field(min_length=1)
    prompt_file: str | None = None
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    expected_format: ExpectedFormat = "text"
    validation_rules: ValidationRules = Field(default_factory=ValidationRules)
    scoring_weights: ScoreWeights | None = None
    tags: list[str] = Field(default_factory=list)
    suites: list[str] = Field(default_factory=lambda: ["core"])
    max_output_tokens: int = Field(default=512, ge=1, le=32000)
    repetitions_default: int = Field(default=1, ge=1, le=20)
    prompt_version: str = "1.0"
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    request_overrides: dict[str, Any] = Field(default_factory=dict)
    quality_reference: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tool_requirements(self) -> "TestCaseDefinition":
        if self.expected_format == "tool_call" and not self.validation_rules.tool_call_required:
            self.validation_rules.tool_call_required = True
        if self.validation_rules.expected_tool_name and not self.tools:
            raise ValueError(
                "A test case with `expected_tool_name` must define a matching `tools` request payload."
            )
        return self

    def effective_repetitions(self, default_repetitions: int) -> int:
        """Resolve per-test repetitions with a predictable fallback."""

        return self.repetitions_default or default_repetitions
