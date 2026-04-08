"""
Purpose: Rule-based validation for JSON structure, tool calls, instruction following and keyword coverage.
Input/Output: Consumes a test definition plus normalized model response and returns explicit validation outcomes.
Important invariants: Validation failures remain explainable and never mutate the original response payload.
How to debug: Inspect the generated validation issue list before changing scoring behavior.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

import jsonschema
import orjson
import yaml
from pydantic import BaseModel, ConfigDict, Field

from llm_benchmark.domain.result import UnifiedResponse
from llm_benchmark.domain.test_case import TestCaseDefinition

IssueCategory = Literal["format", "instruction", "quality", "system"]


class ValidationIssue(BaseModel):
    """Single validation failure with enough context for reports and debugging."""

    model_config = ConfigDict(extra="forbid")

    code: str
    category: IssueCategory
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationOutcome(BaseModel):
    """Validation result plus score-relevant counters."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    parsed_output_json: Any | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class ResponseValidator:
    """Apply deterministic validation rules that favor explainability over opaque heuristics."""

    def validate(self, test_case: TestCaseDefinition, response: UnifiedResponse) -> ValidationOutcome:
        text = response.raw_response_text or ""
        parsed_json: Any | None = None
        issues: list[ValidationIssue] = []
        metrics = {
            "format_checks_total": 0,
            "format_checks_passed": 0,
            "instruction_checks_total": 0,
            "instruction_checks_passed": 0,
            "quality_checks_total": 0,
            "quality_checks_passed": 0,
            "tool_call_present": bool(response.tool_calls),
            "response_length_chars": len(text),
        }

        if test_case.expected_format in {"json", "yaml"}:
            metrics["format_checks_total"] += 1
            parsed_json = (
                self._parse_json_candidate(text)
                if test_case.expected_format == "json"
                else self._parse_yaml_candidate(text)
            )
            if parsed_json is None:
                issues.append(
                    ValidationIssue(
                        code="json_parse_failed" if test_case.expected_format == "json" else "yaml_parse_failed",
                        category="format",
                        message=(
                            "Expected a JSON response, but the output could not be parsed as JSON."
                            if test_case.expected_format == "json"
                            else "Expected a YAML response, but the output could not be parsed as YAML."
                        ),
                    )
                )
            else:
                metrics["format_checks_passed"] += 1

        if test_case.validation_rules.tool_call_required:
            metrics["format_checks_total"] += 1
            if response.tool_calls:
                metrics["format_checks_passed"] += 1
            else:
                issues.append(
                    ValidationIssue(
                        code="tool_call_missing",
                        category="format",
                        message="A tool call was expected, but the response did not include one.",
                    )
                )

        if test_case.validation_rules.expected_tool_name:
            metrics["format_checks_total"] += 1
            tool_names = {
                tool_call.get("function", {}).get("name")
                for tool_call in response.tool_calls
                if isinstance(tool_call, dict)
            }
            if test_case.validation_rules.expected_tool_name in tool_names:
                metrics["format_checks_passed"] += 1
            else:
                issues.append(
                    ValidationIssue(
                        code="unexpected_tool_name",
                        category="format",
                        message=(
                            "The response did not call the expected tool "
                            f"'{test_case.validation_rules.expected_tool_name}'."
                        ),
                        details={"seen_tool_names": sorted(name for name in tool_names if name)},
                    )
                )

        if parsed_json is not None and test_case.validation_rules.json_schema:
            metrics["format_checks_total"] += 1
            validator = jsonschema.Draft202012Validator(test_case.validation_rules.json_schema)
            schema_errors = sorted(validator.iter_errors(parsed_json), key=lambda item: item.path)
            if schema_errors:
                issues.append(
                    ValidationIssue(
                        code="json_schema_failed",
                        category="format",
                        message="JSON output does not satisfy the configured schema.",
                        details={"errors": [error.message for error in schema_errors[:5]]},
                    )
                )
            else:
                metrics["format_checks_passed"] += 1

        if parsed_json is not None:
            for field_name in test_case.validation_rules.required_fields:
                metrics["format_checks_total"] += 1
                if isinstance(parsed_json, dict) and field_name in parsed_json:
                    metrics["format_checks_passed"] += 1
                else:
                    issues.append(
                        ValidationIssue(
                            code="required_field_missing",
                            category="format",
                            message=f"Required JSON field '{field_name}' is missing.",
                        )
                    )

            for json_path in test_case.validation_rules.required_json_paths:
                metrics["format_checks_total"] += 1
                if self._resolve_path(parsed_json, json_path) is not None:
                    metrics["format_checks_passed"] += 1
                else:
                    issues.append(
                        ValidationIssue(
                            code="required_json_path_missing",
                            category="format",
                            message=f"Required JSON path '{json_path}' is missing.",
                        )
                    )

            for json_path, expected_value in test_case.validation_rules.expected_json_values.items():
                metrics["quality_checks_total"] += 1
                actual_value = self._resolve_path(parsed_json, json_path)
                if actual_value == expected_value:
                    metrics["quality_checks_passed"] += 1
                else:
                    issues.append(
                        ValidationIssue(
                            code="unexpected_json_value",
                            category="quality",
                            message=f"JSON path '{json_path}' does not match the expected value.",
                            details={"expected": expected_value, "actual": actual_value},
                        )
                    )

        lowercase_text = text.lower()

        for expected_snippet in test_case.validation_rules.contains_all:
            metrics["instruction_checks_total"] += 1
            if expected_snippet.lower() in lowercase_text:
                metrics["instruction_checks_passed"] += 1
            else:
                issues.append(
                    ValidationIssue(
                        code="required_text_missing",
                        category="instruction",
                        message=f"Required text fragment is missing: '{expected_snippet}'.",
                    )
                )

        for forbidden_snippet in test_case.validation_rules.contains_none:
            metrics["instruction_checks_total"] += 1
            if forbidden_snippet.lower() not in lowercase_text:
                metrics["instruction_checks_passed"] += 1
            else:
                issues.append(
                    ValidationIssue(
                        code="forbidden_text_present",
                        category="instruction",
                        message=f"Forbidden text fragment is present: '{forbidden_snippet}'.",
                    )
                )

        for pattern in test_case.validation_rules.regex_must_match:
            metrics["instruction_checks_total"] += 1
            if re.search(pattern, text, flags=re.MULTILINE):
                metrics["instruction_checks_passed"] += 1
            else:
                issues.append(
                    ValidationIssue(
                        code="regex_match_failed",
                        category="instruction",
                        message=f"Output does not match the required regex: '{pattern}'.",
                    )
                )

        for pattern in test_case.validation_rules.regex_must_not_match:
            metrics["instruction_checks_total"] += 1
            if re.search(pattern, text, flags=re.MULTILINE):
                issues.append(
                    ValidationIssue(
                        code="regex_forbidden_matched",
                        category="instruction",
                        message=f"Output matches a forbidden regex: '{pattern}'.",
                    )
                )
            else:
                metrics["instruction_checks_passed"] += 1

        if test_case.validation_rules.min_length_chars is not None:
            metrics["instruction_checks_total"] += 1
            if len(text) >= test_case.validation_rules.min_length_chars:
                metrics["instruction_checks_passed"] += 1
            else:
                issues.append(
                    ValidationIssue(
                        code="text_too_short",
                        category="instruction",
                        message=(
                            f"Output is shorter than the minimum length of "
                            f"{test_case.validation_rules.min_length_chars} characters."
                        ),
                    )
                )

        if test_case.validation_rules.max_length_chars is not None:
            metrics["instruction_checks_total"] += 1
            if len(text) <= test_case.validation_rules.max_length_chars:
                metrics["instruction_checks_passed"] += 1
            else:
                issues.append(
                    ValidationIssue(
                        code="text_too_long",
                        category="instruction",
                        message=(
                            f"Output is longer than the maximum length of "
                            f"{test_case.validation_rules.max_length_chars} characters."
                        ),
                    )
                )

        searchable_blob = lowercase_text
        if parsed_json is not None:
            searchable_blob += " " + json.dumps(parsed_json, ensure_ascii=False).lower()

        for keyword in test_case.validation_rules.reference_keywords:
            metrics["quality_checks_total"] += 1
            if keyword.lower() in searchable_blob:
                metrics["quality_checks_passed"] += 1
            else:
                issues.append(
                    ValidationIssue(
                        code="reference_keyword_missing",
                        category="quality",
                        message=f"Reference keyword '{keyword}' was not found in the output.",
                    )
                )

        return ValidationOutcome(
            passed=not issues,
            issues=issues,
            parsed_output_json=parsed_json,
            metrics=metrics,
        )

    @staticmethod
    def _parse_json_candidate(text: str) -> Any | None:
        """Try a few safe JSON extraction strategies without making hidden assumptions."""

        candidates: list[str] = []
        stripped = text.strip()
        if stripped:
            candidates.append(stripped)

        code_fence_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
        if code_fence_match:
            candidates.append(code_fence_match.group(1).strip())

        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if 0 <= brace_start < brace_end:
            candidates.append(text[brace_start : brace_end + 1].strip())

        bracket_start = text.find("[")
        bracket_end = text.rfind("]")
        if 0 <= bracket_start < bracket_end:
            candidates.append(text[bracket_start : bracket_end + 1].strip())

        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                return orjson.loads(candidate)
            except orjson.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _parse_yaml_candidate(text: str) -> Any | None:
        """Parse YAML safely, including fenced code blocks, and require structured output."""

        candidates: list[str] = []
        stripped = text.strip()
        if stripped:
            candidates.append(stripped)

        code_fence_match = re.search(r"```(?:yaml|yml)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        if code_fence_match:
            candidates.append(code_fence_match.group(1).strip())

        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                parsed = yaml.safe_load(candidate)
            except yaml.YAMLError:
                continue
            if isinstance(parsed, (dict, list)):
                return parsed
        return None

    @staticmethod
    def _resolve_path(payload: Any, path: str) -> Any | None:
        """Resolve a dotted JSON path with optional list indexes like `items.0.name`."""

        current = payload
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    index = int(part)
                except ValueError:
                    return None
                if index >= len(current):
                    return None
                current = current[index]
            else:
                return None
            if current is None:
                return None
        return current
