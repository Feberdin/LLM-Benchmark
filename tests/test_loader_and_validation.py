"""
Purpose: Regression tests for config loading, test discovery and response validation behavior.
Input/Output: Uses the example fixture files and small in-memory responses to verify core contracts.
Important invariants: Duplicate ids, broken JSON validation and missing tool calls should remain visible failures.
How to debug: Run `pytest tests/test_loader_and_validation.py -q` and inspect the failing assertion plus fixture data.
"""

from __future__ import annotations

from pathlib import Path

from llm_benchmark.config.loader import filter_test_cases_by_suite, load_config, load_test_cases
from llm_benchmark.config.models import BenchmarkModelConfig
from llm_benchmark.domain.result import UnifiedResponse
from llm_benchmark.domain.test_case import TestCaseDefinition
from llm_benchmark.validation.service import ResponseValidator

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def test_loads_example_config_and_tests() -> None:
    config = load_config(FIXTURES_DIR / "config" / "config.example.yaml")
    test_cases = load_test_cases(FIXTURES_DIR / "tests")

    assert config.benchmark_name == "feberdin-llm-benchmark"
    assert len(config.models) == 3
    assert len(config.enabled_models()) == 3
    assert len(test_cases) == 43
    assert any(case.test_case_id == "tool-call-weather" for case in test_cases)
    assert any(case.test_case_id == "quick-chat-triage" for case in test_cases)
    assert any(case.test_case_id == "sb-rag-context-fidelity" for case in test_cases)
    assert any(case.test_case_id == "vg-routing-matrix" for case in test_cases)
    assert any(case.test_case_id == "pk-yaml-generation" for case in test_cases)


def test_filters_suite_membership() -> None:
    test_cases = load_test_cases(FIXTURES_DIR / "tests")
    core_cases = filter_test_cases_by_suite(test_cases, "core")
    long_context_cases = filter_test_cases_by_suite(test_cases, "long_context")
    quick_compare_cases = filter_test_cases_by_suite(test_cases, "quick_compare")
    secondbrain_cases = filter_test_cases_by_suite(test_cases, "secondbrain")
    voice_gateway_cases = filter_test_cases_by_suite(test_cases, "voice_gateway")
    paperless_kiplus_cases = filter_test_cases_by_suite(test_cases, "paperless_kiplus")

    assert len(core_cases) == 5
    assert len(long_context_cases) == 2
    assert len(quick_compare_cases) == 5
    assert len(secondbrain_cases) == 9
    assert len(voice_gateway_cases) == 9
    assert len(paperless_kiplus_cases) == 9
    assert all("long_context" in case.suites for case in long_context_cases)
    assert all("quick_compare" in case.suites for case in quick_compare_cases)
    assert all(case.prompt for case in secondbrain_cases)
    assert all(case.validation_rules.json_schema for case in paperless_kiplus_cases if case.expected_format in {"json", "yaml"})


def test_unraid_config_accepts_alias_api_types() -> None:
    config = load_config(FIXTURES_DIR / "config" / "config.unraid.example.yaml")

    assert [model.api_type for model in config.models] == ["openai_compatible", "openai_compatible", "openai"]
    qwen_model = next(model for model in config.models if model.id == "qwen_local")
    assert qwen_model.default_parameters["reasoning_effort"] == "none"


def test_validator_accepts_valid_json_payload() -> None:
    validator = ResponseValidator()
    case = TestCaseDefinition.model_validate(
        {
            "test_case_id": "json-case",
            "category": "extraction_json",
            "title": "JSON validation",
            "description": "Simple JSON validation smoke test.",
            "prompt": "Return JSON.",
            "expected_format": "json",
            "validation_rules": {
                "json_schema": {
                    "type": "object",
                    "required": ["status"],
                    "properties": {"status": {"type": "string"}}
                },
                "required_fields": ["status"],
                "expected_json_values": {"status": "ok"}
            }
        }
    )
    response = UnifiedResponse(
        http_status=200,
        raw_payload={"choices": []},
        raw_response_text='{"status":"ok"}',
    )

    outcome = validator.validate(case, response)

    assert outcome.passed is True
    assert outcome.parsed_output_json == {"status": "ok"}
    assert outcome.metrics["format_checks_passed"] == outcome.metrics["format_checks_total"]
    assert outcome.metrics["quality_checks_passed"] == outcome.metrics["quality_checks_total"]


def test_validator_flags_missing_tool_call() -> None:
    validator = ResponseValidator()
    case = TestCaseDefinition.model_validate(
        {
            "test_case_id": "tool-case",
            "category": "function_calling",
            "title": "Tool case",
            "description": "Tool calling validation smoke test.",
            "prompt": "Use the tool.",
            "expected_format": "tool_call",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "description": "Fake test tool.",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ],
            "validation_rules": {
                "tool_call_required": True,
                "expected_tool_name": "lookup_weather"
            }
        }
    )
    response = UnifiedResponse(http_status=200, raw_payload={"choices": []}, raw_response_text="I think it will rain.")

    outcome = validator.validate(case, response)

    assert outcome.passed is False
    assert any(issue.code == "tool_call_missing" for issue in outcome.issues)


def test_validator_accepts_valid_yaml_payload() -> None:
    validator = ResponseValidator()
    case = TestCaseDefinition.model_validate(
        {
            "test_case_id": "yaml-case",
            "category": "instruction_following",
            "title": "YAML validation",
            "description": "Simple YAML validation smoke test.",
            "prompt": "Return YAML.",
            "expected_format": "yaml",
            "validation_rules": {
                "json_schema": {
                    "type": "object",
                    "required": ["enabled"],
                    "properties": {"enabled": {"type": "boolean"}},
                },
                "required_fields": ["enabled"],
            },
        }
    )
    response = UnifiedResponse(
        http_status=200,
        raw_payload={"choices": []},
        raw_response_text="enabled: true\n",
    )

    outcome = validator.validate(case, response)

    assert outcome.passed is True
    assert outcome.parsed_output_json == {"enabled": True}


def test_validator_accepts_any_of_reference_keyword_group() -> None:
    validator = ResponseValidator()
    case = TestCaseDefinition.model_validate(
        {
            "test_case_id": "keyword-group-case",
            "category": "chat",
            "title": "Reference keyword group",
            "description": "The model may mention one of several acceptable root cause categories.",
            "prompt": "Return one likely cause category.",
            "validation_rules": {
                "reference_keywords_any": ["capacity", "configuration"],
            },
        }
    )
    response = UnifiedResponse(
        http_status=200,
        raw_payload={"choices": []},
        raw_response_text="Rollback first. The most likely issue is a configuration regression.",
    )

    outcome = validator.validate(case, response)

    assert outcome.passed is True
    assert outcome.metrics["quality_checks_total"] == 1
    assert outcome.metrics["quality_checks_passed"] == 1


def test_validator_accepts_json_value_ranges_and_variants() -> None:
    validator = ResponseValidator()
    case = TestCaseDefinition.model_validate(
        {
            "test_case_id": "json-range-case",
            "category": "classification",
            "title": "JSON ranges and variants",
            "description": "Allows conservative numeric ranges and one of several accepted enum values.",
            "prompt": "Return JSON with conservative defaults.",
            "expected_format": "json",
            "validation_rules": {
                "json_schema": {
                    "type": "object",
                    "required": ["max_parallel_documents", "request_timeout_seconds", "duplicate_policy"],
                    "properties": {
                        "max_parallel_documents": {"type": "integer"},
                        "request_timeout_seconds": {"type": "integer"},
                        "duplicate_policy": {"type": "string"},
                    },
                },
                "numeric_json_ranges": {
                    "max_parallel_documents": {"min": 1, "max": 2},
                    "request_timeout_seconds": {"min": 60, "max": 300},
                },
                "accepted_json_values": {
                    "duplicate_policy": ["metadata_only", "skip"],
                },
            },
        }
    )
    response = UnifiedResponse(
        http_status=200,
        raw_payload={"choices": []},
        raw_response_text='{"max_parallel_documents": 1, "request_timeout_seconds": 300, "duplicate_policy": "skip"}',
    )

    outcome = validator.validate(case, response)

    assert outcome.passed is True
    assert outcome.metrics["quality_checks_total"] == 3
    assert outcome.metrics["quality_checks_passed"] == 3


def test_qwen_model_without_reasoning_control_gets_hint() -> None:
    model = BenchmarkModelConfig.model_validate(
        {
            "id": "qwen-local",
            "label": "Qwen Local",
            "provider": "local",
            "base_url": "http://localhost:11434/v1",
            "api_type": "openai_compatible",
            "model_name": "qwen3.5:35b-a3b",
            "default_parameters": {"temperature": 0.0},
        }
    )
    controlled_model = BenchmarkModelConfig.model_validate(
        {
            "id": "qwen-local-controlled",
            "label": "Qwen Local Controlled",
            "provider": "local",
            "base_url": "http://localhost:11434/v1",
            "api_type": "openai_compatible",
            "model_name": "qwen3.5:35b-a3b",
            "default_parameters": {"temperature": 0.0, "reasoning_effort": "none"},
        }
    )

    assert model.needs_reasoning_control_hint() is True
    assert controlled_model.needs_reasoning_control_hint() is False
