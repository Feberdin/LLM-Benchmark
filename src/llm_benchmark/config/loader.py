"""
Purpose: Load benchmark configs and test case files from YAML or JSON sources with env interpolation.
Input/Output: Reads files from disk and returns validated Pydantic models.
Important invariants: Invalid files fail before benchmark execution begins, and duplicate test ids are rejected.
How to debug: If a file is skipped or parsed incorrectly, enable debug logs and inspect this module first.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import orjson
import yaml

from llm_benchmark.config.models import BenchmarkConfig
from llm_benchmark.domain.test_case import TestCaseDefinition
from llm_benchmark.utils import expand_env_placeholders

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".yaml", ".yml", ".json"}


def _read_structured_file(path: Path) -> Any:
    """Read JSON or YAML from disk and return native Python objects."""

    suffix = path.suffix.lower()
    content = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return orjson.loads(content)
    return yaml.safe_load(content)


def load_config(path: Path) -> BenchmarkConfig:
    """Load and validate the benchmark configuration."""

    raw = expand_env_placeholders(_read_structured_file(path))
    return BenchmarkConfig.model_validate(raw)


def discover_test_files(tests_dir: Path) -> list[Path]:
    """Collect supported test files in a stable order."""

    if not tests_dir.exists():
        raise FileNotFoundError(f"Tests directory does not exist: {tests_dir}")
    return sorted(
        path
        for path in tests_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and "fixtures" not in path.relative_to(tests_dir).parts
    )


def _resolve_support_file(base_path: Path, relative_path: str) -> Path:
    """Resolve prompt or schema sidecar files relative to the test case location."""

    resolved = (base_path.parent / relative_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Referenced support file does not exist: {resolved}")
    return resolved


def _hydrate_external_test_case_fields(raw: dict[str, Any], *, path: Path) -> dict[str, Any]:
    """
    Expand optional prompt and schema file references before model validation.

    Why this exists:
    Large domain-specific prompts and schemas stay maintainable in dedicated fixture files
    instead of turning YAML test case definitions into hard-to-review walls of text.
    """

    prompt_file = raw.get("prompt_file")
    if prompt_file:
        raw["prompt"] = _resolve_support_file(path, prompt_file).read_text(encoding="utf-8").strip()

    system_prompt_file = raw.get("system_prompt_file")
    if system_prompt_file:
        raw["system_prompt"] = _resolve_support_file(path, system_prompt_file).read_text(encoding="utf-8").strip()

    validation_rules = raw.get("validation_rules")
    if isinstance(validation_rules, dict) and validation_rules.get("json_schema_file"):
        schema_path = _resolve_support_file(path, str(validation_rules["json_schema_file"]))
        validation_rules["json_schema"] = _read_structured_file(schema_path)

    return raw


def load_test_cases(tests_dir: Path) -> list[TestCaseDefinition]:
    """Load all test cases from a directory tree and reject duplicate ids."""

    cases: list[TestCaseDefinition] = []
    seen_ids: set[str] = set()
    for path in discover_test_files(tests_dir):
        LOGGER.debug("Loading test case file: %s", path)
        raw = expand_env_placeholders(_read_structured_file(path))
        if not isinstance(raw, dict):
            raise ValueError(f"Test case file must parse to an object: {path}")
        raw = _hydrate_external_test_case_fields(raw, path=path)
        case = TestCaseDefinition.model_validate(raw)
        if case.test_case_id in seen_ids:
            raise ValueError(f"Duplicate test_case_id detected: {case.test_case_id}")
        seen_ids.add(case.test_case_id)
        cases.append(case)
    return cases


def filter_test_cases_by_suite(test_cases: list[TestCaseDefinition], suite: str | None) -> list[TestCaseDefinition]:
    """Return all cases or only those assigned to the requested suite."""

    if not suite:
        return test_cases
    return [case for case in test_cases if suite in case.suites]
