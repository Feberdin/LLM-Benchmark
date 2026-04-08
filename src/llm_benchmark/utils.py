"""
Purpose: Shared helper functions for paths, timestamps, hashing, JSON encoding and environment expansion.
Input/Output: Called across the whole application to keep low-level behavior consistent.
Important invariants: JSON serialization is deterministic enough for stable reports and prompt hashes.
How to debug: If paths, timestamps or env placeholder replacement look wrong, start in this module.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(:-([^}]*))?\}")


def utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(tz=UTC)


def isoformat_utc(value: datetime | None = None) -> str:
    """Return an ISO 8601 string with a `Z` suffix for UTC timestamps."""

    moment = value or utcnow()
    return moment.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_directory(path: Path) -> Path:
    """Create a directory when needed and return the normalized path."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_text(value: str) -> str:
    """Create a stable hash for prompts and other text payloads."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_dumps(value: Any, *, pretty: bool = False) -> bytes:
    """Serialize JSON using `orjson` with sane defaults for datetimes and non-string keys."""

    option = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
    if pretty:
        option |= orjson.OPT_INDENT_2
    return orjson.dumps(value, option=option)


def json_dump_to_path(path: Path, value: Any, *, pretty: bool = False) -> None:
    """Write JSON atomically enough for single-process benchmark runs."""

    path.write_bytes(json_dumps(value, pretty=pretty))


def mask_secret(secret: str | None) -> str | None:
    """Hide most of a secret while still making debugging possible."""

    if not secret:
        return secret
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:3]}***{secret[-3:]}"


def expand_env_placeholders(value: Any) -> Any:
    """
    Recursively expand `${VAR}` and `${VAR:-default}` placeholders in configs.

    Example input/output:
    - `${OPENAI_BASE_URL}` -> `https://api.openai.com/v1`
    - `${MISSING_VAR:-fallback}` -> `fallback`
    """

    if isinstance(value, dict):
        return {key: expand_env_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_placeholders(item) for item in value]
    if not isinstance(value, str):
        return value

    def _replace(match: re.Match[str]) -> str:
        variable = match.group(1)
        default = match.group(3)
        resolved = os.getenv(variable, default)
        if resolved is None:
            raise ValueError(
                f"Environment variable '{variable}' is referenced in configuration but not set, "
                "and no default value was provided."
            )
        return resolved

    return ENV_PATTERN.sub(_replace, value)


def build_environment_info() -> dict[str, Any]:
    """Capture lightweight environment metadata for the final report."""

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "is_containerized": Path("/.dockerenv").exists(),
        "timezone": str(datetime.now().astimezone().tzinfo),
        "selected_environment_variables": {
            "BENCHMARK_CONFIG": os.getenv("BENCHMARK_CONFIG"),
            "BENCHMARK_ACTION": os.getenv("BENCHMARK_ACTION"),
            "RESULTS_DIR": os.getenv("RESULTS_DIR"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL"),
            "DEFAULT_TIMEOUT_SECONDS": os.getenv("DEFAULT_TIMEOUT_SECONDS"),
            "DASHBOARD_PORT": os.getenv("DASHBOARD_PORT"),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
            "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL"),
            "OPENAI_API_KEY": mask_secret(os.getenv("OPENAI_API_KEY")),
        },
    }
