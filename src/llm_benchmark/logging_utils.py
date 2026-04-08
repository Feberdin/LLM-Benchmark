"""
Purpose: Central logging configuration for CLI, Docker and Unraid runtime scenarios.
Input/Output: Called once at process startup to configure reproducible structured-ish logs.
Important invariants: Sensitive values are not logged directly and debug mode is opt-in.
How to debug: If logs are too quiet or too noisy, adjust this module and the `LOG_LEVEL` env var.
"""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO", *, debug: bool = False) -> None:
    """Configure application-wide logging using a format that works in terminals and container logs."""

    effective_level = "DEBUG" if debug else level.upper()
    logging.basicConfig(
        level=getattr(logging, effective_level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
