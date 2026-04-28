"""
Purpose: Package metadata and a single import location for the benchmark application version.
Input/Output: Imported by the CLI and tests.
Important invariants: The version string stays in sync with the package metadata during releases.
How to debug: If CLI output shows an unexpected version, check this file and `pyproject.toml`.
"""

__all__ = ["__version__"]

__version__ = "0.2.0"
