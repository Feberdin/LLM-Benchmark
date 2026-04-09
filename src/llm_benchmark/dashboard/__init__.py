"""
Purpose: Web dashboard package for browsing benchmark results, project-fit recommendations and dashboard-triggered runs.
Input/Output: Exposes a FastAPI application factory plus service helpers for HTML, JSON views and lightweight run control.
Important invariants: The dashboard reuses the same benchmark execution path as the CLI and never implements a second runner.
How to debug: Start with `/health`, then inspect `/api/dashboard/run/current` and the persisted dashboard state files.
"""
