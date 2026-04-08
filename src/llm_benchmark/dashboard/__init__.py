"""
Purpose: Read-only web dashboard package for browsing benchmark results and project-fit recommendations.
Input/Output: Exposes a FastAPI application factory plus service helpers for HTML and JSON views.
Important invariants: The dashboard never mutates benchmark results and always reads from existing artifacts.
How to debug: Start with `/health`, then inspect the dashboard service snapshot data if a view looks wrong.
"""

