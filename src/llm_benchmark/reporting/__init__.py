"""
Purpose: Reporting package for CSV, Markdown, HTML and analysis-focused JSON outputs.
Input/Output: Consumes normalized run results and produces reusable benchmark artifacts.
Important invariants: The final JSON schema remains stable for machine-driven downstream analysis.
How to debug: If summaries or rankings look wrong, trace the aggregation logic from this package.
"""
