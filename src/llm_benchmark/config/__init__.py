"""
Purpose: Configuration package for benchmark settings and loader utilities.
Input/Output: Turns YAML or JSON files plus environment variables into validated runtime models.
Important invariants: Invalid configuration fails fast before any benchmark traffic is sent.
How to debug: Use `benchmark validate-config` to inspect this layer first.
"""
