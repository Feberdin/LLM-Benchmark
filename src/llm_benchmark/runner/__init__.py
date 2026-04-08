"""
Purpose: Runner package for orchestration, execution order and benchmark scoring.
Input/Output: Coordinates model invocations, validation and persistence-ready results.
Important invariants: A single failed request must never abort the whole benchmark run.
How to debug: Use this package when execution order, retries or scoring seem off.
"""
