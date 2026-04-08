"""
Purpose: Validation package for response format, instruction and rule-based checks.
Input/Output: Converts raw model output into validation outcomes and structured issue lists.
Important invariants: Validation errors are explicit and never silently swallowed.
How to debug: Check this package when JSON parsing or format-treue scoring looks suspicious.
"""
