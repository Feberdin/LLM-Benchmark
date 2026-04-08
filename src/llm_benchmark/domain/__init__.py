"""
Purpose: Domain models shared across config loading, execution, validation and reporting.
Input/Output: Provides typed contracts for test cases, responses and persisted results.
Important invariants: These models remain stable because downstream reports depend on them.
How to debug: If serialization breaks, inspect recent field changes in this package.
"""
