"""
Purpose: Client package for LLM backends and transport-specific helpers.
Input/Output: Imported by the runner to talk to OpenAI-compatible APIs.
Important invariants: Client modules expose backend-agnostic response structures to the rest of the app.
How to debug: Start in this package if requests fail before validation or scoring starts.
"""
