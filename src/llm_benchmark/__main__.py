"""
Purpose: Allow `python -m llm_benchmark` to behave exactly like the installed `benchmark` CLI.
Input/Output: Delegates straight into the Typer application entrypoint.
Important invariants: This module stays tiny so CLI behavior cannot drift between entry methods.
How to debug: If `python -m llm_benchmark` behaves differently than `benchmark`, check this file first.
"""

from llm_benchmark.cli import main

if __name__ == "__main__":
    main()
