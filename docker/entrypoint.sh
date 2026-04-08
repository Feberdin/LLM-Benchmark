#!/bin/sh
# Purpose: Container entrypoint that supports explicit CLI arguments and an optional env-driven Unraid auto-run mode.
# Input/Output: Executes the `benchmark` CLI with either passed arguments or derived runtime defaults.
# Invariants: Explicit command-line arguments always win over environment-driven automation.
# Debugging: Set `BENCHMARK_DEBUG=true` and inspect the effective env vars when startup behavior looks unexpected.

set -eu

to_lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

AUTO_RUN="$(to_lower "${BENCHMARK_AUTO_RUN:-false}")"
DEBUG_FLAG="$(to_lower "${BENCHMARK_DEBUG:-false}")"

if [ "$#" -gt 0 ]; then
  exec benchmark "$@"
fi

if [ "$AUTO_RUN" = "true" ] || [ "$AUTO_RUN" = "1" ] || [ "$AUTO_RUN" = "yes" ]; then
  ACTION="${BENCHMARK_ACTION:-run}"
  set -- "$ACTION"

  if [ "$ACTION" = "run" ] || [ "$ACTION" = "doctor" ] || [ "$ACTION" = "validate-config" ] || [ "$ACTION" = "list-models" ] || [ "$ACTION" = "dashboard" ]; then
    set -- "$@" --config "${BENCHMARK_CONFIG:-/config/config.unraid.example.yaml}"
  fi

  if [ "$ACTION" = "run" ] || [ "$ACTION" = "doctor" ] || [ "$ACTION" = "dashboard" ]; then
    set -- "$@" --tests-dir "${BENCHMARK_TESTS_DIR:-/app/tests}" --results-dir "${BENCHMARK_RESULTS_DIR:-/app/results}"
  fi

  if [ "$ACTION" = "run" ] && [ -n "${BENCHMARK_SUITE:-}" ]; then
    set -- "$@" --suite "${BENCHMARK_SUITE}"
  fi

  if [ "$ACTION" = "report" ] && [ -n "${BENCHMARK_INPUT_JSONL:-}" ]; then
    set -- "$@" --input "${BENCHMARK_INPUT_JSONL}" --output "${BENCHMARK_RESULTS_DIR:-/app/results}"
  fi

  if [ "$ACTION" = "dashboard" ]; then
    set -- "$@" --host "${DASHBOARD_HOST:-0.0.0.0}" --port "${DASHBOARD_PORT:-8080}"
  fi

  if [ "$DEBUG_FLAG" = "true" ] || [ "$DEBUG_FLAG" = "1" ] || [ "$DEBUG_FLAG" = "yes" ]; then
    set -- "$@" --debug
  fi

  exec benchmark "$@"
fi

exec benchmark --help
