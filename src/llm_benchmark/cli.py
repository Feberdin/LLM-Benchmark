"""
Purpose: Typer-based CLI for running, validating and reporting LLM benchmarks in Docker or local environments.
Input/Output: Exposes user-facing commands and translates CLI arguments into orchestrator and report operations.
Important invariants: Every command fails with actionable messages instead of stack traces whenever possible.
How to debug: Run commands with `--debug` to enable verbose logging and inspect loaded config plus test paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from llm_benchmark import __version__
from llm_benchmark.config.loader import filter_test_cases_by_suite, load_config, load_test_cases
from llm_benchmark.logging_utils import configure_logging
from llm_benchmark.reporting.builder import build_report_artifacts
from llm_benchmark.reporting.exporters import load_results_from_jsonl, write_report_artifacts
from llm_benchmark.runner.execution import execute_benchmark_run

app = typer.Typer(help="Benchmark multiple OpenAI-compatible LLM endpoints and export structured reports.")


def _resolve_config_path(config: Path | None) -> Path:
    return config or Path(os.getenv("BENCHMARK_CONFIG", "/config/config.yaml"))


def _resolve_tests_dir(tests_dir: Path | None, config_tests_dir: str | None = None) -> Path:
    if tests_dir:
        return tests_dir
    if config_tests_dir:
        return Path(config_tests_dir)
    return Path("/app/tests")


def _resolve_results_dir(results_dir: Path | None, config_results_dir: str | None = None) -> Path:
    if results_dir:
        return results_dir
    if config_results_dir:
        return Path(config_results_dir)
    return Path(os.getenv("RESULTS_DIR", "/app/results"))


@app.callback()
def main_callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        help="Show the installed version and exit.",
    )
) -> None:
    if version:
        typer.echo(f"llm-benchmark {__version__}")
        raise typer.Exit()


@app.command("validate-config")
def validate_config_command(
    config: Path | None = typer.Option(None, "--config", help="Path to the benchmark config file."),
    debug: bool = typer.Option(False, "--debug", help="Enable DEBUG logging."),
) -> None:
    """Validate the benchmark config and print a short summary."""

    configure_logging(os.getenv("LOG_LEVEL", "INFO"), debug=debug)
    try:
        resolved = _resolve_config_path(config)
        benchmark_config = load_config(resolved)
        typer.echo(f"Config OK: {resolved}")
        typer.echo(f"Benchmark name: {benchmark_config.benchmark_name}")
        typer.echo(f"Enabled models: {len(benchmark_config.enabled_models())}/{len(benchmark_config.models)}")
        typer.echo(f"Default tests dir: {benchmark_config.run_defaults.tests_dir}")
        typer.echo(f"Default results dir: {benchmark_config.run_defaults.output_dir}")
    except Exception as exc:
        typer.secho(f"Config validation failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


@app.command("list-models")
def list_models_command(
    config: Path | None = typer.Option(None, "--config", help="Path to the benchmark config file."),
    debug: bool = typer.Option(False, "--debug", help="Enable DEBUG logging."),
) -> None:
    """List configured models with their key runtime settings."""

    configure_logging(os.getenv("LOG_LEVEL", "INFO"), debug=debug)
    benchmark_config = load_config(_resolve_config_path(config))
    for model in benchmark_config.models:
        status = "enabled" if model.enabled else "disabled"
        typer.echo(
            f"{model.id:20} | {status:8} | {model.provider:12} | {model.model_name:24} | {model.base_url}"
        )


@app.command("list-tests")
def list_tests_command(
    config: Path | None = typer.Option(None, "--config", help="Optional config file for default test path resolution."),
    suite: str | None = typer.Option(None, "--suite", help="Optional suite filter."),
    tests_dir: Path | None = typer.Option(None, "--tests-dir", help="Directory with YAML or JSON test case files."),
    debug: bool = typer.Option(False, "--debug", help="Enable DEBUG logging."),
) -> None:
    """List discovered benchmark test cases."""

    configure_logging(os.getenv("LOG_LEVEL", "INFO"), debug=debug)
    benchmark_config = load_config(_resolve_config_path(config)) if config else None
    resolved_tests_dir = _resolve_tests_dir(
        tests_dir,
        benchmark_config.run_defaults.tests_dir if benchmark_config else None,
    )
    test_cases = filter_test_cases_by_suite(load_test_cases(resolved_tests_dir), suite)
    for case in test_cases:
        typer.echo(
            f"{case.test_case_id:24} | {case.category:22} | suites={','.join(case.suites)} | {case.title}"
        )


@app.command("doctor")
def doctor_command(
    config: Path | None = typer.Option(None, "--config", help="Path to the benchmark config file."),
    tests_dir: Path | None = typer.Option(None, "--tests-dir", help="Directory with YAML or JSON test case files."),
    results_dir: Path | None = typer.Option(None, "--results-dir", help="Directory for benchmark outputs."),
    debug: bool = typer.Option(False, "--debug", help="Enable DEBUG logging."),
) -> None:
    """Check local paths, config integrity and missing environment variables."""

    configure_logging(os.getenv("LOG_LEVEL", "INFO"), debug=debug)
    issues: list[str] = []
    benchmark_config = None

    try:
        benchmark_config = load_config(_resolve_config_path(config))
        typer.echo("Config: OK")
    except Exception as exc:
        issues.append(f"Config: {exc}")

    resolved_tests_dir = _resolve_tests_dir(
        tests_dir,
        benchmark_config.run_defaults.tests_dir if benchmark_config else None,
    )
    try:
        test_cases = load_test_cases(resolved_tests_dir)
        typer.echo(f"Tests: OK ({len(test_cases)} discovered in {resolved_tests_dir})")
    except Exception as exc:
        issues.append(f"Tests: {exc}")

    resolved_results_dir = _resolve_results_dir(
        results_dir,
        benchmark_config.run_defaults.output_dir if benchmark_config else None,
    )
    try:
        resolved_results_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Results dir: OK ({resolved_results_dir})")
    except Exception as exc:
        issues.append(f"Results dir: {exc}")

    if benchmark_config:
        missing_keys = [
            model.api_key_env
            for model in benchmark_config.enabled_models()
            if model.api_key_env and not os.getenv(model.api_key_env)
        ]
        if missing_keys:
            issues.append(
                "Missing required API key environment variables: " + ", ".join(sorted(set(missing_keys)))
            )

    if issues:
        for issue in issues:
            typer.secho(f"Doctor issue: {issue}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    typer.secho("Doctor checks passed.", fg=typer.colors.GREEN)


@app.command("run")
def run_command(
    config: Path | None = typer.Option(None, "--config", help="Path to the benchmark config file."),
    suite: str | None = typer.Option(None, "--suite", help="Limit execution to a named suite."),
    tests_dir: Path | None = typer.Option(None, "--tests-dir", help="Directory with YAML or JSON test case files."),
    results_dir: Path | None = typer.Option(None, "--results-dir", help="Directory for benchmark outputs."),
    debug: bool = typer.Option(False, "--debug", help="Enable DEBUG logging."),
) -> None:
    """Execute the benchmark suite and write all report artifacts."""

    configure_logging(os.getenv("LOG_LEVEL", "INFO"), debug=debug)
    try:
        benchmark_config = load_config(_resolve_config_path(config))
        resolved_tests_dir = _resolve_tests_dir(tests_dir, benchmark_config.run_defaults.tests_dir)
        resolved_results_dir = _resolve_results_dir(results_dir, benchmark_config.run_defaults.output_dir)
        execution = execute_benchmark_run(
            config_path=_resolve_config_path(config),
            tests_dir=resolved_tests_dir,
            results_dir=resolved_results_dir,
            suite=suite,
        )

        typer.secho(
            f"Benchmark finished successfully. Artifacts written to {resolved_results_dir}",
            fg=typer.colors.GREEN,
        )
    except Exception as exc:
        typer.secho(f"Benchmark run failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


@app.command("report")
def report_command(
    input: Path = typer.Option(..., "--input", help="Path to a raw_runs.jsonl file."),
    output: Path | None = typer.Option(None, "--output", help="Directory for regenerated report artifacts."),
    benchmark_name: str = typer.Option("llm-benchmark", "--benchmark-name", help="Name for regenerated reports."),
    suite: str | None = typer.Option(None, "--suite", help="Optional suite name for the regenerated report."),
    debug: bool = typer.Option(False, "--debug", help="Enable DEBUG logging."),
) -> None:
    """Regenerate aggregate reports from an existing `raw_runs.jsonl` export."""

    configure_logging(os.getenv("LOG_LEVEL", "INFO"), debug=debug)
    try:
        results = load_results_from_jsonl(input)
        if not results:
            raise ValueError("The input JSONL file does not contain any benchmark records.")
        output_dir = output or input.parent
        first_record = results[0]
        artifacts = build_report_artifacts(
            results=results,
            benchmark_name=benchmark_name,
            benchmark_run_id=first_record.benchmark_run_id,
            suite=suite,
            run_started_at=first_record.run_started_at,
            run_finished_at=results[-1].run_finished_at,
            environment_info={"source": str(input)},
            config=None,
            test_cases=None,
        )
        write_report_artifacts(artifacts, output_dir)
        typer.secho(f"Reports regenerated in {output_dir}", fg=typer.colors.GREEN)
    except Exception as exc:
        typer.secho(f"Report generation failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


@app.command("dashboard")
def dashboard_command(
    config: Path | None = typer.Option(None, "--config", help="Optional config file for default path resolution."),
    tests_dir: Path | None = typer.Option(None, "--tests-dir", help="Directory with YAML or JSON test case files."),
    results_dir: Path | None = typer.Option(None, "--results-dir", help="Directory with benchmark outputs."),
    host: str = typer.Option("0.0.0.0", "--host", help="Dashboard bind host."),
    port: int = typer.Option(8080, "--port", help="Dashboard bind port."),
    debug: bool = typer.Option(False, "--debug", help="Enable DEBUG logging."),
) -> None:
    """Serve the integrated benchmark dashboard with live run controls."""

    configure_logging(os.getenv("LOG_LEVEL", "INFO"), debug=debug)
    resolved_config_path = _resolve_config_path(config)
    benchmark_config = load_config(resolved_config_path) if resolved_config_path.exists() else None
    resolved_tests_dir = _resolve_tests_dir(
        tests_dir,
        benchmark_config.run_defaults.tests_dir if benchmark_config else None,
    )
    resolved_results_dir = _resolve_results_dir(
        results_dir,
        benchmark_config.run_defaults.output_dir if benchmark_config else None,
    )

    try:
        from llm_benchmark.dashboard.app import run_dashboard

        run_dashboard(
            config_path=resolved_config_path,
            results_dir=resolved_results_dir,
            tests_dir=resolved_tests_dir,
            host=host,
            port=port,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
    except Exception as exc:
        typer.secho(f"Dashboard startup failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


def main() -> None:
    """Console entrypoint used by the `benchmark` script."""

    app()
