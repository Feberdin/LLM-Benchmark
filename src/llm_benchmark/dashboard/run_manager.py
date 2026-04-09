"""
Purpose: Manage dashboard-triggered benchmark runs, preflight checks and operator-friendly progress snapshots.
Input/Output: Accepts a suite name plus the configured paths, runs the existing benchmark pipeline in a background thread and persists a JSON status snapshot.
Important invariants: Only one dashboard run can execute at a time, and every failure must become a readable status entry instead of silently disappearing.
How to debug: Inspect `dashboard_run_state.json` and `dashboard_run_history.json` in the results directory, then compare the timeline events with the raw benchmark artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
import os
from pathlib import Path
from threading import RLock, Thread
from typing import Any
import logging
import uuid

import httpx
import orjson

from llm_benchmark.config.loader import filter_test_cases_by_suite, load_config, load_test_cases
from llm_benchmark.dashboard.service import REPORT_FILES
from llm_benchmark.runner.execution import execute_benchmark_run
from llm_benchmark.utils import ensure_directory, isoformat_utc, json_dump_to_path

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TimelineEvent:
    """One operator-visible progress event for the dashboard timeline."""

    timestamp: str
    event: str
    stage: str
    level: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunState:
    """Current background run snapshot that the UI can poll and render."""

    run_id: str | None = None
    benchmark_run_id: str | None = None
    status: str = "idle"
    suite: str | None = None
    config_path: str | None = None
    tests_dir: str | None = None
    results_dir: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    current_stage: str | None = None
    current_step: str | None = None
    current_model_label: str | None = None
    current_test_case_id: str | None = None
    total_planned_records: int = 0
    completed_records: int = 0
    successful_records: int = 0
    failed_records: int = 0
    available_suites: list[str] = field(default_factory=list)
    enabled_models: list[str] = field(default_factory=list)
    discovered_test_case_count: int = 0
    selected_test_case_count: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    latest_error: str | None = None
    generated_files: list[str] = field(default_factory=list)
    events: list[TimelineEvent] = field(default_factory=list)


@dataclass(slots=True)
class ConnectivityResult:
    """One lightweight reachability probe result for a configured model endpoint."""

    model_id: str
    model_label: str
    provider: str
    endpoint: str
    model_name: str
    ok: bool
    status: str
    message: str
    http_status: int | None = None
    duration_ms: float | None = None
    model_listed: bool | None = None
    listed_models_preview: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ConnectivityState:
    """Current status and latest results of the dashboard-triggered connectivity check."""

    check_id: str | None = None
    status: str = "idle"
    started_at: str | None = None
    finished_at: str | None = None
    current_model_label: str | None = None
    total_models: int = 0
    checked_models: int = 0
    reachable_models: int = 0
    failed_models: int = 0
    summary: str | None = None
    latest_error: str | None = None
    results: list[ConnectivityResult] = field(default_factory=list)
    events: list[TimelineEvent] = field(default_factory=list)


class DashboardRunManager:
    """
    Coordinate benchmark runs triggered from the web dashboard.

    Why this exists:
    The dashboard should stay lightweight and reuse the same execution path as the CLI, but operators still
    need a stable place to start a run, observe progress and understand why a run did not even begin.
    """

    def __init__(self, *, config_path: Path, tests_dir: Path, results_dir: Path) -> None:
        self.config_path = config_path
        self.tests_dir = tests_dir
        self.results_dir = ensure_directory(results_dir)
        self.state_path = self.results_dir / "dashboard_run_state.json"
        self.history_path = self.results_dir / "dashboard_run_history.json"
        self.connectivity_state_path = self.results_dir / "dashboard_connectivity_state.json"
        self._lock = RLock()
        self._active_thread: Thread | None = None
        self._connectivity_thread: Thread | None = None
        self._preflight_cache: dict[str, dict[str, Any]] = {}
        self._state = self._load_state()
        self._history = self._load_history()
        self._connectivity_state = self._load_connectivity_state()
        if self._state.status in {"queued", "running"}:
            self._state.status = "interrupted"
            self._state.current_stage = "interrupted"
            self._state.current_step = (
                "Ein frueherer Dashboard-Lauf wurde nicht sauber beendet, zum Beispiel durch einen Container-Neustart."
            )
            self._append_event_locked(
                event="dashboard_run_interrupted",
                stage="interrupted",
                level="warning",
                summary="Vorheriger Dashboard-Lauf wurde beim Neustart oder Stoppen unterbrochen.",
                details={},
            )
            with self._lock:
                self._persist_locked()

    def current_payload(self, suite: str | None = None, *, include_preflight: bool = True) -> dict[str, Any]:
        """Return the current state plus preflight details and recent history for the UI."""

        with self._lock:
            state = asdict(self._state)
            history = list(self._history[:12])
        preflight = self.preflight(suite=suite) if include_preflight else self._cached_preflight(suite=suite)
        return {
            "state": state,
            "preflight": preflight,
            "history": history,
            "connectivity": asdict(self._connectivity_state),
        }

    def history_payload(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history[:20])

    def connectivity_payload(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self._connectivity_state)

    def start_run(self, *, suite: str | None) -> dict[str, Any]:
        """Start a background benchmark run or raise an actionable operator error."""

        normalized_suite = None if not suite or suite == "all" else suite
        preflight = self.preflight(suite=normalized_suite)
        if preflight["errors"]:
            raise ValueError("Preflight fehlgeschlagen: " + " | ".join(preflight["errors"]))

        with self._lock:
            if self._active_thread and self._active_thread.is_alive():
                raise RuntimeError("Es laeuft bereits ein Benchmark. Bitte warte, bis dieser Lauf abgeschlossen ist.")

            run_id = str(uuid.uuid4())
            self._state = RunState(
                run_id=run_id,
                status="queued",
                suite=normalized_suite,
                config_path=str(self.config_path),
                tests_dir=str(self.tests_dir),
                results_dir=str(self.results_dir),
                started_at=isoformat_utc(),
                current_stage="queued",
                current_step="Benchmark wurde ueber das Dashboard eingeplant.",
                available_suites=preflight["available_suites"],
                enabled_models=preflight["enabled_models"],
                discovered_test_case_count=preflight["discovered_test_case_count"],
                selected_test_case_count=preflight["selected_test_case_count"],
                warnings=preflight["warnings"],
                errors=[],
                latest_error=None,
                generated_files=[],
                events=[
                    TimelineEvent(
                        timestamp=isoformat_utc(),
                        event="queued",
                        stage="queued",
                        level="info",
                        summary="Benchmarklauf wurde in die Hintergrundausfuehrung uebernommen.",
                        details={"suite": normalized_suite or "all"},
                    )
                ],
            )
            self._persist_locked()
            self._active_thread = Thread(
                target=self._run_in_background,
                kwargs={"run_id": run_id, "suite": normalized_suite},
                daemon=True,
                name=f"dashboard-benchmark-{run_id[:8]}",
            )
            self._active_thread.start()
            return asdict(self._state)

    def start_connectivity_check(self) -> dict[str, Any]:
        """Run a lightweight reachability check for all enabled model endpoints."""

        benchmark_config = load_config(self.config_path)
        enabled_models = benchmark_config.enabled_models()
        if not enabled_models:
            raise ValueError("Keine aktiven Modelle in der Konfiguration gefunden.")

        with self._lock:
            if self._active_thread and self._active_thread.is_alive():
                raise RuntimeError("Bitte warte mit dem Erreichbarkeits-Check, bis der laufende Benchmark beendet ist.")
            if self._connectivity_thread and self._connectivity_thread.is_alive():
                raise RuntimeError("Es laeuft bereits ein Erreichbarkeits-Check.")

            check_id = str(uuid.uuid4())
            self._connectivity_state = ConnectivityState(
                check_id=check_id,
                status="queued",
                started_at=isoformat_utc(),
                total_models=len(enabled_models),
                summary="Erreichbarkeits-Check wurde eingeplant.",
                events=[
                    TimelineEvent(
                        timestamp=isoformat_utc(),
                        event="connectivity_check_queued",
                        stage="queued",
                        level="info",
                        summary="LLM-Erreichbarkeits-Check wurde gestartet.",
                        details={"model_count": len(enabled_models)},
                    )
                ],
            )
            self._persist_locked()
            self._connectivity_thread = Thread(
                target=self._run_connectivity_check,
                kwargs={"check_id": check_id},
                daemon=True,
                name=f"dashboard-connectivity-{check_id[:8]}",
            )
            self._connectivity_thread.start()
            return asdict(self._connectivity_state)

    def preflight(self, *, suite: str | None) -> dict[str, Any]:
        """Validate config, tests and required secrets before a dashboard-triggered run starts."""

        normalized_suite = None if not suite or suite == "all" else suite
        cache_key = normalized_suite or "all"
        signature = self._preflight_signature()
        cached = self._preflight_cache.get(cache_key)
        if cached and cached.get("_signature") == signature:
            return {key: value for key, value in cached.items() if key != "_signature"}

        warnings: list[str] = []
        errors: list[str] = []
        available_suites: list[str] = []
        enabled_models: list[str] = []
        discovered_test_case_count = 0
        selected_test_case_count = 0

        if not self.config_path.exists():
            errors.append(
                f"Konfigurationsdatei fehlt: {self.config_path}. "
                "Lege zuerst /config/config.unraid.example.yaml oder eine eigene Konfiguration an."
            )
        else:
            try:
                benchmark_config = load_config(self.config_path)
                enabled_models = [model.label for model in benchmark_config.enabled_models()]
                missing_keys = sorted(
                    {
                        model.api_key_env
                        for model in benchmark_config.enabled_models()
                        if model.api_key_env and not os.getenv(model.api_key_env)
                    }
                )
                if missing_keys:
                    warnings.append(
                        "Folgende API-Key-Variablen fehlen derzeit: "
                        + ", ".join(missing_keys)
                        + ". Die betroffenen Modelle werden wahrscheinlich fehlschlagen."
                    )
            except Exception as exc:
                errors.append(f"Konfiguration konnte nicht geladen werden: {exc}")

        if not self.tests_dir.exists():
            errors.append(
                f"Tests-Verzeichnis fehlt: {self.tests_dir}. "
                "Wenn /app/tests von Unraid gemountet wird, muessen dort YAML- oder JSON-Testdateien liegen."
            )
        else:
            try:
                test_cases = load_test_cases(self.tests_dir)
                discovered_test_case_count = len(test_cases)
                available_suites = sorted({suite_name for case in test_cases for suite_name in case.suites})
                selected_test_case_count = len(filter_test_cases_by_suite(test_cases, normalized_suite))
                if not test_cases:
                    errors.append(
                        f"Keine Testfalldateien in {self.tests_dir} gefunden. "
                        "Ein leeres Unraid-Mount ueberdeckt die eingebauten Beispieltests im Container."
                    )
                elif normalized_suite and selected_test_case_count == 0:
                    errors.append(
                        f"Die Suite '{normalized_suite}' hat in {self.tests_dir} keine Testfaelle. "
                        "Pruefe den Suite-Namen oder kopiere die passenden YAML-Dateien in dein gemountetes Tests-Verzeichnis."
                    )
            except Exception as exc:
                errors.append(f"Tests konnten nicht geladen werden: {exc}")

        recommended_suite = self._recommended_suite(available_suites)
        payload = {
            "config_path": str(self.config_path),
            "tests_dir": str(self.tests_dir),
            "results_dir": str(self.results_dir),
            "available_suites": available_suites,
            "recommended_suite": recommended_suite,
            "enabled_models": enabled_models,
            "discovered_test_case_count": discovered_test_case_count,
            "selected_test_case_count": selected_test_case_count,
            "warnings": warnings,
            "errors": errors,
        }
        self._preflight_cache[cache_key] = payload | {"_signature": signature}
        return payload

    def _run_in_background(self, *, run_id: str, suite: str | None) -> None:
        """
        Execute the benchmark in a worker thread and translate internal progress into stable dashboard state.

        Example:
        - Event: `run_step_finished` with `success=false`
        - Visible summary: "Qwen Local / quick-json-extract schlug fehl (timeout)"
        """

        self._set_running_locked(run_id)
        try:
            execution = execute_benchmark_run(
                config_path=self.config_path,
                tests_dir=self.tests_dir,
                results_dir=self.results_dir,
                suite=suite,
                progress_callback=lambda payload: self._handle_progress(run_id=run_id, payload=payload),
            )
            with self._lock:
                if self._state.run_id != run_id:
                    return
                self._state.status = "succeeded"
                self._state.finished_at = isoformat_utc()
                self._state.current_stage = "completed"
                self._state.current_step = "Benchmarklauf erfolgreich abgeschlossen."
                self._state.benchmark_run_id = execution.run_summary.benchmark_run_id
                self._state.generated_files = self._discover_generated_files()
                self._append_event_locked(
                    event="dashboard_run_succeeded",
                    stage="completed",
                    level="success",
                    summary="Alle Reports wurden geschrieben und koennen jetzt im Dashboard geoeffnet werden.",
                    details={
                        "benchmark_run_id": execution.run_summary.benchmark_run_id,
                        "generated_files": self._state.generated_files,
                    },
                )
                self._push_history_locked()
                self._persist_locked()
        except Exception as exc:
            LOGGER.exception("Dashboard-triggered benchmark run failed.")
            with self._lock:
                if self._state.run_id != run_id:
                    return
                self._state.status = "failed"
                self._state.finished_at = isoformat_utc()
                self._state.current_stage = "failed"
                self._state.current_step = "Benchmarklauf wurde mit einem Fehler beendet."
                self._state.latest_error = str(exc)
                self._state.errors = [str(exc)]
                self._append_event_locked(
                    event="dashboard_run_failed",
                    stage="failed",
                    level="error",
                    summary=f"Benchmarklauf fehlgeschlagen: {exc}",
                    details={"error": str(exc)},
                )
                self._push_history_locked()
                self._persist_locked()

    def _run_connectivity_check(self, *, check_id: str) -> None:
        try:
            benchmark_config = load_config(self.config_path)
            enabled_models = benchmark_config.enabled_models()
            with self._lock:
                if self._connectivity_state.check_id != check_id:
                    return
                self._connectivity_state.status = "running"
                self._connectivity_state.summary = "LLM-Endpunkte werden nacheinander geprueft."
                self._append_connectivity_event_locked(
                    event="connectivity_check_started",
                    stage="running",
                    level="info",
                    summary=f"Pruefe {len(enabled_models)} Modellziele auf Erreichbarkeit.",
                    details={"model_count": len(enabled_models)},
                )
                self._persist_locked()

            for model_config in enabled_models:
                with self._lock:
                    if self._connectivity_state.check_id != check_id:
                        return
                    self._connectivity_state.current_model_label = model_config.label
                    self._append_connectivity_event_locked(
                        event="connectivity_probe_started",
                        stage="running",
                        level="info",
                        summary=f"Pruefe {model_config.label} ueber {model_config.base_url}/models.",
                        details={"model_id": model_config.id, "endpoint": model_config.base_url},
                    )
                    self._persist_locked()

                result = self._probe_model_endpoint(
                    model_config=model_config,
                    default_timeout_seconds=benchmark_config.run_defaults.default_timeout_seconds,
                )

                with self._lock:
                    if self._connectivity_state.check_id != check_id:
                        return
                    self._connectivity_state.results.append(result)
                    self._connectivity_state.checked_models += 1
                    if result.ok:
                        self._connectivity_state.reachable_models += 1
                    else:
                        self._connectivity_state.failed_models += 1
                        self._connectivity_state.latest_error = result.message
                    self._append_connectivity_event_locked(
                        event="connectivity_probe_finished",
                        stage="running",
                        level="success" if result.ok else "error",
                        summary=f"{result.model_label}: {result.message}",
                        details=asdict(result),
                    )
                    self._persist_locked()

            with self._lock:
                if self._connectivity_state.check_id != check_id:
                    return
                self._connectivity_state.status = (
                    "succeeded" if self._connectivity_state.failed_models == 0 else "failed"
                )
                self._connectivity_state.finished_at = isoformat_utc()
                self._connectivity_state.current_model_label = None
                self._connectivity_state.summary = (
                    f"{self._connectivity_state.reachable_models} von {self._connectivity_state.total_models} "
                    "Modellzielen antworten auf den Reachability-Check."
                )
                self._append_connectivity_event_locked(
                    event="connectivity_check_finished",
                    stage="completed",
                    level="success" if self._connectivity_state.failed_models == 0 else "warning",
                    summary=self._connectivity_state.summary,
                    details={},
                )
                self._persist_locked()
        except Exception as exc:
            LOGGER.exception("Dashboard connectivity check failed.")
            with self._lock:
                if self._connectivity_state.check_id != check_id:
                    return
                self._connectivity_state.status = "failed"
                self._connectivity_state.finished_at = isoformat_utc()
                self._connectivity_state.latest_error = str(exc)
                self._connectivity_state.summary = f"Erreichbarkeits-Check fehlgeschlagen: {exc}"
                self._append_connectivity_event_locked(
                    event="connectivity_check_failed",
                    stage="failed",
                    level="error",
                    summary=self._connectivity_state.summary,
                    details={"error": str(exc)},
                )
                self._persist_locked()

    def _set_running_locked(self, run_id: str) -> None:
        with self._lock:
            if self._state.run_id != run_id:
                return
            self._state.status = "running"
            self._state.current_stage = "starting"
            self._state.current_step = "Benchmark-Engine startet und bereitet die Ausfuehrung vor."
            self._append_event_locked(
                event="dashboard_run_started",
                stage="starting",
                level="info",
                summary="Hintergrundlauf gestartet. Fortschritt wird jetzt laufend aktualisiert.",
                details={"suite": self._state.suite or "all"},
            )
            self._persist_locked()

    def _handle_progress(self, *, run_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            if self._state.run_id != run_id:
                return

            event = str(payload.get("event", "progress"))
            stage = str(payload.get("stage", "running"))
            self._state.current_stage = stage

            if event == "test_discovery_completed":
                self._state.selected_test_case_count = int(payload.get("selected_test_case_count", 0))
            elif event == "run_started":
                self._state.benchmark_run_id = str(payload.get("benchmark_run_id"))
                self._state.total_planned_records = int(payload.get("total_planned_records", 0))
            elif event == "run_step_started":
                self._state.current_model_label = payload.get("model_label")
                self._state.current_test_case_id = payload.get("test_case_id")
            elif event == "run_step_finished":
                self._state.completed_records += 1
                if payload.get("success") and payload.get("validation_passed"):
                    self._state.successful_records += 1
                else:
                    self._state.failed_records += 1
                    self._state.latest_error = str(
                        payload.get("error_message")
                        or payload.get("error_type")
                        or "Der Schritt wurde nicht erfolgreich abgeschlossen."
                    )
            elif event == "run_finished":
                self._state.successful_records = int(payload.get("successful_records", self._state.successful_records))
                self._state.failed_records = int(payload.get("failed_records", self._state.failed_records))

            summary, level = self._summarize_event(payload)
            self._state.current_step = summary
            self._append_event_locked(
                event=event,
                stage=stage,
                level=level,
                summary=summary,
                details=payload,
            )
            self._persist_locked()

    def _summarize_event(self, payload: dict[str, Any]) -> tuple[str, str]:
        event = str(payload.get("event", "progress"))

        if event == "loading_config":
            return "Konfiguration wird geladen.", "info"
        if event == "discovering_tests":
            suite = payload.get("suite") or "all"
            return f"Tests werden fuer die Suite '{suite}' gesucht.", "info"
        if event == "test_discovery_completed":
            return (
                f"{payload.get('selected_test_case_count', 0)} Testfaelle fuer "
                f"{payload.get('selected_model_count', 0)} Modellziele ausgewaehlt.",
                "success",
            )
        if event == "run_started":
            return (
                f"Benchmarklauf gestartet: {payload.get('model_count', 0)} Modelle, "
                f"{payload.get('test_case_count', 0)} Testfaelle, "
                f"{payload.get('total_planned_records', 0)} geplante Schritte.",
                "info",
            )
        if event == "run_step_started":
            return (
                f"{payload.get('model_label', 'Modell')} bearbeitet "
                f"{payload.get('test_case_id', 'Test')} ({payload.get('phase', 'run')} #{payload.get('repetition_index', 0)}).",
                "info",
            )
        if event == "run_step_finished":
            if payload.get("success") and payload.get("validation_passed"):
                return (
                    f"{payload.get('model_label', 'Modell')} / {payload.get('test_case_id', 'Test')} "
                    f"erfolgreich in {payload.get('duration_ms', '?')} ms abgeschlossen.",
                    "success",
                )
            return (
                f"{payload.get('model_label', 'Modell')} / {payload.get('test_case_id', 'Test')} "
                f"schlug fehl ({payload.get('error_type') or 'validation'}).",
                "error",
            )
        if event == "writing_raw_results":
            return "Rohdaten werden als JSONL und CSV geschrieben.", "info"
        if event == "building_reports":
            return "Aggregierte Reports und Analyse-Exports werden erzeugt.", "info"
        if event == "reports_written":
            return "Reports wurden in das Ergebnisverzeichnis geschrieben.", "success"
        if event == "run_finished":
            return (
                f"Benchmarklauf beendet: {payload.get('successful_records', 0)} erfolgreiche und "
                f"{payload.get('failed_records', 0)} problematische Datensaetze.",
                "success" if int(payload.get("failed_records", 0)) == 0 else "warning",
            )
        return "Fortschritt aktualisiert.", "info"

    def _append_event_locked(
        self,
        *,
        event: str,
        stage: str,
        level: str,
        summary: str,
        details: dict[str, Any],
    ) -> None:
        self._state.events.insert(
            0,
            TimelineEvent(
                timestamp=isoformat_utc(),
                event=event,
                stage=stage,
                level=level,
                summary=summary,
                details=details,
            ),
        )
        self._state.events = self._state.events[:120]

    def _append_connectivity_event_locked(
        self,
        *,
        event: str,
        stage: str,
        level: str,
        summary: str,
        details: dict[str, Any],
    ) -> None:
        self._connectivity_state.events.insert(
            0,
            TimelineEvent(
                timestamp=isoformat_utc(),
                event=event,
                stage=stage,
                level=level,
                summary=summary,
                details=details,
            ),
        )
        self._connectivity_state.events = self._connectivity_state.events[:80]

    def _push_history_locked(self) -> None:
        history_entry = {
            "run_id": self._state.run_id,
            "benchmark_run_id": self._state.benchmark_run_id,
            "status": self._state.status,
            "suite": self._state.suite or "all",
            "started_at": self._state.started_at,
            "finished_at": self._state.finished_at,
            "completed_records": self._state.completed_records,
            "successful_records": self._state.successful_records,
            "failed_records": self._state.failed_records,
            "latest_error": self._state.latest_error,
        }
        self._history = [history_entry] + [
            row for row in self._history if row.get("run_id") != self._state.run_id
        ]
        self._history = self._history[:20]

    def _discover_generated_files(self) -> list[str]:
        return [name for name in REPORT_FILES if (self.results_dir / name).exists()]

    def _cached_preflight(self, *, suite: str | None) -> dict[str, Any]:
        normalized_suite = None if not suite or suite == "all" else suite
        cache_key = normalized_suite or "all"
        cached = self._preflight_cache.get(cache_key)
        if cached:
            return {key: value for key, value in cached.items() if key != "_signature"}
        return self.preflight(suite=normalized_suite)

    def _preflight_signature(self) -> tuple[Any, ...]:
        config_mtime = self.config_path.stat().st_mtime_ns if self.config_path.exists() else 0
        tests_mtime = 0
        if self.tests_dir.exists():
            for path in self.tests_dir.rglob("*"):
                if path.is_file():
                    tests_mtime = max(tests_mtime, path.stat().st_mtime_ns)
        return (
            config_mtime,
            tests_mtime,
            os.getenv("OPENAI_API_KEY"),
            os.getenv("MISTRAL_BASE_URL"),
            os.getenv("QWEN_BASE_URL"),
        )

    def _recommended_suite(self, available_suites: list[str]) -> str | None:
        configured = os.getenv("BENCHMARK_SUITE")
        if configured and configured in available_suites:
            return configured
        if "quick_compare" in available_suites:
            return "quick_compare"
        return available_suites[0] if available_suites else None

    def _persist_locked(self) -> None:
        json_dump_to_path(self.state_path, asdict(self._state), pretty=True)
        json_dump_to_path(self.history_path, self._history, pretty=True)
        json_dump_to_path(self.connectivity_state_path, asdict(self._connectivity_state), pretty=True)

    def _load_state(self) -> RunState:
        if not self.state_path.exists():
            return RunState(
                config_path=str(self.config_path),
                tests_dir=str(self.tests_dir),
                results_dir=str(self.results_dir),
            )
        try:
            raw_state = orjson.loads(self.state_path.read_bytes())
            events = [
                TimelineEvent(**event_payload)
                for event_payload in raw_state.get("events", [])
            ]
            return RunState(**(raw_state | {"events": events}))
        except Exception as exc:
            LOGGER.warning("Could not restore dashboard run state: %s", exc)
            return RunState(
                config_path=str(self.config_path),
                tests_dir=str(self.tests_dir),
                results_dir=str(self.results_dir),
            )

    def _load_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        try:
            raw_history = orjson.loads(self.history_path.read_bytes())
            return raw_history if isinstance(raw_history, list) else []
        except Exception as exc:
            LOGGER.warning("Could not restore dashboard run history: %s", exc)
            return []

    def _load_connectivity_state(self) -> ConnectivityState:
        if not self.connectivity_state_path.exists():
            return ConnectivityState()
        try:
            raw_state = orjson.loads(self.connectivity_state_path.read_bytes())
            events = [
                TimelineEvent(**event_payload)
                for event_payload in raw_state.get("events", [])
            ]
            results = [
                ConnectivityResult(**result_payload)
                for result_payload in raw_state.get("results", [])
            ]
            restored = ConnectivityState(**(raw_state | {"events": events, "results": results}))
            if restored.status == "running":
                restored.status = "interrupted"
                restored.summary = "Vorheriger Erreichbarkeits-Check wurde unterbrochen."
            return restored
        except Exception as exc:
            LOGGER.warning("Could not restore connectivity check state: %s", exc)
            return ConnectivityState()

    def _probe_model_endpoint(self, *, model_config: Any, default_timeout_seconds: float) -> ConnectivityResult:
        """
        Probe a provider with a cheap `/models` request so operators can validate reachability without a real benchmark.

        Example interpretation:
        - `reachable`: endpoint answered and the configured model was listed
        - `reachable_model_missing`: endpoint answered but the configured model name was not returned
        - `missing_api_key` / `timeout` / `network_error`: benchmark would currently fail for this model
        """

        headers = {"Accept": "application/json"}
        if model_config.api_key_env:
            api_key = os.getenv(model_config.api_key_env)
            if not api_key:
                return ConnectivityResult(
                    model_id=model_config.id,
                    model_label=model_config.label,
                    provider=model_config.provider,
                    endpoint=model_config.base_url,
                    model_name=model_config.model_name,
                    ok=False,
                    status="missing_api_key",
                    message=f"API-Key-Variable {model_config.api_key_env} ist nicht gesetzt.",
                )
            headers["Authorization"] = f"Bearer {api_key}"

        timeout_seconds = min(model_config.effective_timeout(default_timeout_seconds), 15.0)
        url = f"{model_config.base_url}/models"
        start = perf_counter()
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout_seconds)) as client:
                response = client.get(url, headers=headers)
            duration_ms = round((perf_counter() - start) * 1000, 2)
        except httpx.TimeoutException:
            return ConnectivityResult(
                model_id=model_config.id,
                model_label=model_config.label,
                provider=model_config.provider,
                endpoint=model_config.base_url,
                model_name=model_config.model_name,
                ok=False,
                status="timeout",
                message=f"Endpoint antwortet nicht innerhalb von {timeout_seconds} Sekunden.",
            )
        except httpx.TransportError as exc:
            return ConnectivityResult(
                model_id=model_config.id,
                model_label=model_config.label,
                provider=model_config.provider,
                endpoint=model_config.base_url,
                model_name=model_config.model_name,
                ok=False,
                status="network_error",
                message=f"Netzwerkfehler beim Zugriff auf {url}: {exc}",
            )

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        listed_models = [
            str(item.get("id"))
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id") is not None
        ] if isinstance(payload, dict) else []
        model_listed = model_config.model_name in listed_models if listed_models else None

        if response.status_code >= 400:
            message = "Endpoint ist erreichbar, hat den Check aber abgelehnt."
            if isinstance(payload, dict):
                error_message = payload.get("error", {}).get("message") if isinstance(payload.get("error"), dict) else payload.get("message")
                if isinstance(error_message, str) and error_message:
                    message = error_message
            return ConnectivityResult(
                model_id=model_config.id,
                model_label=model_config.label,
                provider=model_config.provider,
                endpoint=model_config.base_url,
                model_name=model_config.model_name,
                ok=False,
                status="http_error",
                message=f"HTTP {response.status_code}: {message}",
                http_status=response.status_code,
                duration_ms=duration_ms,
                model_listed=model_listed,
                listed_models_preview=listed_models[:8],
            )

        if model_listed is False:
            return ConnectivityResult(
                model_id=model_config.id,
                model_label=model_config.label,
                provider=model_config.provider,
                endpoint=model_config.base_url,
                model_name=model_config.model_name,
                ok=False,
                status="reachable_model_missing",
                message="Endpoint antwortet, aber der konfigurierte Modellname ist in /models nicht sichtbar.",
                http_status=response.status_code,
                duration_ms=duration_ms,
                model_listed=False,
                listed_models_preview=listed_models[:8],
            )

        return ConnectivityResult(
            model_id=model_config.id,
            model_label=model_config.label,
            provider=model_config.provider,
            endpoint=model_config.base_url,
            model_name=model_config.model_name,
            ok=True,
            status="reachable",
            message="Endpoint und Modell sind erreichbar.",
            http_status=response.status_code,
            duration_ms=duration_ms,
            model_listed=model_listed,
            listed_models_preview=listed_models[:8],
        )
