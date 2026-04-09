"""
Purpose: Manage dashboard-triggered benchmark runs, preflight checks and operator-friendly progress snapshots.
Input/Output: Accepts a suite name plus the configured paths, runs the existing benchmark pipeline in a background thread and persists a JSON status snapshot.
Important invariants: Only one dashboard run can execute at a time, and every failure must become a readable status entry instead of silently disappearing.
How to debug: Inspect `dashboard_run_state.json` and `dashboard_run_history.json` in the results directory, then compare the timeline events with the raw benchmark artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from threading import RLock, Thread
from typing import Any
import logging
import uuid

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
        self._lock = RLock()
        self._active_thread: Thread | None = None
        self._state = self._load_state()
        self._history = self._load_history()
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

    def current_payload(self, suite: str | None = None) -> dict[str, Any]:
        """Return the current state plus preflight details and recent history for the UI."""

        with self._lock:
            state = asdict(self._state)
            history = list(self._history[:12])
        return {
            "state": state,
            "preflight": self.preflight(suite=suite),
            "history": history,
        }

    def history_payload(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history[:20])

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

    def preflight(self, *, suite: str | None) -> dict[str, Any]:
        """Validate config, tests and required secrets before a dashboard-triggered run starts."""

        normalized_suite = None if not suite or suite == "all" else suite
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

        return {
            "config_path": str(self.config_path),
            "tests_dir": str(self.tests_dir),
            "results_dir": str(self.results_dir),
            "available_suites": available_suites,
            "enabled_models": enabled_models,
            "discovered_test_case_count": discovered_test_case_count,
            "selected_test_case_count": selected_test_case_count,
            "warnings": warnings,
            "errors": errors,
        }

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

    def _persist_locked(self) -> None:
        json_dump_to_path(self.state_path, asdict(self._state), pretty=True)
        json_dump_to_path(self.history_path, self._history, pretty=True)

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
