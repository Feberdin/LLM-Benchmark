"""
Purpose: Manage interactive live compare runs, their current UI state, presets and persistent history snapshots.
Input/Output: Starts background compare jobs, exposes pollable state for the dashboard and stores finished runs as JSON.
Important invariants: Fair serial execution is the default for CPU-bound systems, one failed model must never erase the other columns and presets stay loadable without extra services.
How to debug: Inspect `live_compare/current_state.json`, `live_compare/history.json`, `live_compare/runs/<run_id>.json` and the loaded preset file path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock, Thread
from typing import Any
import logging
import os

import orjson
import yaml

from llm_benchmark.config.loader import load_config
from llm_benchmark.domain.live_compare import (
    CompareExecutionMode,
    CompareRunStatus,
    LiveCompareEvent,
    LiveCompareModelResult,
    LiveComparePreset,
    LiveCompareRequest,
    LiveCompareRunRecord,
    LiveCompareSummary,
)
from llm_benchmark.runner.live_compare import DEFAULT_FAIR_ORDER, execute_live_compare_sync
from llm_benchmark.utils import ensure_directory, isoformat_utc, json_dump_to_path

LOGGER = logging.getLogger(__name__)
LIVE_COMPARE_DIR_NAME = "live_compare"


@dataclass(slots=True)
class LiveCompareState:
    """Current live compare state that the UI polls while the requests are running."""

    run_id: str | None = None
    status: CompareRunStatus = "idle"
    created_at: str | None = None
    finished_at: str | None = None
    question: str | None = None
    system_prompt: str | None = None
    mode: str = "chat"
    execution_mode: CompareExecutionMode = "serial"
    execution_order: list[str] = field(default_factory=list)
    max_tokens: int = 600
    temperature: float = 0.0
    top_p: float = 1.0
    manual_note: str | None = None
    selected_models: list[str] = field(default_factory=list)
    results: list[LiveCompareModelResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    latest_error: str | None = None
    events: list[LiveCompareEvent] = field(default_factory=list)


class LiveCompareManager:
    """
    Coordinate the dashboard live compare view.

    Why this exists:
    The operator should be able to compare real responses side-by-side without leaving the dashboard, while the
    server keeps a durable history that can be revisited later and respects CPU-bound fairness by default.
    """

    def __init__(self, *, config_path: Path, results_dir: Path) -> None:
        self.config_path = config_path
        self.results_dir = ensure_directory(results_dir)
        self.live_compare_dir = ensure_directory(self.results_dir / LIVE_COMPARE_DIR_NAME)
        self.runs_dir = ensure_directory(self.live_compare_dir / "runs")
        self.state_path = self.live_compare_dir / "current_state.json"
        self.history_path = self.live_compare_dir / "history.json"
        self.presets_path = self._discover_presets_path()
        self._lock = RLock()
        self._active_thread: Thread | None = None
        self._state = self._load_state()
        self._history = self._load_history()
        self._presets = self._load_presets()
        if self._state.status in {"queued", "running"}:
            self._state.status = "interrupted"
            self._state.latest_error = "Vorheriger Live-Compare-Lauf wurde nicht sauber beendet."
            self._append_event_locked(
                event="live_compare_interrupted",
                level="warning",
                summary="Vorheriger Live-Compare-Lauf wurde beim Neustart oder Stoppen unterbrochen.",
                details={},
            )
            self._persist_locked()

    def page_payload(self) -> dict[str, Any]:
        """Return current state, model choices, presets and history for the live compare HTML page."""

        with self._lock:
            state = self._state_to_dict(self._state)
            history = list(self._history[:12])
        benchmark_config = load_config(self.config_path)
        available_models = [
            {
                "model_id": model.id,
                "model_label": model.label,
                "provider": model.provider,
                "model_name": model.model_name,
                "enabled": model.enabled,
            }
            for model in benchmark_config.enabled_models()
        ]
        default_models = self._default_model_ids([row["model_id"] for row in available_models])
        return {
            "state": state,
            "history": history,
            "available_models": available_models,
            "default_models": default_models,
            "modes": [
                {"value": "chat", "label": "Chat / Freitext"},
                {"value": "json", "label": "JSON / Structured Output"},
                {"value": "technical", "label": "Technical / Coding"},
                {"value": "summarization", "label": "Summarization"},
            ],
            "execution_modes": [
                {
                    "value": "serial",
                    "label": "Fair Compare (seriell)",
                    "description": "Standard fuer CPU-lastige lokale Server. Modelle laufen nacheinander fuer fairere Latenzen.",
                },
                {
                    "value": "parallel",
                    "label": "Parallel Compare",
                    "description": "Kann lokale Laufzeiten verfälschen, weil mehrere Modelle gleichzeitig um CPU-Zeit konkurrieren.",
                },
            ],
            "presets": [preset.model_dump(mode="json") for preset in self._presets],
        }

    def current_payload(self) -> dict[str, Any]:
        with self._lock:
            return self._state_to_dict(self._state)

    def history_payload(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history[:40])

    def run_payload(self, run_id: str) -> dict[str, Any]:
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(run_id)
        return orjson.loads(path.read_bytes())

    def start_compare(self, raw_request: dict[str, Any]) -> dict[str, Any]:
        """Start a background live compare run and return the queued state."""

        benchmark_config = load_config(self.config_path)
        enabled_model_ids = [model.id for model in benchmark_config.enabled_models()]
        request = LiveCompareRequest.model_validate(
            raw_request
            | {
                "models": raw_request.get("models") or self._default_model_ids(enabled_model_ids),
                "execution_mode": raw_request.get("execution_mode") or "serial",
            }
        )
        selected_models = self._validate_requested_models(enabled_model_ids, request.models)
        execution_order = self._ordered_selection(selected_models)

        with self._lock:
            if self._active_thread and self._active_thread.is_alive():
                raise RuntimeError("Es laeuft bereits ein Live Compare. Bitte warte, bis dieser Vergleich fertig ist.")

            enabled_models_by_id = {model.id: model for model in benchmark_config.enabled_models()}
            placeholder_results = [
                LiveCompareModelResult(
                    model_id=enabled_models_by_id[model_id].id,
                    model_label=enabled_models_by_id[model_id].label,
                    provider=enabled_models_by_id[model_id].provider,
                    model_name=enabled_models_by_id[model_id].model_name,
                    endpoint=enabled_models_by_id[model_id].base_url,
                    status="waiting",
                )
                for model_id in execution_order
            ]
            self._state = LiveCompareState(
                run_id=f"livecmp_{os.urandom(6).hex()}",
                status="queued",
                created_at=isoformat_utc(),
                question=request.question,
                system_prompt=request.system_prompt,
                mode=request.mode,
                execution_mode=request.execution_mode,
                execution_order=execution_order,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                manual_note=request.manual_note,
                selected_models=execution_order,
                results=placeholder_results,
                summary={
                    "execution_mode": request.execution_mode,
                    "execution_order": execution_order,
                    "advisory": self._execution_mode_advisory(request.execution_mode),
                },
                events=[
                    LiveCompareEvent(
                        timestamp=isoformat_utc(),
                        event="live_compare_queued",
                        level="info",
                        summary="Live Compare wurde in die Hintergrundausfuehrung uebernommen.",
                        details={"models": execution_order, "execution_mode": request.execution_mode},
                    )
                ],
            )
            self._persist_locked()
            run_id = self._state.run_id or "livecmp_unknown"
            self._active_thread = Thread(
                target=self._run_in_background,
                kwargs={"run_id": run_id, "request": request},
                daemon=True,
                name=f"live-compare-{run_id[:8]}",
            )
            self._active_thread.start()
            return self._state_to_dict(self._state)

    def _run_in_background(self, *, run_id: str, request: LiveCompareRequest) -> None:
        benchmark_config = load_config(self.config_path)
        with self._lock:
            if self._state.run_id != run_id:
                return
            self._state.status = "running"
            self._append_event_locked(
                event="live_compare_started",
                level="info",
                summary=(
                    "Live Compare gestartet. Modelle werden nacheinander ausgefuehrt."
                    if request.execution_mode == "serial"
                    else "Live Compare gestartet. Modelle laufen parallel."
                ),
                details={"mode": request.mode, "execution_mode": request.execution_mode},
            )
            self._persist_locked()

        try:
            record = execute_live_compare_sync(
                config=benchmark_config,
                request=request,
                run_id=run_id,
                progress_callback=lambda payload: self._handle_progress(run_id=run_id, payload=payload),
            )
            with self._lock:
                if self._state.run_id != run_id:
                    return
                self._state.status = record.status
                self._state.finished_at = record.finished_at
                self._state.execution_mode = record.execution_mode
                self._state.execution_order = record.execution_order
                self._state.manual_note = record.manual_note
                self._state.summary = record.summary.model_dump(mode="json")
                self._state.results = record.results
                self._state.latest_error = record.latest_error
                self._append_event_locked(
                    event="live_compare_persisted",
                    level="success" if record.status == "succeeded" else "warning",
                    summary="Live Compare abgeschlossen und im Verlauf gespeichert.",
                    details={"run_id": record.run_id, "status": record.status},
                )
                self._write_run_record(record)
                self._push_history_locked(record)
                self._persist_locked()
        except Exception as exc:
            LOGGER.exception("Live compare run failed.")
            with self._lock:
                if self._state.run_id != run_id:
                    return
                self._state.status = "failed"
                self._state.finished_at = isoformat_utc()
                self._state.latest_error = str(exc)
                self._append_event_locked(
                    event="live_compare_failed",
                    level="error",
                    summary=f"Live Compare fehlgeschlagen: {exc}",
                    details={"error": str(exc)},
                )
                self._persist_locked()

    def _handle_progress(self, *, run_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            if self._state.run_id != run_id:
                return

            event = str(payload.get("event", "progress"))
            if event == "compare_started":
                self._state.execution_mode = payload.get("execution_mode", self._state.execution_mode)
                self._state.execution_order = list(payload.get("execution_order") or self._state.execution_order)
                self._state.summary = {
                    **self._state.summary,
                    "execution_mode": self._state.execution_mode,
                    "execution_order": self._state.execution_order,
                    "advisory": self._execution_mode_advisory(self._state.execution_mode),
                }
            elif event == "compare_model_started":
                result = self._find_result_locked(str(payload.get("model_id")))
                if result is not None:
                    result.status = "running"
                    result.started_at = payload.get("execution_start_at")
                    result.execution_start_at = payload.get("execution_start_at")
                    result.endpoint = payload.get("endpoint")
                    result.queue_wait_ms = payload.get("queue_wait_ms")
            elif event == "compare_model_finished":
                result_payload = payload.get("result") or {}
                model_id = str(result_payload.get("model_id", ""))
                result = self._find_result_locked(model_id)
                if result is not None:
                    updated = LiveCompareModelResult.model_validate(result_payload)
                    result_index = self._state.results.index(result)
                    self._state.results[result_index] = updated
                    if updated.error_message:
                        self._state.latest_error = updated.error_message
            elif event == "compare_finished":
                self._state.summary = payload.get("summary") or self._state.summary

            summary = self._summarize_event(payload)
            self._append_event_locked(
                event=event,
                level=summary["level"],
                summary=summary["summary"],
                details=payload,
            )
            self._persist_locked()

    def _find_result_locked(self, model_id: str) -> LiveCompareModelResult | None:
        for result in self._state.results:
            if result.model_id == model_id:
                return result
        return None

    def _summarize_event(self, payload: dict[str, Any]) -> dict[str, str]:
        event = str(payload.get("event", "progress"))
        if event == "compare_started":
            mode = payload.get("execution_mode", "serial")
            return {
                "level": "info",
                "summary": (
                    "Fair Compare wurde vorbereitet."
                    if mode == "serial"
                    else "Parallel Compare wurde vorbereitet."
                ),
            }
        if event == "compare_model_started":
            return {
                "level": "info",
                "summary": (
                    f"{payload.get('model_label', 'Modell')} startet jetzt nach "
                    f"{payload.get('queue_wait_ms', 0)} ms Wartezeit."
                ),
            }
        if event == "compare_model_finished":
            result = payload.get("result") or {}
            if result.get("success"):
                return {
                    "level": "success",
                    "summary": (
                        f"{result.get('model_label', 'Modell')} beendet: "
                        f"{result.get('isolated_duration_ms') or result.get('duration_ms') or 'n/a'} ms isolierte Dauer."
                    ),
                }
            return {
                "level": "error",
                "summary": f"{result.get('model_label', 'Modell')} schlug fehl ({result.get('error_type') or 'error'}).",
            }
        if event == "compare_finished":
            summary = payload.get("summary") or {}
            return {
                "level": "success" if summary.get("all_successful") else "warning",
                "summary": (
                    "Alle Modellantworten liegen vor."
                    if summary.get("all_successful")
                    else "Live Compare abgeschlossen, aber mindestens ein Modell hatte Probleme."
                ),
            }
        return {"level": "info", "summary": "Live Compare Fortschritt aktualisiert."}

    def _state_to_dict(self, state: LiveCompareState) -> dict[str, Any]:
        payload = {
            "run_id": state.run_id,
            "status": state.status,
            "created_at": state.created_at,
            "finished_at": state.finished_at,
            "question": state.question,
            "system_prompt": state.system_prompt,
            "mode": state.mode,
            "execution_mode": state.execution_mode,
            "execution_order": list(state.execution_order),
            "max_tokens": state.max_tokens,
            "temperature": state.temperature,
            "top_p": state.top_p,
            "manual_note": state.manual_note,
            "selected_models": list(state.selected_models),
            "results": [result.model_dump(mode="json") for result in state.results],
            "summary": state.summary or {
                "execution_mode": state.execution_mode,
                "execution_order": list(state.execution_order),
                "advisory": self._execution_mode_advisory(state.execution_mode),
            },
            "latest_error": state.latest_error,
            "events": [event.model_dump(mode="json") for event in state.events],
        }
        now = datetime.now(tz=UTC)
        for result in payload.get("results", []):
            if result.get("status") == "running" and result.get("execution_start_at"):
                started_at = _parse_iso(result["execution_start_at"])
                if started_at is not None:
                    elapsed_ms = max((now - started_at).total_seconds() * 1000, 0)
                    result["elapsed_ms"] = round(elapsed_ms, 2)
                    result["elapsed_human"] = _humanize_duration_ms(elapsed_ms)
                    result["isolated_duration_live_ms"] = round(elapsed_ms, 2)
            elif result.get("status") == "waiting":
                result["elapsed_ms"] = None
                result["elapsed_human"] = None
        if payload.get("status") == "running" and payload.get("created_at"):
            started_at = _parse_iso(payload["created_at"])
            if started_at is not None:
                payload["elapsed_ms"] = round(max((now - started_at).total_seconds() * 1000, 0), 2)
        return payload

    def _write_run_record(self, record: LiveCompareRunRecord) -> None:
        path = self.runs_dir / f"{record.run_id}.json"
        json_dump_to_path(path, record.model_dump(mode="json"), pretty=True)

    def _push_history_locked(self, record: LiveCompareRunRecord) -> None:
        history_entry = {
            "run_id": record.run_id,
            "status": record.status,
            "created_at": record.created_at,
            "finished_at": record.finished_at,
            "mode": record.mode,
            "execution_mode": record.execution_mode,
            "execution_order": record.execution_order,
            "question_excerpt": (record.question[:117] + "...") if len(record.question) > 120 else record.question,
            "manual_note": record.manual_note,
            "selected_models": record.selected_models,
            "successful_model_count": record.summary.successful_model_count,
            "failed_model_count": record.summary.failed_model_count,
            "fastest_model_label": record.summary.fastest_model_label,
            "per_model_durations": [
                {
                    "model_id": result.model_id,
                    "model_label": result.model_label,
                    "isolated_duration_ms": result.isolated_duration_ms,
                    "total_elapsed_since_run_start_ms": result.total_elapsed_since_run_start_ms,
                }
                for result in record.results
            ],
        }
        self._history = [history_entry] + [row for row in self._history if row.get("run_id") != record.run_id]
        self._history = self._history[:40]

    def _append_event_locked(self, *, event: str, level: str, summary: str, details: dict[str, Any]) -> None:
        self._state.events.insert(
            0,
            LiveCompareEvent(
                timestamp=isoformat_utc(),
                event=event,
                level=level,
                summary=summary,
                details=details,
            ),
        )
        self._state.events = self._state.events[:80]

    def _validate_requested_models(self, enabled_model_ids: list[str], requested_model_ids: list[str]) -> list[str]:
        if not requested_model_ids:
            requested_model_ids = self._default_model_ids(enabled_model_ids)
        missing = [model_id for model_id in requested_model_ids if model_id not in enabled_model_ids]
        if missing:
            raise ValueError("Unbekannte oder deaktivierte Modelle: " + ", ".join(sorted(missing)))
        if not 2 <= len(requested_model_ids) <= 4:
            raise ValueError("Live Compare unterstuetzt aktuell 2 bis 4 Modelle gleichzeitig.")
        return requested_model_ids

    @staticmethod
    def _default_model_ids(enabled_model_ids: list[str]) -> list[str]:
        selected = [model_id for model_id in DEFAULT_FAIR_ORDER if model_id in enabled_model_ids]
        if len(selected) >= 2:
            return selected[:3]
        return enabled_model_ids[:3]

    def _ordered_selection(self, model_ids: list[str]) -> list[str]:
        fair_index = {model_id: index for index, model_id in enumerate(DEFAULT_FAIR_ORDER)}
        return sorted(model_ids, key=lambda model_id: (fair_index.get(model_id, len(DEFAULT_FAIR_ORDER)), model_id))

    @staticmethod
    def _execution_mode_advisory(execution_mode: CompareExecutionMode) -> str:
        if execution_mode == "serial":
            return (
                "Im seriellen Modus werden Modelle nacheinander ausgefuehrt, damit Laufzeiten auf CPU-lastiger "
                "lokaler Hardware fair vergleichbar bleiben."
            )
        return (
            "Parallelmodus kann lokale Laufzeiten verfälschen, weil Modelle gleichzeitig um CPU-Ressourcen konkurrieren."
        )

    def _persist_locked(self) -> None:
        json_dump_to_path(self.state_path, self._state_to_dict(self._state), pretty=True)
        json_dump_to_path(self.history_path, self._history, pretty=True)

    def _load_state(self) -> LiveCompareState:
        if not self.state_path.exists():
            return LiveCompareState()
        try:
            raw_state = orjson.loads(self.state_path.read_bytes())
            results = [LiveCompareModelResult.model_validate(item) for item in raw_state.get("results", [])]
            events = [LiveCompareEvent.model_validate(item) for item in raw_state.get("events", [])]
            return LiveCompareState(**(raw_state | {"results": results, "events": events}))
        except Exception as exc:
            LOGGER.warning("Could not restore live compare state: %s", exc)
            return LiveCompareState()

    def _load_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        try:
            raw_history = orjson.loads(self.history_path.read_bytes())
            return raw_history if isinstance(raw_history, list) else []
        except Exception as exc:
            LOGGER.warning("Could not restore live compare history: %s", exc)
            return []

    def _load_presets(self) -> list[LiveComparePreset]:
        path = self.presets_path
        if path is None or not path.exists():
            LOGGER.warning("Live compare presets file not found.")
            return []
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            LOGGER.warning("Could not load live compare presets: %s", exc)
            return []
        preset_rows = payload.get("presets", []) if isinstance(payload, dict) else []
        presets: list[LiveComparePreset] = []
        for row in preset_rows:
            try:
                presets.append(LiveComparePreset.model_validate(row))
            except Exception as exc:  # pragma: no cover - guard rail for malformed presets
                LOGGER.warning("Ignoring invalid live compare preset: %s", exc)
        return presets

    @staticmethod
    def _discover_presets_path() -> Path | None:
        candidates = [
            Path("/app/fixtures/live_compare/presets.yaml"),
            Path(__file__).resolve().parents[3] / "fixtures" / "live_compare" / "presets.yaml",
            Path.cwd() / "fixtures" / "live_compare" / "presets.yaml",
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _humanize_duration_ms(duration_ms: float | None) -> str | None:
    if duration_ms is None:
        return None
    seconds = duration_ms / 1000.0
    if seconds < 1:
        return f"{duration_ms:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"{minutes}m {remainder:.1f}s"
