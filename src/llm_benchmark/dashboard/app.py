"""
Purpose: FastAPI entrypoint for the integrated benchmark dashboard, JSON API and lightweight run controls.
Input/Output: Serves HTML, download links, structured JSON views and a small set of endpoints for starting and tracking runs.
Important invariants: The dashboard stays lightweight, reuses the same execution path as the CLI and never hosts benchmark logic twice.
How to debug: Start with `/health`, then inspect `/api/dashboard/run/current` and `/api/dashboard/summary`.
"""

from __future__ import annotations

from pathlib import Path

import jinja2
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from llm_benchmark.dashboard.run_manager import DashboardRunManager
from llm_benchmark.dashboard.service import DashboardFilters, DashboardService


def create_dashboard_app(*, config_path: Path, results_dir: Path, tests_dir: Path) -> FastAPI:
    """Create the dashboard application with HTML views, JSON APIs and run controls."""

    app = FastAPI(
        title="LLM Benchmark Dashboard",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    service = DashboardService(results_dir=results_dir, tests_dir=tests_dir)
    run_manager = DashboardRunManager(config_path=config_path, tests_dir=tests_dir, results_dir=results_dir)
    package_root = Path(__file__).resolve().parent
    template_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(package_root / "templates")),
        autoescape=jinja2.select_autoescape(enabled_extensions=("html", "xml")),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    app.mount("/static", StaticFiles(directory=str(package_root / "static")), name="static")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=307)

    @app.get("/health")
    async def health() -> JSONResponse:
        payload = service.health()
        current_payload = run_manager.current_payload(include_preflight=False)
        payload["run_status"] = current_payload.get("state", {}).get("status")
        payload["connectivity_status"] = current_payload.get("connectivity", {}).get("status")
        return JSONResponse(payload)

    @app.get("/dashboard")
    async def dashboard(
        request: Request,
        view: str = Query("overview"),
        model: str | None = Query(None),
        category: str | None = Query(None),
        suite: str | None = Query(None),
        status: str = Query("all"),
        error_type: str | None = Query(None),
        search: str | None = Query(None),
    ) -> HTMLResponse:
        context = service.build_dashboard_context(
            DashboardFilters(
                view=view,
                model=model,
                category=category,
                suite=suite,
                status=status,
                error_type=error_type,
                search=search,
            )
        )
        run_payload = run_manager.current_payload(suite=suite)
        context["run_payload"] = run_payload
        context["run_state"] = run_payload["state"]
        context["run_preflight"] = run_payload["preflight"]
        context["run_history"] = run_payload["history"]
        context["connectivity"] = run_payload["connectivity"]
        template = template_env.get_template("dashboard.html")
        return HTMLResponse(template.render(request=request, **context))

    @app.get("/api/dashboard/summary")
    async def api_summary() -> JSONResponse:
        return JSONResponse(service.api_summary())

    @app.get("/api/dashboard/run/current")
    async def api_run_current(suite: str | None = Query(None)) -> JSONResponse:
        return JSONResponse(run_manager.current_payload(suite=suite, include_preflight=False))

    @app.get("/api/dashboard/run/history")
    async def api_run_history() -> JSONResponse:
        return JSONResponse(run_manager.history_payload())

    @app.post("/api/dashboard/run/start")
    async def api_run_start(suite: str | None = Query(None)) -> JSONResponse:
        try:
            payload = run_manager.start_run(suite=suite)
            return JSONResponse(payload, status_code=202)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/dashboard/connectivity/current")
    async def api_connectivity_current() -> JSONResponse:
        return JSONResponse(run_manager.connectivity_payload())

    @app.post("/api/dashboard/connectivity/check")
    async def api_connectivity_check() -> JSONResponse:
        try:
            payload = run_manager.start_connectivity_check()
            return JSONResponse(payload, status_code=202)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/dashboard/models")
    async def api_models() -> JSONResponse:
        return JSONResponse(service.api_models())

    @app.get("/api/dashboard/categories")
    async def api_categories() -> JSONResponse:
        return JSONResponse(service.api_categories())

    @app.get("/api/dashboard/failures")
    async def api_failures() -> JSONResponse:
        return JSONResponse(service.api_failures())

    @app.get("/api/dashboard/domain")
    async def api_domain() -> JSONResponse:
        return JSONResponse(service.api_domains())

    @app.get("/api/dashboard/tests")
    async def api_tests(
        model: str | None = Query(None),
        category: str | None = Query(None),
        suite: str | None = Query(None),
        status: str = Query("all"),
        error_type: str | None = Query(None),
        search: str | None = Query(None),
    ) -> JSONResponse:
        rows = service.api_tests(
            DashboardFilters(
                view="tests",
                model=model,
                category=category,
                suite=suite,
                status=status,
                error_type=error_type,
                search=search,
            )
        )
        return JSONResponse(rows)

    @app.get("/downloads/{filename}")
    async def download(filename: str) -> FileResponse:
        path = service.available_download(filename)
        if path is None:
            raise HTTPException(status_code=404, detail="Report artifact not found.")
        return FileResponse(path, filename=path.name)

    return app


def run_dashboard(
    *,
    config_path: Path,
    results_dir: Path,
    tests_dir: Path,
    host: str,
    port: int,
    log_level: str,
) -> None:
    """Run the dashboard with Uvicorn in-process so the existing container can expose it directly."""

    app = create_dashboard_app(config_path=config_path, results_dir=results_dir, tests_dir=tests_dir)
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
