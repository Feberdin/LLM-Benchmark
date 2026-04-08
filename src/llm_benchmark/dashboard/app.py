"""
Purpose: FastAPI entrypoint for the integrated read-only benchmark dashboard and JSON API.
Input/Output: Serves HTML, download links and structured JSON views backed by benchmark artifact files.
Important invariants: The dashboard must stay lightweight, read-only and easy to run inside the existing container.
How to debug: Start with `/health`, then inspect dashboard service output through `/api/dashboard/summary`.
"""

from __future__ import annotations

from pathlib import Path

import jinja2
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from llm_benchmark.dashboard.service import DashboardFilters, DashboardService


def create_dashboard_app(*, results_dir: Path, tests_dir: Path | None) -> FastAPI:
    """Create the read-only dashboard application with HTML and JSON routes."""

    app = FastAPI(
        title="LLM Benchmark Dashboard",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    service = DashboardService(results_dir=results_dir, tests_dir=tests_dir)
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
        return JSONResponse(service.health())

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
        template = template_env.get_template("dashboard.html")
        return HTMLResponse(template.render(request=request, **context))

    @app.get("/api/dashboard/summary")
    async def api_summary() -> JSONResponse:
        return JSONResponse(service.api_summary())

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


def run_dashboard(*, results_dir: Path, tests_dir: Path | None, host: str, port: int, log_level: str) -> None:
    """Run the dashboard with Uvicorn in-process so the existing container can expose it directly."""

    app = create_dashboard_app(results_dir=results_dir, tests_dir=tests_dir)
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
