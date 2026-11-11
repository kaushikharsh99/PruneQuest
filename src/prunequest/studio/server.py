"""
FastAPI application server for PruneQuest Studio.
Mounts API routes, security middleware, and compiled frontend static assets
with SPA fallback routing.
"""

import os
from typing import Optional, Set
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from prunequest import __version__
from prunequest.studio.api import api_router
from prunequest.studio.security import SecurityHeadersMiddleware

DEFAULT_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

FALLBACK_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PruneQuest Studio</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: #fbfcf9;
      color: #1e2b20;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
      padding: 20px;
    }}
    .card {{
      background: white;
      border: 1px solid #e2e8df;
      border-radius: 12px;
      padding: 32px;
      max-width: 540px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.03);
      text-align: center;
    }}
    h1 {{ color: #2d5a3f; margin-bottom: 8px; font-size: 24px; }}
    p {{ color: #4a5d50; line-height: 1.5; font-size: 15px; }}
    .badge {{
      display: inline-block;
      background: #e8f0e8;
      color: #2d5a3f;
      padding: 4px 12px;
      border-radius: 999px;
      font-weight: 600;
      font-size: 12px;
      margin-bottom: 16px;
    }}
    code {{
      background: #f4f6f0;
      padding: 3px 6px;
      border-radius: 4px;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="badge">PruneQuest Studio v{__version__}</div>
    <h1>Backend Service Ready</h1>
    <p>The PruneQuest Studio API backend is running successfully.</p>
    <p>To view the full React web dashboard, ensure frontend assets are compiled into <code>prunequest/studio/static/</code> or run the Vite dev server.</p>
    <p>Explore the interactive API at <a href="/docs" style="color: #2d5a3f; font-weight: 600;">/docs</a></p>
  </div>
</body>
</html>
"""


def create_app(
    static_dir: Optional[str] = None,
    allowed_hosts: Optional[Set[str]] = None,
) -> FastAPI:
    """
    Creates and configures the FastAPI application for PruneQuest Studio.
    """
    app = FastAPI(
        title="PruneQuest Studio",
        version=__version__,
        description="Local-first Web Interface for Autonomous LLM Pruning",
        docs_url="/docs",
        redoc_url=None,
    )

    # 1. Register Security Middleware
    app.add_middleware(SecurityHeadersMiddleware, allowed_hosts=allowed_hosts)

    # 2. Register API Routes
    app.include_router(api_router)

    # 3. Mount Static Assets & SPA Routing
    resolved_static = static_dir or DEFAULT_STATIC_DIR
    index_file = os.path.join(resolved_static, "index.html")

    if os.path.exists(resolved_static) and os.path.isdir(resolved_static):
        # Mount assets folder if it exists
        assets_dir = os.path.join(resolved_static, "assets")
        if os.path.exists(assets_dir) and os.path.isdir(assets_dir):
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    # SPA Fallback Route: any non-API path serves index.html or fallback
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(request: Request, full_path: str):
        # Allow /docs and /openapi.json to pass through
        if full_path.startswith(("docs", "openapi.json")):
            return None

        # Check if requesting a specific static file (e.g. favicon, logo)
        target_file = os.path.join(resolved_static, full_path)
        if full_path and os.path.isfile(target_file):
            return FileResponse(target_file)

        # Serve index.html if it exists
        if os.path.exists(index_file):
            return FileResponse(index_file)

        # Serve fallback HTML
        return HTMLResponse(content=FALLBACK_HTML, status_code=200)

    return app
