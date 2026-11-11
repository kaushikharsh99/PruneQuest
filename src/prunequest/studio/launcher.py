"""
Studio launcher and server lifecycle management for PruneQuest Studio.
Handles dependency checking, port discovery, readiness polling,
and automatic browser opening.
"""

import os
import socket
import sys
import time
import webbrowser
from typing import Optional
from rich.console import Console
from rich.panel import Panel

console = Console()


def check_studio_dependencies() -> bool:
    """Verifies that web-server dependencies (fastapi, uvicorn, sse-starlette) are present."""
    missing = []
    for pkg in ("fastapi", "uvicorn", "sse_starlette"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return len(missing) == 0


def find_available_port(start_port: int = 8765, max_attempts: int = 50) -> int:
    """Finds an available TCP port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts}")


def wait_for_server_ready(host: str, port: int, timeout: float = 10.0) -> bool:
    """Polls TCP connection until server is listening or timeout expires."""
    start_time = time.time()
    # Normalize host for local socket connection
    check_host = "127.0.0.1" if host in ("0.0.0.0", "localhost") else host
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((check_host, port), timeout=0.5):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(0.1)
    return False


def launch_studio(
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    no_browser: bool = False,
    reload: bool = False,
    static_dir: Optional[str] = None,
):
    """
    Main entry point for 'prunequest studio' command.
    """
    # 1. Dependency Check
    if not check_studio_dependencies():
        console.print(
            "[bold red]Studio dependencies not found.[/bold red]\n\n"
            "PruneQuest Studio requires FastAPI and Uvicorn. Install them with:\n\n"
            "    [bold green]pip install \"prunequest[studio]\"[/bold green]\n"
            "or:\n"
            "    [bold green]uv tool install \"prunequest[studio]\"[/bold green]\n"
        )
        sys.exit(1)

    import uvicorn
    from prunequest.studio.server import create_app, DEFAULT_STATIC_DIR

    # 2. Port resolution
    target_port = port or find_available_port(start_port=8765)
    url = f"http://{host}:{target_port}"

    # 3. Static assets inspection
    resolved_static = static_dir or DEFAULT_STATIC_DIR
    has_ui = os.path.exists(os.path.join(resolved_static, "index.html"))

    panel_text = (
        f"[bold #2d5a3f]PruneQuest Studio[/bold #2d5a3f]\n\n"
        f"Local URL: [bold underline]{url}[/bold underline]\n"
        f"Host: [dim]{host}[/dim] | Port: [dim]{target_port}[/dim]\n"
        f"Frontend UI: [green]{'Ready (Compiled)' if has_ui else 'API Mode (Fallback)'}[/green]\n\n"
        f"[dim]Press Ctrl+C to gracefully stop the studio server.[/dim]"
    )
    console.print(Panel(panel_text, border_style="#4a7c59"))

    # 4. Asynchronous browser opening in background thread
    if not no_browser:
        import threading

        def _open_browser_when_ready():
            if wait_for_server_ready(host, target_port):
                time.sleep(0.2)
                try:
                    webbrowser.open(url)
                except Exception:
                    pass

        threading.Thread(target=_open_browser_when_ready, daemon=True).start()

    # 5. Start Uvicorn Server
    app = create_app(static_dir=resolved_static, allowed_hosts={host, "127.0.0.1", "localhost", "[::1]"})
    config = uvicorn.Config(
        app=app,
        host=host,
        port=target_port,
        log_level="info",
        reload=reload,
    )
    server = uvicorn.Server(config)
    try:
        server.run()
    except (KeyboardInterrupt, SystemExit):
        console.print("\n[yellow]PruneQuest Studio server stopped.[/yellow]")
