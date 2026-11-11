"""
Security middleware and controls for PruneQuest Studio.
Enforces loopback origin protection, host header validation,
path traversal checks, and token sanitization.
"""

import os
from typing import Set, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Validates Host and Origin headers to guard against DNS rebinding
    and Cross-Site Request Forgery (CSRF).
    """

    def __init__(self, app, allowed_hosts: Optional[Set[str]] = None):
        super().__init__(app)
        self.allowed_hosts = allowed_hosts or {
            "127.0.0.1",
            "localhost",
            "[::1]",
            "0.0.0.0",
            "testserver",
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        # 1. Host Header Validation
        host = request.headers.get("host", "").split(":")[0].strip()
        if host and host not in self.allowed_hosts:
            return JSONResponse(
                status_code=403,
                content={"error": f"Forbidden: Untrusted Host header '{host}'"},
            )

        # 2. Origin/Referer Protection on Mutating Requests
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("origin")
            if origin:
                origin_clean = origin.split("://")[-1].split(":")[0].strip()
                if origin_clean not in self.allowed_hosts:
                    return JSONResponse(
                        status_code=403,
                        content={"error": f"Forbidden: Cross-origin mutating request from '{origin}' blocked"},
                    )

        response = await call_next(request)

        # 3. Add standard secure local headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        return response


def sanitize_sensitive_data(obj: any) -> any:
    """
    Recursively strips or masks sensitive environment secrets like HF_TOKEN.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if any(secret in k.lower() for secret in ("token", "secret", "password", "key")):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = sanitize_sensitive_data(v)
        return cleaned
    elif isinstance(obj, list):
        return [sanitize_sensitive_data(x) for x in obj]
    return obj


def validate_safe_path(path: str, base_dir: Optional[str] = None) -> str:
    """
    Validates that a path does not perform unsafe directory traversal.
    Returns normalized absolute path.
    """
    normalized = os.path.abspath(os.path.expanduser(path))
    if base_dir:
        base_abs = os.path.abspath(base_dir)
        if not normalized.startswith(base_abs):
            raise ValueError(f"Path '{path}' escapes authorized base directory '{base_dir}'")
    return normalized
