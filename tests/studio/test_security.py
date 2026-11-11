"""
Security tests for PruneQuest Studio: host header validation,
CSRF origin protection, path traversal defenses, and token masking.
"""

from fastapi.testclient import TestClient
from prunequest.studio.server import create_app
from prunequest.studio.security.middleware import validate_safe_path, sanitize_sensitive_data
import pytest


def test_untrusted_host_header_blocked():
    app = create_app(allowed_hosts={"127.0.0.1", "localhost"})
    client = TestClient(app)

    # Request with hostile host header
    res = client.get("/api/v1/health", headers={"Host": "malicious-site.com"})
    assert res.status_code == 403
    assert "Forbidden" in res.json()["error"]

    # Request with trusted host header
    res_valid = client.get("/api/v1/health", headers={"Host": "127.0.0.1:8765"})
    assert res_valid.status_code == 200


def test_cross_origin_mutating_request_blocked():
    app = create_app(allowed_hosts={"127.0.0.1", "localhost"})
    client = TestClient(app)

    # Cross-origin POST request from untrusted origin
    res = client.post(
        "/api/v1/models/inspect",
        json={"model_id": "test"},
        headers={"Origin": "https://attacker-evil-origin.com"},
    )
    assert res.status_code == 403
    assert "Forbidden" in res.json()["error"]


def test_safe_path_validation():
    # Valid relative/absolute path within base
    valid = validate_safe_path("runs/my_run", base_dir="runs")
    assert "runs/my_run" in valid

    # Traversal escaping base directory must raise ValueError
    with pytest.raises(ValueError):
        validate_safe_path("runs/../../etc/passwd", base_dir="runs")


def test_sensitive_data_sanitization():
    raw = {
        "model_id": "meta-llama/Llama-3.2-1B",
        "hf_token": "hf_secret_12345",
        "user_api_key": "secret_key_abc",
        "nested": {"access_token": "bearer_999", "samples": 100},
    }
    cleaned = sanitize_sensitive_data(raw)
    assert cleaned["model_id"] == "meta-llama/Llama-3.2-1B"
    assert cleaned["hf_token"] == "[REDACTED]"
    assert cleaned["user_api_key"] == "[REDACTED]"
    assert cleaned["nested"]["access_token"] == "[REDACTED]"
    assert cleaned["nested"]["samples"] == 100
