"""
Tests for Studio Launcher and server bootstrapping.
"""

from fastapi.testclient import TestClient
from prunequest.studio.launcher import check_studio_dependencies, find_available_port
from prunequest.studio.server import create_app
import tempfile
import os


def test_studio_dependencies_check():
    assert check_studio_dependencies() is True


def test_find_available_port():
    port = find_available_port(start_port=8890, max_attempts=10)
    assert 8890 <= port <= 8900


def test_spa_static_and_fallback_serving():
    # Test fallback when directory is empty
    with tempfile.TemporaryDirectory() as empty_dir:
        app_empty = create_app(static_dir=empty_dir)
        client_empty = TestClient(app_empty)

        res_fallback = client_empty.get("/")
        assert res_fallback.status_code == 200
        assert "PruneQuest Studio" in res_fallback.text
        assert "Backend Service Ready" in res_fallback.text

    # Test serving actual compiled index.html
    with tempfile.TemporaryDirectory() as mock_static:
        index_path = os.path.join(mock_static, "index.html")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("<!doctype html><html><body><div id='root'>Studio Loaded</div></body></html>")

        app_spa = create_app(static_dir=mock_static)
        client_spa = TestClient(app_spa)

        res_spa = client_spa.get("/")
        assert res_spa.status_code == 200
        assert "Studio Loaded" in res_spa.text

        # Test non-API route returns SPA index.html for client routing
        res_route = client_spa.get("/new-optimization")
        assert res_route.status_code == 200
        assert "Studio Loaded" in res_route.text
