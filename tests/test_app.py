"""Tests for the FastAPI application and /health endpoint."""

import os

os.environ["WIREFRAME_MOCK"] = "1"

from fastapi.testclient import TestClient

from main import app


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"

    def test_health_mock_mode(self):
        with TestClient(app) as client:
            data = client.get("/health").json()
            assert data["model_status"] == "mock"
            assert data["mock_mode"] is True

    def test_health_schema_version(self):
        with TestClient(app) as client:
            data = client.get("/health").json()
            assert data["schema_version"] >= 2

    def test_root_serves_landing_page(self):
        with TestClient(app) as client:
            resp = client.get("/")
            assert resp.status_code == 200
            assert "WIREFRAME" in resp.text


def test_global_exception_handler():
    from fastapi import APIRouter
    from main import app
    from fastapi.testclient import TestClient
    
    router = APIRouter()
    @router.get("/api/test-error")
    def trigger_error():
        raise RuntimeError("Test error for global handler")
    
    app.include_router(router)
    
    with TestClient(app, raise_server_exceptions=False) as client:
        res = client.get("/api/test-error")
        assert res.status_code == 500
        data = res.json()
        assert "error_id" in data
        assert len(data["error_id"]) > 0
