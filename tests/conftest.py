"""Shared fixtures for WIREFRAME test suite."""

import os

import numpy as np
import pytest

os.environ["WIREFRAME_MOCK"] = "1"
os.environ["WIREFRAME_LOG_LEVEL"] = "DEBUG"


@pytest.fixture(autouse=True)
def _init_settings(tmp_path):
    """Create a fresh DB and config store for every test."""
    db_path = str(tmp_path / "test.db")
    os.environ["WIREFRAME_DB_PATH"] = db_path

    from backend.config_store import init_config_store, reset_config_store
    from backend.database import get_connection, run_migrations
    from backend.services.worker import stop_worker, batch_queue

    # Ensure clean state before test
    batch_queue.queue.clear()
    stop_worker()
    
    conn = get_connection(db_path)
    run_migrations(conn)
    init_config_store(conn)
    yield conn
    reset_config_store()
    conn.close()
    
    # Clean up after test
    stop_worker()
    batch_queue.queue.clear()


@pytest.fixture
def clean_image():
    """Solid-grey 640x480 image (low noise)."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def noisy_image():
    """Random-noise 640x480 image."""
    return np.random.RandomState(42).randint(
        0, 256, (480, 640, 3)).astype(np.uint8)


@pytest.fixture
def dark_image():
    """Dark 640x480 image (mean brightness < 90)."""
    return np.full((480, 640, 3), 40, dtype=np.uint8)

@pytest.fixture(autouse=True)
def _reset_rate_limit():
    from backend.auth import reset_rate_limits
    reset_rate_limits()

from fastapi.testclient import TestClient

from backend.auth import hash_password
from main import app


def _create_user(conn, username, role):
    phash, salt = hash_password("password123")
    cur = conn.execute(
        "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
        (username, phash, salt, role)
    )
    conn.commit()
    return cur.lastrowid

@pytest.fixture
def client_unauth():
    return TestClient(app)

def _get_auth_client(conn, role, username):
    _create_user(conn, username, role)
    client = TestClient(app)
    # login
    res = client.post("/api/auth/login", json={"username": username, "password": "password123"}, headers={"X-Requested-With": "XMLHttpRequest"})
    assert res.status_code == 200
    # Client now has the session cookie
    if "session" in res.cookies:
        client.cookies.set("session", res.cookies["session"])
    # Add CSRF header by default to all requests
    client.headers.update({"X-Requested-With": "XMLHttpRequest"})
    return client

@pytest.fixture
def client_admin(_init_settings):
    return _get_auth_client(_init_settings, "admin", "admin_user")

@pytest.fixture
def client_inspector(_init_settings):
    return _get_auth_client(_init_settings, "field_inspector", "inspector_user")

@pytest.fixture
def client_engineer(_init_settings):
    return _get_auth_client(_init_settings, "lgu_engineer", "engineer_user")
