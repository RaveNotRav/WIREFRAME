"""Tests for Audit Batch C fixes (#9, #10, #13, + orphan upload on GPS 422).

- #10: utcnow_naive replaces datetime.utcnow(); session expires_at has no tzinfo and remains backwards-compatible.
- #13: /api/health ready field is True in mock mode or when model loaded; model_loaded remains truthful.
- #Orphan: invalid GPS upload leaves no orphan files in uploads/.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import cv2
import pytest
from backend.auth import create_session, utcnow_naive, _hash_token
from backend import main_state


def test_utcnow_naive_and_session_format_and_expiry(client_unauth, _init_settings):
    conn = _init_settings

    # 1. utcnow_naive() returns a naive UTC datetime
    now_dt = utcnow_naive()
    assert now_dt.tzinfo is None, "utcnow_naive must return naive datetime (tzinfo is None)"

    # 2. create_session stores naive ISO without timezone suffix (+00:00 or Z)
    cur = conn.execute("INSERT INTO users (username, password_hash, salt, role, active) VALUES ('test_user', 'h', 's', 'field_inspector', 1)")
    user_id = cur.lastrowid
    token = create_session(conn, user_id, expiry_hours=8)

    row = conn.execute("SELECT expires_at FROM sessions WHERE token_hash = ?", (_hash_token(token),)).fetchone()
    assert row is not None
    expires_at_str = row["expires_at"]

    # String format check: no + or Z timezone indicators
    assert "+" not in expires_at_str, "expires_at string must not contain timezone offset '+'"
    assert not expires_at_str.endswith("Z"), "expires_at string must not end with 'Z'"

    parsed = datetime.fromisoformat(expires_at_str)
    assert parsed.tzinfo is None, "Parsed expires_at must be naive"
    assert parsed > now_dt

    # 3. Valid session works through API
    client_unauth.cookies.set("session", token)
    res_valid = client_unauth.get("/api/auth/me")
    assert res_valid.status_code == 200
    assert res_valid.json()["username"] == "test_user"

    # 4. Expired session returns 401 and deletes the session row
    expired_str = (utcnow_naive() - timedelta(minutes=5)).isoformat()
    conn.execute("UPDATE sessions SET expires_at = ? WHERE token_hash = ?", (expired_str, _hash_token(token)))
    conn.commit()

    res_expired = client_unauth.get("/api/auth/me")
    assert res_expired.status_code == 401
    assert "expired" in res_expired.json()["detail"].lower()

    row_after = conn.execute("SELECT 1 FROM sessions WHERE token_hash = ?", (_hash_token(token),)).fetchone()
    assert row_after is None, "Expired session must be purged from database"


def test_api_health_ready_and_model_loaded(client_unauth, monkeypatch):
    # 1. Mock mode: ready=True, model_loaded=False (truthful), model_status='mock'
    monkeypatch.setattr("main.is_mock_mode", lambda: True)
    main_state.set_model(None)

    res_mock = client_unauth.get("/api/health")
    assert res_mock.status_code == 200
    body_mock = res_mock.json()
    assert body_mock["status"] == "ok"
    assert body_mock["ready"] is True
    assert body_mock["mock_mode"] is True
    assert body_mock["model_loaded"] is False
    assert body_mock["model_status"] == "mock"

    # 2. Production mode without model loaded: ready=False, model_loaded=False, model_status='not_loaded'
    monkeypatch.setattr("main.is_mock_mode", lambda: False)
    main_state.set_model(None)

    res_unloaded = client_unauth.get("/api/health")
    assert res_unloaded.status_code == 200
    body_unloaded = res_unloaded.json()
    assert body_unloaded["ready"] is False
    assert body_unloaded["model_loaded"] is False
    assert body_unloaded["model_status"] == "not_loaded"

    # 3. Production mode with model loaded: ready=True, model_loaded=True, model_status='loaded'
    monkeypatch.setattr("main.is_mock_mode", lambda: False)
    dummy_model = object()
    main_state.set_model(dummy_model)

    res_loaded = client_unauth.get("/api/health")
    assert res_loaded.status_code == 200
    body_loaded = res_loaded.json()
    assert body_loaded["ready"] is True
    assert body_loaded["model_loaded"] is True
    assert body_loaded["model_status"] == "loaded"

    # Clean up state
    main_state.set_model(None)


def test_orphan_upload_on_gps_422(client_admin, clean_image):
    image_bytes = cv2.imencode('.jpg', clean_image)[1].tobytes()

    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    before_files = set(uploads_dir.glob("*"))

    # Invalid latitude 999 -> 422
    data = {
        "company": "Clean Power",
        "barangay": "San Jose",
        "street": "Rizal St",
        "gps_lat": "999",
        "gps_lng": "120.9842"
    }
    files = [("file", ("test_orphan.jpg", image_bytes, "image/jpeg"))]

    res = client_admin.post("/api/analysis/upload", data=data, files=files)
    assert res.status_code == 422

    after_files = set(uploads_dir.glob("*"))
    new_files = after_files - before_files
    assert len(new_files) == 0, f"Upload with invalid GPS left orphan files: {new_files}"
