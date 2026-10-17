"""Tests verifying the Batch A audit fixes and B3/B4 constraints."""

import csv
import io
import os
import secrets
import sqlite3
from pathlib import Path
from PIL import Image
from PIL.Image import DecompressionBombError
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from main import app
from backend.database import get_connection
from backend.auth import (
    hash_password,
    check_rate_limit,
    clear_rate_limit,
    _rate_limits,
    _rate_limit_lock,
    _hash_token,
)
from backend.config_store import NO_WIRES_ACTION_TEXT
from backend.utils.uploads import generate_safe_filename, validate_file_extension
from backend.routes.reporting import sanitize_csv_cell, _get_analysis_data
from backend.utils.report_pdf import build_pdf


def _create_user(conn, username, password, role="admin"):
    phash, salt = hash_password(password)
    cur = conn.execute(
        "INSERT OR REPLACE INTO users (username, password_hash, salt, role, active) VALUES (?, ?, ?, ?, 1)",
        (username, phash, salt, role)
    )
    conn.commit()
    return cur.lastrowid


def _login_client(client, username, password, headers=None):
    h = {"X-Requested-With": "XMLHttpRequest"}
    if headers:
        h.update(headers)
    res = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
        headers=h
    )
    return res


# --- Task 1: Rate Limiter Tests ---

def test_rate_limit_behavior_clear_on_success_and_multi_ip(monkeypatch):
    """Test 4 failed logins, successful login clears IP/username key, 5 failures yield 429, and other IP is untouched."""
    monkeypatch.setenv("TRUSTED_PROXY", "1")
    conn = get_connection()
    try:
        _create_user(conn, "rl_test_user", "CorrectPassword123", role="admin")
        client = TestClient(app)

        ip1 = "192.168.10.1"
        ip2 = "192.168.10.2"

        # 1 failure from IP 2
        r_ip2 = _login_client(client, "rl_test_user", "wrong", headers={"X-Forwarded-For": ip2})
        assert r_ip2.status_code == 401

        # 4 failures from IP 1
        for _ in range(4):
            r = _login_client(client, "rl_test_user", "wrong", headers={"X-Forwarded-For": ip1})
            assert r.status_code == 401

        with _rate_limit_lock:
            key1 = f"login:{ip1}:rl_test_user"
            key2 = f"login:{ip2}:rl_test_user"
            assert len(_rate_limits.get(key1, [])) == 4
            assert len(_rate_limits.get(key2, [])) == 1

        # Successful login from IP 1 clears only IP 1's key
        r_ok = _login_client(client, "rl_test_user", "CorrectPassword123", headers={"X-Forwarded-For": ip1})
        assert r_ok.status_code == 200

        with _rate_limit_lock:
            assert key1 not in _rate_limits or len(_rate_limits[key1]) == 0
            # IP 2 counter is completely untouched
            assert len(_rate_limits.get(key2, [])) == 1

        # 5 consecutive failures from IP 1 now triggers 429
        for _ in range(5):
            r_fail = _login_client(client, "rl_test_user", "wrong", headers={"X-Forwarded-For": ip1})
            assert r_fail.status_code == 401

        # 6th attempt triggers 429
        r_blocked = _login_client(client, "rl_test_user", "wrong", headers={"X-Forwarded-For": ip1})
        assert r_blocked.status_code == 429
        assert "Too many requests" in r_blocked.json()["detail"]

        # IP 2 is still not blocked (it only had 1 failure)
        r_ip2_again = _login_client(client, "rl_test_user", "wrong", headers={"X-Forwarded-For": ip2})
        assert r_ip2_again.status_code == 401
    finally:
        conn.close()


# --- Task 3: Uploads Sanitization Tests ---

def test_uploads_empty_and_whitespace_name():
    """Empty and whitespace filenames raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        generate_safe_filename("")
    with pytest.raises(ValueError, match="cannot be empty"):
        generate_safe_filename("   ")


def test_uploads_no_extension():
    """Filenames without an extension raise ValueError."""
    with pytest.raises(ValueError, match="must have an extension"):
        generate_safe_filename("myphoto")


def test_uploads_only_dots():
    """Filenames consisting only of dots raise ValueError."""
    with pytest.raises(ValueError, match="only of dots"):
        generate_safe_filename(".")
    with pytest.raises(ValueError, match="only of dots"):
        generate_safe_filename("..")
    with pytest.raises(ValueError, match="only of dots"):
        generate_safe_filename("...")


def test_uploads_dots_with_valid_extension():
    """Filenames with dot-only stem and valid extension fall back to 'upload' stem."""
    safe = generate_safe_filename("....jpg")
    assert safe.endswith("_upload.jpg")
    safe2 = generate_safe_filename(".png")
    assert safe2.endswith("_upload.png")


def test_uploads_unicode_sanitized():
    """Unicode characters are sanitized while preserving valid extension and alphanumeric chars."""
    safe = generate_safe_filename("mötör_⚡_test.jpg")
    assert safe.endswith(".jpg")
    assert ".." not in safe
    assert "m_t_r" in safe


def test_uploads_stem_over_100_chars_truncated():
    """Filenames with stems longer than 100 characters are truncated to 100 characters."""
    long_stem = "a" * 150
    safe = generate_safe_filename(f"{long_stem}.png")
    assert safe.endswith(".png")
    # Base after uuid_ prefix is 100 chars + 4 chars (.png)
    body = safe.split("_", 1)[1]
    stem = body[:-4]
    assert len(stem) == 100
    assert stem == "a" * 100


def test_uploads_windows_reserved_names():
    """Windows reserved names (CON, NUL, etc.) are prefixed with upload_."""
    safe_con = generate_safe_filename("CON.jpg")
    assert safe_con.endswith("_upload_CON.jpg")

    safe_nul = generate_safe_filename("nul.png")
    assert safe_nul.endswith("_upload_nul.png")

    safe_prn = generate_safe_filename("PRN.jpeg")
    assert safe_prn.endswith("_upload_PRN.jpeg")


def test_uploads_extension_validation():
    """Disallowed extensions are rejected with ValueError."""
    with pytest.raises(ValueError, match="Disallowed file extension"):
        generate_safe_filename("shell.exe")
    with pytest.raises(ValueError, match="Disallowed file extension"):
        generate_safe_filename("script.sh")
    with pytest.raises(ValueError, match="Disallowed file extension"):
        generate_safe_filename("doc.pdf")


def test_uploads_collision_and_path_traversal():
    """Path traversal components are stripped and identical names receive distinct prefixes."""
    raw = "../../secret/pole_photo.jpg"
    s1 = generate_safe_filename(raw)
    s2 = generate_safe_filename(raw)
    assert s1 != s2
    assert ".." not in s1
    assert "/" not in s1
    assert "\\" not in s1
    assert s1.endswith("_pole_photo.jpg")
    assert s2.endswith("_pole_photo.jpg")


# --- Task 4: CSV Export Formula Quoting & Header Integrity ---

def test_export_csv_escapes_cmd_and_writes_filename_and_notes():
    """CSV export quotes =cmd formula cells and writes both filename and notes."""
    conn = get_connection()
    try:
        uid = _create_user(conn, "csv_admin", "password123", role="admin")
        cur_site = conn.execute("INSERT INTO sites (name, barangay, street) VALUES ('Site CSV', 'Brgy A', 'St B')")
        site_id = cur_site.lastrowid

        cur_an = conn.execute(
            """INSERT INTO analyses (filename, status, company, barangay, street, notes, site_id, is_mock, created_by)
               VALUES ('test_pole.jpg', 'complete', '=cmd|'' /C calc''!A0', 'Brgy A', 'St B', 'sample notes', ?, 1, ?)""",
            (site_id, uid)
        )
        aid = cur_an.lastrowid
        conn.commit()

        client = TestClient(app)
        _login_client(client, "csv_admin", "password123")

        res = client.post(
            "/api/reports/csv",
            json={"analysis_ids": [aid], "include_mock": True},
            headers={"X-Requested-With": "XMLHttpRequest"}
        )
        assert res.status_code == 200
        assert res.headers["content-type"] == "text/csv; charset=utf-8"

        csv_reader = csv.DictReader(io.StringIO(res.text))
        rows = list(csv_reader)
        assert len(rows) == 1

        row = rows[0]
        # Assert filename and notes headers and values exist
        assert "filename" in row
        assert "notes" in row
        assert row["filename"] == "test_pole.jpg"
        assert row["notes"] == "sample notes"

        # Assert =cmd was escaped with leading single quote
        assert row["company"] == "'=cmd|' /C calc'!A0"
        assert row["company"].startswith("'=")
    finally:
        conn.close()


def test_field_inspector_cannot_export_guest_records():
    """Field inspector cannot export guest analyses via CSV or GeoJSON."""
    conn = get_connection()
    try:
        insp_id = _create_user(conn, "field_user_exp", "password123", role="field_inspector")
        cur_guest = conn.execute(
            "INSERT INTO analyses (filename, status, is_public, created_by) VALUES ('guest_exp.jpg', 'complete', 1, NULL)"
        )
        guest_aid = cur_guest.lastrowid
        conn.commit()

        client = TestClient(app)
        _login_client(client, "field_user_exp", "password123")

        res_csv = client.post(
            "/api/reports/csv",
            json={"analysis_ids": [guest_aid], "include_mock": True},
            headers={"X-Requested-With": "XMLHttpRequest"}
        )
        assert res_csv.status_code == 403

        res_geo = client.post(
            "/api/reports/geojson",
            json={"analysis_ids": [guest_aid], "include_mock": True},
            headers={"X-Requested-With": "XMLHttpRequest"}
        )
        assert res_geo.status_code == 403
    finally:
        conn.close()


# --- Task 5: no_wires PDF Action Text & Non-Safe Presentation ---

def test_no_wires_pdf_data_never_says_safe():
    """no_wires status produces safety_level 'No Wires Detected' (never 'Safe') and standard action_text."""
    conn = get_connection()
    try:
        cur = conn.execute("INSERT INTO analyses (filename, status) VALUES ('nowires_test.jpg', 'no_wires')")
        aid = cur.lastrowid
        conn.commit()

        data = _get_analysis_data(conn, aid)
        assert data["safety_level"] == "No Wires Detected"
        assert data["safety_level"] != "Safe"
        expected_action = "No wires were detected in this image. This is not a clearance or safety certification. Retake the photo if wires are visible."
        assert data["action_text"] == expected_action
    finally:
        conn.close()


# --- Task 7: Batch Upload Limits and Decompression Bomb ---

def test_batch_upload_max_10_files_limit():
    """Batch upload with more than 10 files returns 400."""
    client = TestClient(app)
    # Prepare 11 dummy files
    files = [
        ("files", (f"file_{i}.jpg", io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\xff\xd9"), "image/jpeg"))
        for i in range(11)
    ]
    res = client.post(
        "/api/batches",
        files=files,
        data={"company": "Test Co", "barangay": "Brgy 1", "street": "St 1"},
        headers={"X-Requested-With": "XMLHttpRequest"}
    )
    assert res.status_code == 400
    assert "Maximum 10 files allowed per batch" in res.json()["detail"]


def test_batch_upload_max_60mb_total_limit(monkeypatch):
    """Batch upload with total size exceeding 60MB returns 400."""
    import tempfile
    client = TestClient(app)
    files = [
        ("files", ("f1.jpg", io.BytesIO(b"fake data"), "image/jpeg")),
    ]

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "tell", lambda self: 65 * 1024 * 1024)

    res = client.post(
        "/api/batches",
        files=files,
        data={"company": "Test Co", "barangay": "Brgy 1", "street": "St 1"},
        headers={"X-Requested-With": "XMLHttpRequest"}
    )
    assert res.status_code == 400
    assert "Batch exceeds maximum size of 60MB" in res.json()["detail"]


def test_batch_upload_decompression_bomb_error(monkeypatch):
    """Batch upload catching PIL DecompressionBombError returns 400."""
    client = TestClient(app)
    files = [
        ("files", ("bomb.jpg", io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF"), "image/jpeg")),
    ]

    def mock_open(*args, **kwargs):
        raise DecompressionBombError("Image exceeds decompression limit")

    monkeypatch.setattr("backend.routes.batches.Image.open", mock_open)

    res = client.post(
        "/api/batches",
        files=files,
        data={"company": "Test Co", "barangay": "Brgy 1", "street": "St 1"},
        headers={"X-Requested-With": "XMLHttpRequest"}
    )
    assert res.status_code == 400
    assert "Image exceeds decompression pixel limit" in res.json()["detail"]


# --- Guest can fetch own original & Worker error sweep ---

def test_guest_can_fetch_own_original(tmp_path, monkeypatch):
    """Guest can fetch their own original analysis image using guest_session cookie."""
    db_abs = os.environ.get("WIREFRAME_DB_PATH", os.path.abspath("wireframe.db"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_DB_PATH", db_abs)

    conn = get_connection()
    try:
        Path("uploads").mkdir(exist_ok=True)
        img = Image.new("RGB", (100, 100), color="blue")
        img_filename = "test_guest_orig.jpg"
        img.save(os.path.join("uploads", img_filename))

        guest_token = secrets.token_urlsafe(32)
        guest_sid = _hash_token(guest_token)

        cur = conn.execute(
            "INSERT INTO analyses (filename, status, is_public, guest_session_id) VALUES (?, 'complete', 1, ?)",
            (img_filename, guest_sid)
        )
        aid = cur.lastrowid
        conn.commit()

        client = TestClient(app)
        client.cookies.set("guest_session", guest_token)
        res = client.get(f"/api/analysis/{aid}/original")
        assert res.status_code == 200
        assert res.headers["content-type"] in ("image/jpeg", "application/octet-stream")
    finally:
        conn.close()
        p = Path("uploads") / "test_guest_orig.jpg"
        if p.exists():
            p.unlink()


def test_ranking_excludes_mock_by_default():
    """Ranking route /api/mpc/ranking excludes mock analyses by default, includes with include_mock=true."""
    conn = get_connection()
    try:
        uid = _create_user(conn, "eng_user", "password12345", role="lgu_engineer")
        cur_site = conn.execute("INSERT INTO sites (name, barangay, street) VALUES ('Site A', 'Brgy 1', 'St 1')")
        site_id = cur_site.lastrowid

        conn.execute(
            "INSERT INTO analyses (filename, status, site_id, risk_score, is_mock) VALUES ('mock.jpg', 'complete', ?, 80.0, 1)",
            (site_id,)
        )
        conn.execute(
            "INSERT INTO analyses (filename, status, site_id, risk_score, is_mock) VALUES ('real.jpg', 'complete', ?, 40.0, 0)",
            (site_id,)
        )
        conn.commit()

        client = TestClient(app)
        _login_client(client, "eng_user", "password12345")

        res_default = client.get("/api/mpc/ranking")
        assert res_default.status_code == 200
        rankings = res_default.json()["rankings"]
        assert len(rankings) == 1
        assert rankings[0]["analysis_count"] == 1
        assert abs(rankings[0]["avg_risk"] - 40.0) < 0.01

        res_mock = client.get("/api/mpc/ranking?include_mock=true")
        assert res_mock.status_code == 200
        rankings_mock = res_mock.json()["rankings"]
        assert len(rankings_mock) == 1
        assert rankings_mock[0]["analysis_count"] == 2
        assert abs(rankings_mock[0]["avg_risk"] - 60.0) < 0.01
    finally:
        conn.close()


def test_csrf_origin_vs_host_validation():
    """Matching Origin passes, foreign Origin gives 403, and missing both Origin and X-Requested-With gives 403."""
    client = TestClient(app)

    # 1. Matching Origin passes
    res_match = client.post(
        "/api/auth/login",
        json={"username": "fake", "password": "fake"},
        headers={"Origin": "http://testserver", "Host": "testserver"}
    )
    # Reaches credentials check (401), proving it passed verify_csrf
    assert res_match.status_code == 401
    assert "Invalid username or password" in res_match.json()["detail"]

    # 2. Foreign Origin gives 403
    res_foreign = client.post(
        "/api/auth/login",
        json={"username": "fake", "password": "fake"},
        headers={"Origin": "http://evil.com", "Host": "testserver"}
    )
    assert res_foreign.status_code == 403
    assert "CSRF check failed" in res_foreign.json()["detail"]

    # 3. Neither header gives 403
    res_neither = client.post(
        "/api/auth/login",
        json={"username": "fake", "password": "fake"}
    )
    assert res_neither.status_code == 403
    assert "CSRF check failed" in res_neither.json()["detail"]

    # 4. X-Requested-With bypass still passes
    res_xhr = client.post(
        "/api/auth/login",
        json={"username": "fake", "password": "fake"},
        headers={"X-Requested-With": "XMLHttpRequest"}
    )
    assert res_xhr.status_code == 401


def test_batch_upload_path_traversal_route_level(tmp_path, monkeypatch):
    """POST /api/batches with filename '../../evil.jpg' saves inside uploads/ with prefixed name and nothing outside."""
    db_abs = os.environ.get("WIREFRAME_DB_PATH", os.path.abspath("wireframe.db"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_DB_PATH", db_abs)

    client = TestClient(app)
    
    # Valid 100x100 JPEG bytes
    img_buf = io.BytesIO()
    Image.new("RGB", (100, 100), color="green").save(img_buf, format="JPEG")
    img_bytes = img_buf.getvalue()

    files = [
        ("files", ("../../evil.jpg", io.BytesIO(img_bytes), "image/jpeg"))
    ]

    res = client.post(
        "/api/batches",
        files=files,
        data={"company": "Traversal Co", "barangay": "Brgy 1", "street": "St 1"},
        headers={"X-Requested-With": "XMLHttpRequest"}
    )
    assert res.status_code == 200
    batch_id = res.json()["batch_id"]

    # Check database record
    conn = get_connection()
    target_path = None
    saved_filename = None
    try:
        row = conn.execute("SELECT filename FROM analyses WHERE batch_id = ?", (batch_id,)).fetchone()
        assert row is not None
        saved_filename = row["filename"]
        assert ".." not in saved_filename
        assert "/" not in saved_filename
        assert "\\" not in saved_filename
        assert saved_filename.endswith("_evil.jpg")

        # Verify file is inside uploads/
        target_path = Path("uploads") / saved_filename
        assert target_path.exists()

        # Assert no file was written outside uploads/
        assert not Path("evil.jpg").exists()
        assert not Path("../../evil.jpg").exists()
    finally:
        conn.close()
        if target_path and target_path.exists():
            target_path.unlink()
        if saved_filename:
            thumb_path = Path("uploads") / f"thumb_{saved_filename}"
            if thumb_path.exists():
                thumb_path.unlink()


def test_temp_file_removed(tmp_path):
    """Temporary PDF image files are created by process_image and deleted during PDF generation."""
    from backend.utils.report_pdf import build_pdf, process_image

    img_dir = tmp_path / "img_test"
    img_dir.mkdir()
    orig_path = str(img_dir / "orig.jpg")
    Image.new("RGB", (200, 200), color="red").save(orig_path)

    # Prove process_image really creates the temp file
    temp_path = process_image(orig_path)
    assert temp_path is not None
    assert temp_path == orig_path + "_temp_pdf.jpg"
    assert os.path.exists(temp_path)
    # Remove it so we can test build_pdf creating and removing it
    os.remove(temp_path)

    pdf_out = str(img_dir / "report.pdf")
    data = {
        "id": 9999,
        "company": "Clean Corp",
        "original_path": orig_path,
        "wires": []
    }

    build_pdf(data, pdf_out)
    assert os.path.exists(pdf_out)
    assert not os.path.exists(temp_path)


def test_worker_failure_leaves_no_queued_rows(monkeypatch):
    """Worker exception marks remaining queued/processing analyses as error."""
    from backend.services.worker import _batch_worker, batch_queue

    conn = get_connection()
    try:
        cur_batch = conn.execute("INSERT INTO batches (status) VALUES ('queued')")
        bid = cur_batch.lastrowid
        conn.execute("INSERT INTO analyses (filename, status, batch_id) VALUES ('item1.jpg', 'queued', ?)", (bid,))
        conn.execute("INSERT INTO analyses (filename, status, batch_id) VALUES ('item2.jpg', 'queued', ?)", (bid,))
        conn.commit()

        def mock_crash(*args, **kwargs):
            raise RuntimeError("Simulated crash in worker engine")

        import backend.services.worker as worker_module
        monkeypatch.setattr(worker_module, "process_analysis", mock_crash)

        batch_queue.queue.clear()
        batch_queue.put(bid)
        batch_queue.put(None)
        _batch_worker()

        b = conn.execute("SELECT status FROM batches WHERE id = ?", (bid,)).fetchone()
        assert b["status"] == "error"

        rows = conn.execute("SELECT id, status, summary FROM analyses WHERE batch_id = ?", (bid,)).fetchall()
        assert len(rows) == 2
        for r in rows:
            assert r["status"] == "error"
            assert "Batch worker exception" in r["summary"]
    finally:
        conn.close()


def test_clear_rate_limit_all_ips_fallback():
    """When ip is not provided, clear_rate_limit clears all IP keys for that username."""
    with _rate_limit_lock:
        _rate_limits["login:1.1.1.1:user_fb"] = [100.0]
        _rate_limits["login:2.2.2.2:user_fb"] = [101.0]
        _rate_limits["login:3.3.3.3:other_user"] = [102.0]

    clear_rate_limit(action="login", username="user_fb", ip=None)

    with _rate_limit_lock:
        assert "login:1.1.1.1:user_fb" not in _rate_limits
        assert "login:2.2.2.2:user_fb" not in _rate_limits
        assert "login:3.3.3.3:other_user" in _rate_limits


def test_no_wires_text_forced_in_public_config_and_reporting():
    """Even if database settings table contains 'Site clear.', public config and reporting force NO_WIRES_ACTION_TEXT."""
    import json
    conn = get_connection()
    original_row = conn.execute("SELECT value_json FROM settings WHERE key = 'action_mapping'").fetchone()
    client = TestClient(app)
    try:
        # Override settings with old "Site clear." text
        stale_mapping = {
            "low": "No immediate action required.",
            "critical": "Immediate dispatch required.",
            "no_wires": "No wires detected. Site clear."
        }
        conn.execute("UPDATE settings SET value_json = ? WHERE key = 'action_mapping'", (json.dumps(stale_mapping),))
        conn.commit()

        # 1. Check GET /api/config/public
        res = client.get("/api/config/public")
        assert res.status_code == 200
        data = res.json()
        assert "action_mapping" in data
        assert data["action_mapping"]["no_wires"] == NO_WIRES_ACTION_TEXT
        assert "Site clear." not in json.dumps(data)

        # 2. Check reporting._get_analysis_data for no_wires analysis
        cur = conn.execute("INSERT INTO analyses (filename, status) VALUES ('nowires.jpg', 'no_wires')")
        aid = cur.lastrowid
        conn.commit()

        report_data = _get_analysis_data(conn, aid)
        assert report_data is not None
        assert report_data["safety_level"] == "No Wires Detected"
        assert report_data["action_text"] == NO_WIRES_ACTION_TEXT
        assert "Site clear." not in report_data["action_text"]
    finally:
        if original_row:
            conn.execute("UPDATE settings SET value_json = ? WHERE key = 'action_mapping'", (original_row["value_json"],))
        conn.commit()
        conn.close()


def test_pdf_margin_config_snapshot_uses_analysis_margin_error_m(tmp_path, monkeypatch):
    """PDF and data config snapshot uses margin_error_m (0.8 -> 0.8, None -> 'Not available'), never 0.5."""
    captured_flowables = []
    from reportlab.platypus import SimpleDocTemplate
    orig_build = SimpleDocTemplate.build

    def mock_build(self, flowables, *args, **kwargs):
        captured_flowables.extend(flowables)
        return orig_build(self, flowables, *args, **kwargs)

    monkeypatch.setattr(SimpleDocTemplate, "build", mock_build)

    conn = get_connection()
    try:
        # Case 1: margin_error_m = 0.8
        cur1 = conn.execute("INSERT INTO analyses (filename, status, margin_error_m) VALUES ('item_08.jpg', 'complete', 0.8)")
        aid1 = cur1.lastrowid
        conn.commit()

        data_08 = _get_analysis_data(conn, aid1)
        assert data_08["margin_error_m"] == 0.8
        assert data_08["config"]["margin"] == 0.8

        pdf_path1 = str(tmp_path / "report_08.pdf")
        captured_flowables.clear()
        build_pdf(data_08, pdf_path1)

        snapshot_paragraphs = [f.text for f in captured_flowables if hasattr(f, "text") and "Config Snapshot" in f.text]
        assert len(snapshot_paragraphs) == 1
        assert "Margin: 0.8m" in snapshot_paragraphs[0]
        assert "0.5" not in snapshot_paragraphs[0]

        # Case 2: margin_error_m = None
        cur2 = conn.execute("INSERT INTO analyses (filename, status) VALUES ('item_none.jpg', 'complete')")
        aid2 = cur2.lastrowid
        conn.commit()

        data_none = _get_analysis_data(conn, aid2)
        # In SQLite default table, margin_error_m is absent or None
        assert data_none["config"]["margin"] == "Not available"

        pdf_path2 = str(tmp_path / "report_none.pdf")
        captured_flowables.clear()
        build_pdf(data_none, pdf_path2)

        snapshot_paragraphs2 = [f.text for f in captured_flowables if hasattr(f, "text") and "Config Snapshot" in f.text]
        assert len(snapshot_paragraphs2) == 1
        assert "Margin: Not available" in snapshot_paragraphs2[0]
        assert "0.5" not in snapshot_paragraphs2[0]
    finally:
        conn.close()


@pytest.mark.anyio
async def test_batch_upload_missing_or_blank_filename_returns_400():
    """Batch upload with None, empty string, or whitespace filename returns 400, not 500."""
    from starlette.datastructures import UploadFile as StarletteUploadFile
    from starlette.requests import Request
    from backend.routes.batches import create_batch

    mock_req = Request({"type": "http", "method": "POST", "headers": [], "client": ("127.0.0.1", 12345)})

    # Test None, empty string, and whitespace
    for bad_name in [None, "", "   "]:
        mock_file = StarletteUploadFile(file=io.BytesIO(b"fake data"), filename=bad_name)
        with pytest.raises(HTTPException) as exc_info:
            await create_batch(
                req=mock_req,
                files=[mock_file],
                company="Test Co",
                barangay="Brgy 1",
                street="St 1",
                sector=None,
                gps_lat=None,
                gps_lng=None,
                user=None
            )

# --- Task 5 & 6: Safe Parsing and B1 Calibration Tests ---

import json
import math
from backend.utils.clearance_helpers import (
    parse_params_json,
    parse_centerline_json,
    get_lowest_y,
    evaluate_wire,
    rollup_analysis_clearance,
    determine_overall_safety,
)
from backend.utils.pipeline import execute_pipeline


def test_safe_parsing_params_and_centerline():
    """Verify parse_params_json and parse_centerline_json handle JSON, legacy str(), and malicious input."""
    # 1. Valid JSON
    valid_json_params = json.dumps({"a": 12.5, "residual": 0.42})
    assert parse_params_json(valid_json_params) == {"a": 12.5, "residual": 0.42}

    valid_json_centerline = json.dumps([[10.0, 20.0], [30.0, 45.0]])
    assert parse_centerline_json(valid_json_centerline) == [[10.0, 20.0], [30.0, 45.0]]

    # 2. Legacy str(dict) and str(list)
    legacy_str_params = str({"a": 15.0, "residual": 0.88})
    assert parse_params_json(legacy_str_params) == {"a": 15.0, "residual": 0.88}

    legacy_str_centerline = str([[5.0, 10.0], [15.0, 25.0]])
    assert parse_centerline_json(legacy_str_centerline) == [[5.0, 10.0], [15.0, 25.0]]

    # 3. Malicious code string must NOT execute and must return safe fallback
    malicious_params = "__import__('os').system('echo hacked')"
    assert parse_params_json(malicious_params) == {}

    malicious_centerline = "__import__('os').system('echo hacked')"
    assert parse_centerline_json(malicious_centerline) == []

    # 4. None and empty
    assert parse_params_json(None) == {}
    assert parse_params_json("") == {}
    assert parse_centerline_json(None) == []
    assert parse_centerline_json("") == []


def test_determine_overall_safety_worst_of_severity_and_guards():
    """TASK 1: determine_overall_safety worst-of severity order and evaluate_wire unavailable guard."""
    assert determine_overall_safety(["Safe"]) == "Safe"
    assert determine_overall_safety(["Safe", "Unknown"]) == "Unknown"
    assert determine_overall_safety(["Safe", "Caution"]) == "Caution"
    assert determine_overall_safety([]) == "Unknown"
    assert determine_overall_safety(["Unknown"]) == "Unknown"
    assert determine_overall_safety(["Unsafe", "Safe"]) == "Unsafe"
    assert determine_overall_safety(["Critical", "Caution"]) == "Critical"
    assert determine_overall_safety(["foo_unrecognized"]) == "Unknown"

    # evaluate_wire with pole_bbox=None gives clearance_status "unavailable" and safety_level != "Safe"
    ew = evaluate_wire(
        class_name="power_line",
        confidence=0.9,
        sag_px=10.0,
        span_px=100.0,
        crs_weighted=5.0,  # low crs
        lowest_y=50.0,
        pole_bbox=None,  # No pole bbox!
        meters_per_px=0.02,
        image_height=500,
        pec_min_map={"default": 5.5},
        params_dict={"a": 100.0, "residual": 0.05},
    )
    assert ew["clearance_status"] == "unavailable"
    assert ew["safety_level"] != "Safe"
    assert ew["safety_level"] == "Unknown"


def test_recompute_consistency_with_pipeline_original_meters_per_px(tmp_path, monkeypatch):
    """TASK 3: Non-vacuous consistency test with positive clearance and valid clearance_ok."""
    db_abs = os.environ.get("WIREFRAME_DB_PATH", os.path.abspath("wireframe.db"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_DB_PATH", db_abs)

    import cv2
    import numpy as np
    from backend.utils.cv.types import SegmentationResult
    from backend.utils.cv.calibration import calibrate_scale
    import backend.utils.pipeline

    # Custom mock segmentation ensuring wires have lowest_y < ground_y
    def mock_segment_image(image_path, original_shape=(480, 640)):
        h, w = original_shape
        boxes = []
        classes = []
        confidences = []
        masks = []
        names = {0: "telecom_cable", 1: "power_line", 2: "dead_abandoned", 3: "pole"}

        # Pole bbox: ground_y = 440 < image_height - 5 (475)
        boxes.append([400.0, 50.0, 450.0, 440.0])
        classes.append(3)
        confidences.append(0.95)
        pole_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(pole_mask, (400, 50), (450, 440), 255, -1)
        masks.append(pole_mask)

        # Wire 1: power_line, y between 100 and 140 (lowest_y = 140 < ground_y 440)
        pts1 = []
        for x in range(50, 380, 5):
            y = int(100 + 40 * ((x - 215) / 165) ** 2)
            pts1.append((x, y))
        pts1_arr = np.array(pts1, np.int32)
        mask1 = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(mask1, [pts1_arr], False, 255, 10)
        x, y, bw, bh = cv2.boundingRect(pts1_arr)
        boxes.append([x, y, x + bw, y + bh])
        classes.append(1)
        confidences.append(0.92)
        masks.append(mask1)

        # Wire 2: telecom_cable, y between 150 and 190 (lowest_y = 190 < ground_y 440)
        pts2 = []
        for x in range(60, 390, 5):
            y = int(150 + 40 * ((x - 225) / 165) ** 2)
            pts2.append((x, y))
        pts2_arr = np.array(pts2, np.int32)
        mask2 = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(mask2, [pts2_arr], False, 255, 10)
        x, y, bw, bh = cv2.boundingRect(pts2_arr)
        boxes.append([x, y, x + bw, y + bh])
        classes.append(0)
        confidences.append(0.88)
        masks.append(mask2)

        return SegmentationResult(
            mask_available=True,
            boxes=boxes,
            classes=classes,
            confidences=confidences,
            masks=masks,
            names=names,
        )

    import backend.utils.cv.mock_generator
    monkeypatch.setattr(backend.utils.cv.mock_generator, "generate_mock_segmentation", mock_segment_image)

    img_path = str(tmp_path / "real_cons_mock.jpg")
    noise_arr = np.random.RandomState(42).randint(0, 256, (480, 640, 3)).astype(np.uint8)
    img = Image.fromarray(noise_arr)
    img.save(img_path)

    conn = get_connection()
    try:
        cur = conn.execute("INSERT INTO analyses (filename, original_path, status) VALUES ('real_cons_mock.jpg', ?, 'queued')", (img_path,))
        aid = cur.lastrowid
        _create_user(conn, "cal_real_admin", "Password123!", role="admin")
        conn.commit()

        execute_pipeline(aid, img_path, conn, model=None)

        db_a = conn.execute(
            "SELECT status, width, height, pole_bbox_json, clearance_available, clearance_ok, sag_meters, margin_error_m FROM analyses WHERE id = ?",
            (aid,)
        ).fetchone()
        assert db_a["status"] == "complete"
        assert db_a["pole_bbox_json"] is not None

        # Non-vacuous checks
        assert db_a["clearance_available"] is not None and db_a["clearance_available"] > 0
        assert db_a["clearance_ok"] in (1, 2)
        assert db_a["sag_meters"] is not None and db_a["sag_meters"] > 0

        db_wires = conn.execute(
            "SELECT id, clearance_m, sag_m, clearance_status, safety_level FROM wires WHERE analysis_id = ? ORDER BY id",
            (aid,)
        ).fetchall()
        assert len(db_wires) > 0
        for dw in db_wires:
            assert dw["clearance_m"] is not None and dw["clearance_m"] > 0
            assert dw["sag_m"] is not None and dw["sag_m"] > 0

        pole_bbox = json.loads(db_a["pole_bbox_json"])
        scale_res = calibrate_scale(pole_bbox=pole_bbox, image_height=db_a["height"])
        pipeline_m_per_px = scale_res.meters_per_px
        assert pipeline_m_per_px is not None and pipeline_m_per_px > 0

        # Two in-bounds points D px apart
        D = 100.0
        p1 = {"x": 50.0, "y": 50.0}
        p2 = {"x": 50.0, "y": 150.0}
        real_length_m = pipeline_m_per_px * D
        assert 0.1 <= real_length_m <= 100.0

        client = TestClient(app)
        _login_client(client, "cal_real_admin", "Password123!")

        res = client.post(
            f"/api/analysis/{aid}/calibrate",
            json={"points": [p1, p2], "real_length_m": real_length_m},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert res.status_code == 200, res.text

        post_a = conn.execute(
            "SELECT clearance_available, clearance_ok, sag_meters, margin_error_m FROM analyses WHERE id = ?",
            (aid,)
        ).fetchone()
        post_wires = conn.execute(
            "SELECT id, clearance_m, sag_m, clearance_status, safety_level FROM wires WHERE analysis_id = ? ORDER BY id",
            (aid,)
        ).fetchall()

        assert math.isclose(post_a["clearance_available"], db_a["clearance_available"], abs_tol=1e-6)
        assert post_a["clearance_ok"] == db_a["clearance_ok"]
        assert math.isclose(post_a["sag_meters"], db_a["sag_meters"], abs_tol=1e-6)
        assert math.isclose(post_a["margin_error_m"], db_a["margin_error_m"], abs_tol=1e-6)

        assert len(post_wires) == len(db_wires)
        for pw, ow in zip(post_wires, db_wires):
            assert math.isclose(pw["clearance_m"], ow["clearance_m"], abs_tol=1e-6)
            assert math.isclose(pw["sag_m"], ow["sag_m"], abs_tol=1e-6)
            assert pw["clearance_status"] == ow["clearance_status"]
            assert pw["safety_level"] == ow["safety_level"]
    finally:
        conn.close()


def test_rollup_mixed_safe_unknown_never_produces_safe(tmp_path, monkeypatch):
    """TASK 4: A report/analysis with wire levels [Safe, Unknown] must not produce safety_level 'Safe'."""
    from backend.utils.clearance_helpers import determine_overall_safety
    from backend.routes.reporting import _get_analysis_data

    # Pure helper test
    assert determine_overall_safety(["Safe", "Unknown"]) == "Unknown"
    assert determine_overall_safety(["Safe", "Unknown"]) != "Safe"

    # Backend reporting test
    db_abs = os.environ.get("WIREFRAME_DB_PATH", os.path.abspath("wireframe.db"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_DB_PATH", db_abs)

    conn = get_connection()
    try:
        cur = conn.execute("INSERT INTO analyses (filename, status) VALUES ('rollup_test.jpg', 'complete')")
        aid = cur.lastrowid
        conn.execute(
            "INSERT INTO wires (analysis_id, class_name, confidence, crs, band, safety_level) VALUES (?, 'power_line', 0.9, 10.0, 'Low', 'Safe')",
            (aid,)
        )
        conn.execute(
            "INSERT INTO wires (analysis_id, class_name, confidence, crs, band, safety_level) VALUES (?, 'telecom_cable', 0.85, 12.0, 'Low', 'Unknown')",
            (aid,)
        )
        conn.commit()

        rep_data = _get_analysis_data(conn, aid)
        assert rep_data is not None
        assert rep_data["safety_level"] != "Safe"
        assert rep_data["safety_level"] == "Unknown"
    finally:
        conn.close()


def test_calibrate_ratio_two_lengths_and_db_holds_latest(tmp_path, monkeypatch):
    """TASK 3a: two different real_length_m on the same analysis give clearance in exact ratio, DB holds latest."""
    db_abs = os.environ.get("WIREFRAME_DB_PATH", os.path.abspath("wireframe.db"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_DB_PATH", db_abs)

    conn = get_connection()
    try:
        pole_bbox = [100.0, 50.0, 150.0, 400.0]
        cur = conn.execute(
            """INSERT INTO analyses (filename, status, width, height, pole_bbox_json, assumed_pole_height_m,
                                      sag_meters, clearance_available, clearance_ok, margin_error_m)
               VALUES ('ratio_test.jpg', 'complete', 600, 500, ?, 10.0, 1.0, 5.0, 1, 1.0)""",
            (json.dumps(pole_bbox),)
        )
        aid = cur.lastrowid
        conn.execute(
            """INSERT INTO wires (analysis_id, class_name, confidence, crs, band, sag_px, span_px,
                                  sag_m, clearance_m, clearance_status, safety_level, centerline_json, params_json)
               VALUES (?, 'power_line', 0.9, 15.0, 'Low', 50.0, 200.0, 1.0, 5.0, 'ok', 'Safe', '[[100, 200], [200, 250]]', '{"a": 100, "residual": 0.5}')""",
            (aid,)
        )
        _create_user(conn, "cal_ratio_user", "Password123!", role="admin")
        conn.commit()

        client = TestClient(app)
        _login_client(client, "cal_ratio_user", "Password123!")

        p1 = {"x": 50.0, "y": 50.0}
        p2 = {"x": 50.0, "y": 150.0}

        # Calibrate 1: real_length_m = 2.0
        res1 = client.post(
            f"/api/analysis/{aid}/calibrate",
            json={"points": [p1, p2], "real_length_m": 2.0},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert res1.status_code == 200
        c1 = res1.json()["clearance_m"]

        # Calibrate 2: real_length_m = 4.0
        res2 = client.post(
            f"/api/analysis/{aid}/calibrate",
            json={"points": [p1, p2], "real_length_m": 4.0},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert res2.status_code == 200
        c2 = res2.json()["clearance_m"]

        # Exact ratio check
        assert math.isclose(c2 / c1, 4.0 / 2.0, rel_tol=1e-6)

        # DB holds the latest
        db_a = conn.execute("SELECT clearance_available FROM analyses WHERE id = ?", (aid,)).fetchone()
        assert math.isclose(db_a["clearance_available"], c2, abs_tol=1e-6)

        # Calibrations audit log holds both, latest matches c2
        cal_rows = conn.execute("SELECT clearance_m FROM calibrations WHERE analysis_id = ? ORDER BY id", (aid,)).fetchall()
        assert len(cal_rows) == 2
        assert math.isclose(cal_rows[-1]["clearance_m"], c2, abs_tol=1e-6)
    finally:
        conn.close()


def test_calibrate_nan_and_infinity_return_422(tmp_path, monkeypatch):
    """TASK 3b: raw JSON bodies containing NaN and Infinity return 422."""
    db_abs = os.environ.get("WIREFRAME_DB_PATH", os.path.abspath("wireframe.db"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_DB_PATH", db_abs)

    conn = get_connection()
    try:
        pole_bbox = [100.0, 50.0, 150.0, 400.0]
        cur = conn.execute(
            "INSERT INTO analyses (filename, status, width, height, pole_bbox_json) VALUES ('nan_test.jpg', 'complete', 600, 500, ?)",
            (json.dumps(pole_bbox),)
        )
        aid = cur.lastrowid
        _create_user(conn, "cal_nan_user", "Password123!", role="admin")
        conn.commit()

        client = TestClient(app)
        _login_client(client, "cal_nan_user", "Password123!")

        headers = {"X-Requested-With": "XMLHttpRequest", "Content-Type": "application/json"}

        # NaN in point coordinates
        body_nan_x = b'{"points": [{"x": NaN, "y": 50.0}, {"x": 50.0, "y": 150.0}], "real_length_m": 5.0}'
        res = client.post(f"/api/analysis/{aid}/calibrate", content=body_nan_x, headers=headers)
        assert res.status_code == 422

        # NaN in real_length_m
        body_nan_len = b'{"points": [{"x": 50.0, "y": 50.0}, {"x": 50.0, "y": 150.0}], "real_length_m": NaN}'
        res = client.post(f"/api/analysis/{aid}/calibrate", content=body_nan_len, headers=headers)
        assert res.status_code == 422

        # Infinity in point coordinates
        body_inf_y = b'{"points": [{"x": 50.0, "y": Infinity}, {"x": 50.0, "y": 150.0}], "real_length_m": 5.0}'
        res = client.post(f"/api/analysis/{aid}/calibrate", content=body_inf_y, headers=headers)
        assert res.status_code == 422

        # Infinity in real_length_m
        body_inf_len = b'{"points": [{"x": 50.0, "y": 50.0}, {"x": 50.0, "y": 150.0}], "real_length_m": Infinity}'
        res = client.post(f"/api/analysis/{aid}/calibrate", content=body_inf_len, headers=headers)
        assert res.status_code == 422

        # -Infinity in real_length_m
        body_ninf_len = b'{"points": [{"x": 50.0, "y": 50.0}, {"x": 50.0, "y": 150.0}], "real_length_m": -Infinity}'
        res = client.post(f"/api/analysis/{aid}/calibrate", content=body_ninf_len, headers=headers)
        assert res.status_code == 422
    finally:
        conn.close()


def test_calibrate_rollback_on_abort_trigger(tmp_path, monkeypatch):
    """TASK 3c: ROLLBACK abort trigger in DB ensures no new calibrations row and unchanged wires values."""
    db_abs = os.environ.get("WIREFRAME_DB_PATH", os.path.abspath("wireframe.db"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_DB_PATH", db_abs)

    conn = get_connection()
    try:
        pole_bbox = [100.0, 50.0, 150.0, 400.0]
        cur = conn.execute(
            """INSERT INTO analyses (filename, status, width, height, pole_bbox_json, assumed_pole_height_m,
                                      sag_meters, clearance_available, clearance_ok, margin_error_m)
               VALUES ('rollback_test.jpg', 'complete', 600, 500, ?, 10.0, 0.5, 5.0, 1, 1.0)""",
            (json.dumps(pole_bbox),)
        )
        aid = cur.lastrowid
        conn.execute(
            """INSERT INTO wires (analysis_id, class_name, confidence, crs, band, sag_px, span_px,
                                  sag_m, clearance_m, clearance_status, safety_level, centerline_json, params_json)
               VALUES (?, 'power_line', 0.9, 15.0, 'Low', 50.0, 200.0, 0.5, 5.0, 'ok', 'Safe', '[[100, 200], [200, 250]]', '{"a": 100, "residual": 0.5}')""",
            (aid,)
        )
        _create_user(conn, "cal_rollback_user", "Password123!", role="admin")
        conn.commit()

        # Add ABORT trigger on analyses
        conn.execute("CREATE TRIGGER test_abort_analyses BEFORE UPDATE ON analyses BEGIN SELECT RAISE(ABORT, 'boom'); END;")
        conn.commit()

        cal_before = conn.execute("SELECT COUNT(*) AS c FROM calibrations WHERE analysis_id = ?", (aid,)).fetchone()["c"]
        wire_before = conn.execute("SELECT clearance_m, sag_m, safety_level FROM wires WHERE analysis_id = ?", (aid,)).fetchone()

        client = TestClient(app, raise_server_exceptions=False)
        _login_client(client, "cal_rollback_user", "Password123!")

        res = client.post(
            f"/api/analysis/{aid}/calibrate",
            json={"points": [{"x": 50.0, "y": 50.0}, {"x": 50.0, "y": 150.0}], "real_length_m": 8.0},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert res.status_code == 500

        # Assert no new calibrations row and unchanged wires values
        cal_after = conn.execute("SELECT COUNT(*) AS c FROM calibrations WHERE analysis_id = ?", (aid,)).fetchone()["c"]
        wire_after = conn.execute("SELECT clearance_m, sag_m, safety_level FROM wires WHERE analysis_id = ?", (aid,)).fetchone()

        assert cal_after == cal_before
        assert math.isclose(wire_after["clearance_m"], wire_before["clearance_m"], abs_tol=1e-6)
        assert math.isclose(wire_after["sag_m"], wire_before["sag_m"], abs_tol=1e-6)
        assert wire_after["safety_level"] == wire_before["safety_level"]
    finally:
        try:
            conn.execute("DROP TRIGGER IF EXISTS test_abort_analyses")
            conn.commit()
        except Exception:
            pass
        conn.close()


def test_calibrate_legacy_str_dict_and_json_params_json(tmp_path, monkeypatch):
    """TASK 3d: legacy-format params_json (str(dict)) and JSON params_json both calibrate correctly."""
    db_abs = os.environ.get("WIREFRAME_DB_PATH", os.path.abspath("wireframe.db"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_DB_PATH", db_abs)

    conn = get_connection()
    try:
        pole_bbox = [100.0, 50.0, 150.0, 400.0]
        cur = conn.execute(
            """INSERT INTO analyses (filename, status, width, height, pole_bbox_json, assumed_pole_height_m)
               VALUES ('params_format_test.jpg', 'complete', 600, 500, ?, 10.0)""",
            (json.dumps(pole_bbox),)
        )
        aid = cur.lastrowid
        # Wire 1: legacy str(dict)
        conn.execute(
            """INSERT INTO wires (analysis_id, class_name, confidence, crs, band, sag_px, span_px,
                                  centerline_json, params_json)
               VALUES (?, 'power_line', 0.9, 15.0, 'Low', 50.0, 200.0, '[[100, 200], [200, 250]]', "{'a': 100.0, 'residual': 0.5}")""",
            (aid,)
        )
        # Wire 2: standard JSON
        conn.execute(
            """INSERT INTO wires (analysis_id, class_name, confidence, crs, band, sag_px, span_px,
                                  centerline_json, params_json)
               VALUES (?, 'telecom_cable', 0.85, 20.0, 'Low', 40.0, 180.0, '[[120, 220], [220, 260]]', '{"a": 120.0, "residual": 0.4}')""",
            (aid,)
        )
        _create_user(conn, "cal_params_user", "Password123!", role="admin")
        conn.commit()

        client = TestClient(app)
        _login_client(client, "cal_params_user", "Password123!")

        res = client.post(
            f"/api/analysis/{aid}/calibrate",
            json={"points": [{"x": 50.0, "y": 50.0}, {"x": 50.0, "y": 150.0}], "real_length_m": 5.0},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert res.status_code == 200, res.text

        # Verify both wires were calibrated and params_json is valid JSON
        updated_wires = conn.execute("SELECT params_json, clearance_m, clearance_status FROM wires WHERE analysis_id = ? ORDER BY id", (aid,)).fetchall()
        assert len(updated_wires) == 2
        for w in updated_wires:
            assert w["clearance_m"] is not None and w["clearance_m"] > 0
            assert w["clearance_status"] in ("ok", "marginal", "below")
            parsed = json.loads(w["params_json"])
            assert "clearance_low_m" in parsed
            assert "clearance_high_m" in parsed
    finally:
        conn.close()


def test_parse_params_json_and_analysis_engine_no_eval():
    """TASK 4: parse_params_json covers JSON, legacy str(dict), and malicious strings without executing them."""
    # JSON
    assert parse_params_json('{"a": 50.5, "residual": 0.12}') == {"a": 50.5, "residual": 0.12}
    # Legacy str(dict)
    assert parse_params_json("{'a': 50.5, 'residual': 0.12}") == {"a": 50.5, "residual": 0.12}
    # Malicious string
    malicious = "__import__('os').system('echo pwned')"
    assert parse_params_json(malicious) == {}
    # Non-dict JSON or ast
    assert parse_params_json("[1, 2, 3]") == {}
    assert parse_params_json("42") == {}


def test_calibrate_public_analysis_rbac_lgu_vs_inspector(tmp_path, monkeypatch):
    """TASK 5: lgu_engineer calibrates guest/public analysis (created_by NULL, is_public=1) -> 200; field_inspector -> 403."""
    db_abs = os.environ.get("WIREFRAME_DB_PATH", os.path.abspath("wireframe.db"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_DB_PATH", db_abs)

    conn = get_connection()
    try:
        pole_bbox = [100.0, 50.0, 150.0, 400.0]
        cur = conn.execute(
            """INSERT INTO analyses (filename, status, width, height, pole_bbox_json, assumed_pole_height_m,
                                      created_by, is_public, guest_session_id)
               VALUES ('public_cal.jpg', 'complete', 600, 500, ?, 10.0, NULL, 1, 'guest_hash_123')""",
            (json.dumps(pole_bbox),)
        )
        aid = cur.lastrowid
        conn.execute(
            """INSERT INTO wires (analysis_id, class_name, confidence, crs, band, sag_px, span_px,
                                  centerline_json, params_json)
               VALUES (?, 'power_line', 0.9, 15.0, 'Low', 50.0, 200.0, '[[100, 200], [200, 250]]', '{"a": 100, "residual": 0.5}')""",
            (aid,)
        )
        _create_user(conn, "lgu_cal_user", "Password123!", role="lgu_engineer")
        _create_user(conn, "insp_cal_user", "Password123!", role="field_inspector")
        conn.commit()

        client = TestClient(app)

        # 1. Field inspector gets 403 Forbidden
        _login_client(client, "insp_cal_user", "Password123!")
        res_insp = client.post(
            f"/api/analysis/{aid}/calibrate",
            json={"points": [{"x": 50.0, "y": 50.0}, {"x": 50.0, "y": 150.0}], "real_length_m": 5.0},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert res_insp.status_code == 403

        # 2. LGU engineer gets 200 OK
        _login_client(client, "lgu_cal_user", "Password123!")
        res_lgu = client.post(
            f"/api/analysis/{aid}/calibrate",
            json={"points": [{"x": 50.0, "y": 50.0}, {"x": 50.0, "y": 150.0}], "real_length_m": 5.0},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert res_lgu.status_code == 200
    finally:
        conn.close()


def test_pipeline_persists_ground_reference_and_margin(tmp_path, monkeypatch):
    """Pipeline persists width, height, pole_bbox_json, and margin_error_m for a complete analysis."""
    db_abs = os.environ.get("WIREFRAME_DB_PATH", os.path.abspath("wireframe.db"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_DB_PATH", db_abs)
    import numpy as np
    # Create test image with texture to pass Laplacian variance check
    img_path = str(tmp_path / "safe_pipeline_persist.jpg")
    noise_arr = np.random.RandomState(42).randint(0, 256, (480, 640, 3)).astype(np.uint8)
    img = Image.fromarray(noise_arr)
    img.save(img_path)

    conn = get_connection()
    try:
        cur = conn.execute("INSERT INTO analyses (filename, original_path, status) VALUES ('safe_pipeline_persist.jpg', ?, 'queued')", (img_path,))
        aid = cur.lastrowid
        conn.commit()

        execute_pipeline(aid, img_path, conn, model=None)

        row = conn.execute("SELECT status, width, height, pole_bbox_json, margin_error_m FROM analyses WHERE id = ?", (aid,)).fetchone()
        assert row["status"] == "complete"
        assert row["width"] == 640
        assert row["height"] == 480
        assert row["pole_bbox_json"] is not None
        bbox = json.loads(row["pole_bbox_json"])
        assert len(bbox) == 4
        assert row["margin_error_m"] is not None
        assert row["margin_error_m"] > 0
    finally:
        conn.close()


def test_calibrate_legacy_null_pole_bbox_returns_409():
    """Legacy row with NULL pole_bbox_json returns 409 with exact ground reference error message."""
    conn = get_connection()
    try:
        cur = conn.execute("INSERT INTO analyses (filename, status, width, height, pole_bbox_json) VALUES ('legacy.jpg', 'complete', 600, 500, NULL)")
        aid = cur.lastrowid
        _create_user(conn, "cal_user", "Password123!", role="admin")
        conn.commit()

        client = TestClient(app)
        _login_client(client, "cal_user", "Password123!")

        res = client.post(
            f"/api/analysis/{aid}/calibrate",
            json={"points": [{"x": 10, "y": 10}, {"x": 50, "y": 50}], "real_length_m": 5.0},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert res.status_code == 409
        assert res.json()["detail"] == "This analysis has no stored ground reference; re-analyze the image to enable calibration."
    finally:
        conn.close()


def test_calibrated_analysis_updates_margin_in_reporting_and_pdf(tmp_path, monkeypatch):
    """A calibrated analysis shows the updated margin in _get_analysis_data config and in the PDF snapshot."""
    captured_flowables = []
    from reportlab.platypus import SimpleDocTemplate
    orig_build = SimpleDocTemplate.build

    def mock_build(self, flowables, *args, **kwargs):
        captured_flowables.extend(flowables)
        return orig_build(self, flowables, *args, **kwargs)

    monkeypatch.setattr(SimpleDocTemplate, "build", mock_build)

    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO analyses (
                filename, status, width, height, pole_bbox_json, assumed_pole_height_m,
                clearance_available, clearance_ok, margin_error_m
            ) VALUES ('cal_report.jpg', 'complete', 600, 500, ?, 10.0, 5.0, 1, 1.0)""",
            (json.dumps([100.0, 50.0, 150.0, 450.0]),),
        )
        aid = cur.lastrowid
        conn.execute(
            """INSERT INTO wires (
                analysis_id, class_name, confidence, crs, band, sag_px, span_px,
                sag_m, clearance_m, clearance_status, safety_level, centerline_json, params_json
            ) VALUES (?, 'power_line', 0.9, 10.0, 'Low', 20.0, 200.0, 0.4, 5.0, 'ok', 'Safe', ?, ?)""",
            (aid, json.dumps([[100, 200], [200, 250], [300, 200]]), json.dumps({"a": 100, "residual": 0.5})),
        )
        _create_user(conn, "cal_report_admin", "Password123!", role="admin")
        conn.commit()

        client = TestClient(app)
        _login_client(client, "cal_report_admin", "Password123!")

        # Calibrate with length = 10m over 100px -> 0.1 m/px
        # ground_y = 450, lowest_y = 250 -> clearance_px = 200 -> clearance_m = 200 * 0.1 = 20.0m (wait: 20m is 2*pole_height, boundary)
        # Let's use length = 2.0m over 100px -> 0.02 m/px -> clearance_px = 200 -> clearance_m = 4.0m
        # margin_error_m = 4.0 * 0.20 = 0.8m
        res = client.post(
            f"/api/analysis/{aid}/calibrate",
            json={"points": [{"x": 100, "y": 50}, {"x": 100, "y": 150}], "real_length_m": 2.0},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert res.status_code == 200
        data = res.json()
        assert math.isclose(data["clearance_m"], 4.0, abs_tol=1e-5)
        assert math.isclose(data["margin_error_m"], 0.8, abs_tol=1e-5)

        # Check _get_analysis_data
        report_data = _get_analysis_data(conn, aid)
        assert math.isclose(report_data["config"]["margin"], 0.8, abs_tol=1e-5)
        assert report_data["scale_source"] == "Manual Calibration"

        pdf_path = str(tmp_path / "calibrated_report.pdf")
        captured_flowables.clear()
        build_pdf(report_data, pdf_path)

        snapshot_paragraphs = [f.text for f in captured_flowables if hasattr(f, "text") and "Config Snapshot" in f.text]
        assert len(snapshot_paragraphs) == 1
        assert "Margin: 0.8m" in snapshot_paragraphs[0]

        scale_paragraphs = [f.text for f in captured_flowables if hasattr(f, "text") and "Scale Source" in f.text]
        assert len(scale_paragraphs) == 1
        assert "Manual Calibration" in scale_paragraphs[0]
    finally:
        conn.close()


def test_calibrate_auth_and_access_control():
    """Guest gets 401, inspector on other user's analysis gets 403, inspector on own gets 200, admin gets 200."""
    conn = get_connection()
    try:
        u_insp1 = _create_user(conn, "inspector_1", "Pass1!", role="field_inspector")
        u_insp2 = _create_user(conn, "inspector_2", "Pass2!", role="field_inspector")
        _create_user(conn, "admin_user", "AdminPass1!", role="admin")

        # Analysis owned by inspector 1
        cur = conn.execute(
            """INSERT INTO analyses (
                filename, status, width, height, pole_bbox_json, assumed_pole_height_m, created_by
            ) VALUES ('insp1.jpg', 'complete', 600, 500, ?, 10.0, ?)""",
            (json.dumps([100.0, 50.0, 150.0, 450.0]), u_insp1),
        )
        aid = cur.lastrowid
        conn.execute(
            """INSERT INTO wires (
                analysis_id, class_name, confidence, crs, band, sag_px, span_px,
                sag_m, clearance_m, clearance_status, safety_level, centerline_json, params_json
            ) VALUES (?, 'power_line', 0.9, 10.0, 'Low', 20.0, 200.0, 0.4, 5.0, 'ok', 'Safe', ?, ?)""",
            (aid, json.dumps([[100, 200], [200, 250], [300, 200]]), json.dumps({"a": 100, "residual": 0.5})),
        )
        conn.commit()

        # 1. Unauthenticated guest -> 401
        guest_client = TestClient(app)
        res_guest = guest_client.post(
            f"/api/analysis/{aid}/calibrate",
            json={"points": [{"x": 100, "y": 50}, {"x": 100, "y": 150}], "real_length_m": 2.0},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert res_guest.status_code == 401

        # 2. Inspector 2 (different user) -> 403
        insp2_client = TestClient(app)
        _login_client(insp2_client, "inspector_2", "Pass2!")
        res_insp2 = insp2_client.post(
            f"/api/analysis/{aid}/calibrate",
            json={"points": [{"x": 100, "y": 50}, {"x": 100, "y": 150}], "real_length_m": 2.0},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert res_insp2.status_code == 403

        # 3. Inspector 1 (owner) -> 200
        insp1_client = TestClient(app)
        _login_client(insp1_client, "inspector_1", "Pass1!")
        res_insp1 = insp1_client.post(
            f"/api/analysis/{aid}/calibrate",
            json={"points": [{"x": 100, "y": 50}, {"x": 100, "y": 150}], "real_length_m": 2.0},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert res_insp1.status_code == 200

        # 4. Admin -> 200
        admin_client = TestClient(app)
        _login_client(admin_client, "admin_user", "AdminPass1!")
        res_admin = admin_client.post(
            f"/api/analysis/{aid}/calibrate",
            json={"points": [{"x": 100, "y": 50}, {"x": 100, "y": 150}], "real_length_m": 2.0},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert res_admin.status_code == 200
    finally:
        conn.close()


def test_calibrate_validations_and_guards():
    """Verify 409 for non-complete/no-wires, 422 for distance/bounds/NaN/implausible clearance."""
    conn = get_connection()
    try:
        _create_user(conn, "guard_admin", "Password123!", role="admin")

        # Analysis not complete (processing)
        cur_proc = conn.execute("INSERT INTO analyses (filename, status, width, height, pole_bbox_json) VALUES ('p.jpg', 'processing', 600, 500, '[0,0,10,10]')")
        aid_proc = cur_proc.lastrowid

        # Analysis complete with 0 wires
        cur_nowires = conn.execute("INSERT INTO analyses (filename, status, width, height, pole_bbox_json, assumed_pole_height_m) VALUES ('nw.jpg', 'complete', 600, 500, '[100,50,150,450]', 10.0)")
        aid_nowires = cur_nowires.lastrowid

        # Valid analysis with 1 wire
        cur_valid = conn.execute("INSERT INTO analyses (filename, status, width, height, pole_bbox_json, assumed_pole_height_m) VALUES ('v.jpg', 'complete', 600, 500, '[100,50,150,450]', 10.0)")
        aid_valid = cur_valid.lastrowid
        conn.execute(
            """INSERT INTO wires (
                analysis_id, class_name, confidence, crs, band, sag_px, span_px,
                sag_m, clearance_m, clearance_status, safety_level, centerline_json, params_json
            ) VALUES (?, 'power_line', 0.9, 10.0, 'Low', 20.0, 200.0, 0.4, 5.0, 'ok', 'Safe', ?, ?)""",
            (aid_valid, json.dumps([[100, 200], [200, 250], [300, 200]]), json.dumps({"a": 100, "residual": 0.5})),
        )
        conn.commit()

        client = TestClient(app)
        _login_client(client, "guard_admin", "Password123!")
        headers = {"X-Requested-With": "XMLHttpRequest"}

        # 1. Status not complete -> 409
        r = client.post(f"/api/analysis/{aid_proc}/calibrate", json={"points": [{"x": 10, "y": 10}, {"x": 50, "y": 50}], "real_length_m": 5.0}, headers=headers)
        assert r.status_code == 409
        assert "complete" in r.json()["detail"].lower()

        # 2. 0 wires -> 409
        r = client.post(f"/api/analysis/{aid_nowires}/calibrate", json={"points": [{"x": 10, "y": 10}, {"x": 50, "y": 50}], "real_length_m": 5.0}, headers=headers)
        assert r.status_code == 409
        assert "no wires" in r.json()["detail"].lower()

        # 3. Points length != 2
        r = client.post(f"/api/analysis/{aid_valid}/calibrate", json={"points": [{"x": 10, "y": 10}], "real_length_m": 5.0}, headers=headers)
        assert r.status_code == 422

        r = client.post(f"/api/analysis/{aid_valid}/calibrate", json={"points": [{"x": 10, "y": 10}, {"x": 20, "y": 20}, {"x": 30, "y": 30}], "real_length_m": 5.0}, headers=headers)
        assert r.status_code == 422

        # 4. Length out of bounds
        r = client.post(f"/api/analysis/{aid_valid}/calibrate", json={"points": [{"x": 10, "y": 10}, {"x": 50, "y": 50}], "real_length_m": 0.05}, headers=headers)
        assert r.status_code == 422

        r = client.post(f"/api/analysis/{aid_valid}/calibrate", json={"points": [{"x": 10, "y": 10}, {"x": 50, "y": 50}], "real_length_m": 150.0}, headers=headers)
        assert r.status_code == 422

        # 5. Distance < 10 px
        r = client.post(f"/api/analysis/{aid_valid}/calibrate", json={"points": [{"x": 10, "y": 10}, {"x": 12, "y": 14}], "real_length_m": 5.0}, headers=headers)
        assert r.status_code == 422
        assert "10 pixels" in r.json()["detail"]

        # 6. Points outside image bounds (img is 600x500)
        r = client.post(f"/api/analysis/{aid_valid}/calibrate", json={"points": [{"x": -5, "y": 10}, {"x": 50, "y": 50}], "real_length_m": 5.0}, headers=headers)
        assert r.status_code == 422

        r = client.post(f"/api/analysis/{aid_valid}/calibrate", json={"points": [{"x": 10, "y": 10}, {"x": 650, "y": 50}], "real_length_m": 5.0}, headers=headers)
        assert r.status_code == 422

        # 7. Implausible clearance: distance=10px, real_length=100m -> 10 m/px -> clearance = 200px * 10 = 2000m > 2*10m
        r = client.post(f"/api/analysis/{aid_valid}/calibrate", json={"points": [{"x": 100, "y": 100}, {"x": 100, "y": 110}], "real_length_m": 100.0}, headers=headers)
        assert r.status_code == 422
        assert "implausible" in r.json()["detail"].lower()
    finally:
        conn.close()


def test_calibrate_idempotency_and_cache_invalidation(tmp_path, monkeypatch):
    """Calling calibrate is idempotent, records audit trail, and removes cached PDF reports."""
    db_abs = os.environ.get("WIREFRAME_DB_PATH", os.path.abspath("wireframe.db"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_DB_PATH", db_abs)

    conn = get_connection()
    try:
        _create_user(conn, "idem_admin", "Password123!", role="admin")

        cur = conn.execute(
            """INSERT INTO analyses (
                filename, status, width, height, pole_bbox_json, assumed_pole_height_m,
                sag_meters, clearance_available, clearance_ok, margin_error_m
            ) VALUES ('idem.jpg', 'complete', 600, 500, ?, 10.0, 0.5, 5.0, 1, 1.0)""",
            (json.dumps([100.0, 50.0, 150.0, 450.0]),),
        )
        aid = cur.lastrowid
        conn.execute(
            """INSERT INTO wires (
                analysis_id, class_name, confidence, crs, band, sag_px, span_px,
                sag_m, clearance_m, clearance_status, safety_level, centerline_json, params_json
            ) VALUES (?, 'power_line', 0.9, 10.0, 'Low', 20.0, 200.0, 0.4, 5.0, 'ok', 'Safe', ?, ?)""",
            (aid, json.dumps([[100, 200], [200, 250], [300, 200]]), json.dumps({"a": 100, "residual": 0.5})),
        )

        # Create dummy report cache
        base_outputs = Path("outputs").resolve()
        base_outputs.mkdir(exist_ok=True)
        rep_file = base_outputs / f"report_test_inval_{aid}.pdf"
        rep_file.write_text("dummy report pdf")
        conn.execute("INSERT INTO reports (analysis_id, path) VALUES (?, ?)", (aid, str(rep_file)))
        conn.commit()

        client = TestClient(app)
        _login_client(client, "idem_admin", "Password123!")
        headers = {"X-Requested-With": "XMLHttpRequest"}

        cal_payload = {"points": [{"x": 100, "y": 50}, {"x": 100, "y": 150}], "real_length_m": 2.0}

        # First calibration
        r1 = client.post(f"/api/analysis/{aid}/calibrate", json=cal_payload, headers=headers)
        assert r1.status_code == 200
        d1 = r1.json()

        # Cache file must have been deleted
        assert not rep_file.exists()
        assert conn.execute("SELECT COUNT(*) as c FROM reports WHERE analysis_id = ?", (aid,)).fetchone()["c"] == 0

        # Verify calibration audit entry exists
        cal_row = conn.execute("SELECT payload_json FROM calibrations WHERE analysis_id = ? ORDER BY id DESC", (aid,)).fetchone()
        audit_payload = json.loads(cal_row["payload_json"])
        assert "previous_state" in audit_payload
        assert audit_payload["previous_state"]["clearance_available"] == 5.0

        # Second calibration with same parameters (idempotent)
        r2 = client.post(f"/api/analysis/{aid}/calibrate", json=cal_payload, headers=headers)
        assert r2.status_code == 200
        d2 = r2.json()

        assert math.isclose(d1["clearance_m"], d2["clearance_m"], abs_tol=1e-6)
        assert math.isclose(d1["sag_m"], d2["sag_m"], abs_tol=1e-6)
        assert math.isclose(d1["margin_error_m"], d2["margin_error_m"], abs_tol=1e-6)
        assert d1["safety_level"] == d2["safety_level"]
    finally:
        conn.close()


def test_estimate_clearance_below_ground_returns_unavailable():
    """TASK 1: estimate_clearance returns status 'unavailable' when lowest_wire_y > ground_y, never 'below'."""
    from backend.utils.cv.calibration import estimate_clearance
    from backend.formulas import compute_clearance_status

    # 1. Wire below ground (lowest_wire_y = 950 > ground_y = 900)
    res_below = estimate_clearance(950, [0, 0, 10, 900], 0.1, 1000)
    assert res_below["status"] == "unavailable"
    assert res_below["clearance_m"] is None
    assert res_below["clearance_low_m"] is None
    assert res_below["clearance_high_m"] is None
    assert res_below["reason"] == "wire below ground reference (invalid geometry)"

    # Status helper check: never "below"
    status = compute_clearance_status(res_below["clearance_m"], res_below["clearance_low_m"], res_below["clearance_high_m"], 5.5)
    assert status == "unavailable"
    assert status != "below"

    # 2. Wire above ground unchanged (lowest_wire_y = 500 < ground_y = 900)
    res_above = estimate_clearance(500, [0, 0, 10, 900], 0.1, 1000)
    assert res_above["status"] == "estimated"
    assert res_above["clearance_m"] == 40.0
    assert res_above["clearance_low_m"] == 32.0
    assert res_above["clearance_high_m"] == 48.0


def test_compute_safety_level_source_guard_low_vs_high(tmp_path, monkeypatch):
    """TASK 2: compute_safety_level guard at the source: ('Low', 'unavailable') -> Unknown; ('High', 'unavailable') -> Unsafe. Both pipeline and analysis_engine produce Unknown."""
    import numpy as np
    from backend.formulas import compute_safety_level
    from backend.utils.clearance_helpers import evaluate_wire

    # 1. Direct compute_safety_level tests
    res_low = compute_safety_level("Low", "unavailable", "Clearance unavailable.")
    assert res_low["safety_level"] == "Unknown"
    assert "Clearance could not be estimated; cannot confirm safe." in res_low["reasons"]

    res_high = compute_safety_level("High", "unavailable", "Clearance unavailable.")
    assert res_high["safety_level"] == "Unsafe"
    assert "Clearance could not be estimated; verdict is based on CRS only." in res_high["reasons"]

    # 2. evaluate_wire path (used by pipeline and calibration)
    ew = evaluate_wire(
        class_name="power_line",
        confidence=0.9,
        sag_px=50.0,
        span_px=200.0,
        crs_weighted=10.0,  # Low CRS band
        lowest_y=150.0,
        pole_bbox=None,  # clearance unavailable
        meters_per_px=0.02,
        image_height=500,
        pec_min_map={"power_line": 5.5},
        params_dict={},
    )
    assert ew["clearance_status"] == "unavailable"
    assert ew["safety_level"] == "Unknown"

    # 3. analysis_engine path
    db_abs = os.environ.get("WIREFRAME_DB_PATH", os.path.abspath("wireframe.db"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_DB_PATH", db_abs)
    monkeypatch.setenv("WIREFRAME_MOCK", "1")

    from backend.utils.analysis_engine import analyze_image
    from backend.utils.cv.mock_generator import generate_mock_segmentation
    # Generate nopole mock image
    nopole_img_path = str(tmp_path / "nopole_test.jpg")
    noise_arr = np.random.RandomState(42).randint(0, 256, (480, 640, 3)).astype(np.uint8)
    Image.fromarray(noise_arr).save(nopole_img_path)

    monkeypatch.setattr(
        "backend.utils.analysis_engine.segment_image",
        lambda rgb, model, shape: generate_mock_segmentation(nopole_img_path, shape),
    )

    engine_res = analyze_image(nopole_img_path, model=None)
    assert engine_res["status"] == "success"
    # When pole is absent, clearance is unavailable and safety level must not be Safe
    assert engine_res["image_safety_level"] != "Safe"
    assert engine_res["image_safety_level"] == "Unknown"
    for w in engine_res["wires"]:
        assert w["clearance_status"] == "unavailable"
        assert w["safety_level"] != "Safe"
        assert w["safety_level"] == "Unknown"

    # 4. pipeline path
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO analyses (filename, original_path, status) VALUES ('nopole_test.jpg', ?, 'queued')",
            (nopole_img_path,),
        )
        pipe_aid = cur.lastrowid
        conn.commit()

        execute_pipeline(pipe_aid, nopole_img_path, conn, model=None)

        pipe_wires = conn.execute(
            "SELECT clearance_status, safety_level FROM wires WHERE analysis_id = ?",
            (pipe_aid,),
        ).fetchall()
        assert len(pipe_wires) > 0
        for pw in pipe_wires:
            assert pw["clearance_status"] == "unavailable"
            assert pw["safety_level"] != "Safe"
            assert pw["safety_level"] == "Unknown"
    finally:
        conn.close()


def test_standard_mock_analysis_produces_real_clearance_without_clamping(tmp_path, monkeypatch):
    """TASK 3: Standard mock analysis produces clearance_available > 0 and clearance_ok in (0,1,2) from real geometry."""
    import numpy as np
    db_abs = os.environ.get("WIREFRAME_DB_PATH", os.path.abspath("wireframe.db"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WIREFRAME_DB_PATH", db_abs)
    monkeypatch.setenv("WIREFRAME_MOCK", "1")

    img_path = str(tmp_path / "standard_mock.jpg")
    noise_arr = np.random.RandomState(42).randint(0, 256, (480, 640, 3)).astype(np.uint8)
    Image.fromarray(noise_arr).save(img_path)

    conn = get_connection()
    try:
        cur = conn.execute("INSERT INTO analyses (filename, original_path, status) VALUES ('standard_mock.jpg', ?, 'queued')", (img_path,))
        aid = cur.lastrowid
        conn.commit()

        execute_pipeline(aid, img_path, conn, model=None)

        row = conn.execute("SELECT status, pole_bbox_json, clearance_available, clearance_ok, sag_meters FROM analyses WHERE id = ?", (aid,)).fetchone()
        assert row["status"] == "complete"
        assert row["pole_bbox_json"] is not None
        assert row["clearance_available"] is not None and row["clearance_available"] > 0
        assert row["clearance_ok"] in (0, 1, 2)
        assert row["sag_meters"] is not None

        # Verify wires were not below ground and not clamped
        wires = conn.execute("SELECT clearance_m, clearance_status, safety_level FROM wires WHERE analysis_id = ?", (aid,)).fetchall()
        assert len(wires) > 0
        for w in wires:
            assert w["clearance_m"] is not None and w["clearance_m"] > 0
            assert w["clearance_status"] in ("ok", "marginal", "below")
    finally:
        conn.close()




