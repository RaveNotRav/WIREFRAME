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

def test_guest_can_fetch_own_original(tmp_path):
    """Guest can fetch their own original analysis image using guest_session cookie."""
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


def test_batch_upload_path_traversal_route_level(tmp_path):
    """POST /api/batches with filename '../../evil.jpg' saves inside uploads/ with prefixed name and nothing outside."""
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
        if target_path.exists():
            target_path.unlink()
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

