def test_auth_gaps(client_unauth):
    print("\n--- AUTH TESTS ---")
    res = client_unauth.get("/api/stats/dashboard")
    print("GET /api/stats/dashboard (No Auth):", res.status_code, res.json() if res.status_code != 401 else "")
    assert res.status_code == 401
    
    res = client_unauth.post("/api/batches/1/cancel")
    print("POST /api/batches/1/cancel (No Auth):", res.status_code, res.json() if res.status_code != 401 else "")
    assert res.status_code in [401, 403]

def test_upload_hardening(client_admin):
    print("\n--- UPLOAD HARDENING ---")
    fake_img = b"<?php echo 'hello'; ?>"
    files = {"file": ("../../../malicious.jpg", fake_img, "image/jpeg")}
    res = client_admin.post(
        "/api/analysis/upload", 
        files=files,
        headers={"X-Requested-With": "XMLHttpRequest"}
    )
    print("POST /api/analysis/upload (PHP as JPG):", res.status_code, res.json())
    assert res.status_code == 400
    assert "Corrupt image" in res.text

def test_form_limits(client_admin):
    print("\n--- FORM LIMITS ---")
    huge_company = "A" * 300
    huge_notes = "B" * 2500
    real_img = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    
    files = {"file": ("test.png", real_img, "image/png")}
    data = {"company": huge_company, "notes": huge_notes}
    res = client_admin.post(
        "/api/analysis/upload", 
        files=files, 
        data=data,
        headers={"X-Requested-With": "XMLHttpRequest"}
    )
    print("POST /api/analysis/upload (Oversized forms):", res.status_code, res.json())
    assert res.status_code == 422


# --- Security Batch S1 Tests ---

def test_s1_cookie_secure_flags(client_unauth, monkeypatch, _init_settings):
    from fastapi.testclient import TestClient
    from main import app
    # 1. Without WIREFRAME_PUBLIC (default/local dev over http)
    monkeypatch.delenv("WIREFRAME_PUBLIC", raising=False)
    # Upload guest image
    real_img = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    data = {"company": "Test Co", "barangay": "Test Brgy", "street": "Test St"}
    res = client_unauth.post(
        "/api/analysis/upload",
        data=data,
        files={"file": ("guest1.png", real_img, "image/png")},
        headers={"X-Requested-With": "XMLHttpRequest"}
    )
    assert res.status_code == 200
    set_cookie_header = res.headers.get("set-cookie", "")
    assert "guest_session=" in set_cookie_header
    assert "Secure" not in set_cookie_header
    assert "HttpOnly" in set_cookie_header
    assert "samesite=lax" in set_cookie_header.lower()

    # 2. With WIREFRAME_PUBLIC=1
    monkeypatch.setenv("WIREFRAME_PUBLIC", "1")
    client_unauth_pub = TestClient(app)
    res_pub = client_unauth_pub.post(
        "/api/analysis/upload",
        data=data,
        files={"file": ("guest2.png", real_img, "image/png")},
        headers={"X-Requested-With": "XMLHttpRequest"}
    )
    assert res_pub.status_code == 200
    set_cookie_pub = res_pub.headers.get("set-cookie", "")
    assert "guest_session=" in set_cookie_pub
    assert "secure" in set_cookie_pub.lower()
    assert "httponly" in set_cookie_pub.lower()

    # 3. Login session cookie with WIREFRAME_PUBLIC=1
    # Create test user
    from backend.auth import hash_password
    h, s = hash_password("Password123!")
    _init_settings.execute(
        "INSERT INTO users (username, password_hash, salt, role, active) VALUES (?, ?, ?, 'admin', 1)",
        ("s1_admin", h, s)
    )
    _init_settings.commit()

    login_res = client_unauth.post(
        "/api/auth/login",
        json={"username": "s1_admin", "password": "Password123!"},
        headers={"X-Requested-With": "XMLHttpRequest"}
    )
    assert login_res.status_code == 200
    login_set_cookie = login_res.headers.get("set-cookie", "")
    assert "session=" in login_set_cookie
    assert "secure" in login_set_cookie.lower()
    assert "httponly" in login_set_cookie.lower()
    assert "samesite=lax" in login_set_cookie.lower()


def test_s1_xff_trusted_proxy(client_unauth, monkeypatch):
    from backend.auth import reset_rate_limits, check_rate_limit
    from fastapi import Request

    reset_rate_limits()

    # When TRUSTED_PROXY is unset, right-most IP in XFF is ignored and client.host (testclient) is used
    monkeypatch.delenv("TRUSTED_PROXY", raising=False)
    scope = {
        "type": "http",
        "client": ("127.0.0.1", 12345),
        "headers": [(b"x-forwarded-for", b"6.6.6.6, 1.2.3.4")]
    }
    req1 = Request(scope)
    check_rate_limit(req1, "test_action", "user1", limit=2, window=60)
    check_rate_limit(req1, "test_action", "user1", limit=2, window=60)
    import pytest
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        check_rate_limit(req1, "test_action", "user1", limit=2, window=60)
    assert exc.value.status_code == 429

    reset_rate_limits()

    # When TRUSTED_PROXY=1, uses the last entry in X-Forwarded-For
    monkeypatch.setenv("TRUSTED_PROXY", "1")
    scope_a = {
        "type": "http",
        "client": ("127.0.0.1", 12345),
        "headers": [(b"x-forwarded-for", b"6.6.6.6, 1.2.3.4")]
    }
    scope_b = {
        "type": "http",
        "client": ("127.0.0.1", 12345),
        "headers": [(b"x-forwarded-for", b"9.9.9.9, 5.6.7.8")]
    }
    req_a = Request(scope_a)
    req_b = Request(scope_b)

    # 2 requests from 1.2.3.4 reach the limit
    check_rate_limit(req_a, "test_action", "user1", limit=2, window=60)
    check_rate_limit(req_a, "test_action", "user1", limit=2, window=60)
    with pytest.raises(HTTPException):
        check_rate_limit(req_a, "test_action", "user1", limit=2, window=60)

    # But 5.6.7.8 has its own separate bucket and succeeds
    check_rate_limit(req_b, "test_action", "user1", limit=2, window=60)


def test_s1_csrf_allowlist(client_unauth, monkeypatch):
    monkeypatch.setenv("WIREFRAME_ALLOWED_ORIGINS", "https://wireframe.test.ts.net, https://sub.example.com/")

    # 1. Matching allowed origin -> accepted
    res1 = client_unauth.post(
        "/api/auth/login",
        json={"username": "foo", "password": "bar"},
        headers={"Origin": "https://wireframe.test.ts.net"}
    )
    # Not 403 CSRF failed (will be 401 or similar auth error)
    assert res1.status_code != 403

    # 2. Matching referer when origin is absent -> accepted
    res2 = client_unauth.post(
        "/api/auth/login",
        json={"username": "foo", "password": "bar"},
        headers={"Referer": "https://wireframe.test.ts.net/login.html"}
    )
    assert res2.status_code != 403

    # 3. Disallowed evil origin -> 403
    res3 = client_unauth.post(
        "/api/auth/login",
        json={"username": "foo", "password": "bar"},
        headers={"Origin": "https://evil.attacker.com"}
    )
    assert res3.status_code == 403
    assert "CSRF" in res3.text

    # 4. Empty allowlist with mismatched host -> 403
    monkeypatch.setenv("WIREFRAME_ALLOWED_ORIGINS", "")
    res4 = client_unauth.post(
        "/api/auth/login",
        json={"username": "foo", "password": "bar"},
        headers={"Origin": "https://wireframe.test.ts.net", "Host": "127.0.0.1:8000"}
    )
    assert res4.status_code == 403


def test_s1_docs_and_states_preview_toggles(client_unauth, monkeypatch):
    # 1. /states-preview.html returns 200 locally, 404 with WIREFRAME_PUBLIC=1
    monkeypatch.delenv("WIREFRAME_PUBLIC", raising=False)
    res_preview_dev = client_unauth.get("/states-preview.html")
    assert res_preview_dev.status_code == 200

    monkeypatch.setenv("WIREFRAME_PUBLIC", "1")
    res_preview_pub = client_unauth.get("/states-preview.html")
    assert res_preview_pub.status_code == 404

    # 2. Docs endpoints default to 404 unless WIREFRAME_DOCS=1
    # Note: FastAPI builds OpenAPI routes at app instantiation. Test both via client / app.
    res_docs = client_unauth.get("/docs")
    res_openapi = client_unauth.get("/openapi.json")
    # By default WIREFRAME_DOCS is unset in test session -> 404
    assert res_docs.status_code == 404
    assert res_openapi.status_code == 404


def test_s2_security_headers_and_csp_report_only(client_unauth, monkeypatch):
    expected_enforced_csp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    )
    expected_csp = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    )

    def assert_common_headers(res, expect_hsts: bool, expect_api_no_store: bool):
        assert res.headers.get("X-Content-Type-Options") == "nosniff"
        assert res.headers.get("X-Frame-Options") == "DENY"
        assert res.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert (
            res.headers.get("Permissions-Policy")
            == "geolocation=(self), camera=(), microphone=(), payment=(), usb=()"
        )
        assert res.headers.get("Cross-Origin-Opener-Policy") == "same-origin"
        assert res.headers.get("Content-Security-Policy") == expected_enforced_csp
        assert res.headers.get("Content-Security-Policy-Report-Only") == expected_csp

        if expect_hsts:
            assert res.headers.get("Strict-Transport-Security") == "max-age=31536000"
        else:
            assert "Strict-Transport-Security" not in res.headers

        if expect_api_no_store:
            assert res.headers.get("Cache-Control") == "no-store"

    # 1. Default mode (WIREFRAME_PUBLIC unset) -> HSTS absent, design-system reachable
    monkeypatch.delenv("WIREFRAME_PUBLIC", raising=False)

    res_root = client_unauth.get("/")
    assert res_root.status_code == 200
    assert_common_headers(res_root, expect_hsts=False, expect_api_no_store=False)

    res_analyze = client_unauth.get("/analyze.html")
    assert res_analyze.status_code == 200
    assert_common_headers(res_analyze, expect_hsts=False, expect_api_no_store=False)

    res_health = client_unauth.get("/api/health")
    assert res_health.status_code == 200
    assert_common_headers(res_health, expect_hsts=False, expect_api_no_store=True)

    res_404 = client_unauth.get("/nonexistent-s2-test-path")
    assert res_404.status_code == 404
    assert_common_headers(res_404, expect_hsts=False, expect_api_no_store=False)

    res_401 = client_unauth.get("/api/auth/me")
    assert res_401.status_code == 401
    assert_common_headers(res_401, expect_hsts=False, expect_api_no_store=True)

    res_static = client_unauth.get("/assets/css/tokens.css")
    assert res_static.status_code == 200
    assert res_static.headers.get("X-Content-Type-Options") == "nosniff"
    assert_common_headers(res_static, expect_hsts=False, expect_api_no_store=False)

    res_design_dev = client_unauth.get("/pages/design-system.html")
    assert res_design_dev.status_code == 200
    assert_common_headers(res_design_dev, expect_hsts=False, expect_api_no_store=False)

    res_states_pages_dev = client_unauth.get("/pages/states-preview.html")
    assert res_states_pages_dev.status_code == 200
    assert_common_headers(res_states_pages_dev, expect_hsts=False, expect_api_no_store=False)

    res_test_js_dev = client_unauth.get("/assets/js/dashboard.test.js")
    assert res_test_js_dev.status_code == 200
    assert_common_headers(res_test_js_dev, expect_hsts=False, expect_api_no_store=False)

    # 2. Public mode (WIREFRAME_PUBLIC=1) -> HSTS present, dev pages and *.test.js return 404
    monkeypatch.setenv("WIREFRAME_PUBLIC", "1")
    for path, is_api, expected_code in [
        ("/", False, 200),
        ("/analyze.html", False, 200),
        ("/pages/analyze.html", False, 200),
        ("/assets/css/tokens.css", False, 200),
        ("/assets/js/dashboard.js", False, 200),
        ("/api/health", True, 200),
        ("/nonexistent-s2-test-path", False, 404),
        ("/api/auth/me", True, 401),
        ("/states-preview.html", False, 404),
        ("/pages/states-preview.html", False, 404),
        ("/pages/Design-System.html", False, 404),
        ("/pages/design%2Dsystem.html", False, 404),
        ("/assets/js/dashboard.test.js", False, 404),
    ]:
        res = client_unauth.get(path)
        assert res.status_code == expected_code, f"Expected {expected_code} for {path}, got {res.status_code}"
        assert_common_headers(res, expect_hsts=True, expect_api_no_store=is_api)


def test_events_endpoint_never_leaks_server_paths(client_unauth, tmp_path):
    """#12: /api/analysis/{id}/events messages contain no 'uploads' path, no backslash, and no absolute path."""
    import cv2
    import numpy as np
    from backend.auth import _hash_token
    from backend.database import get_connection
    from backend.utils.pipeline import execute_pipeline

    img_dir = tmp_path / "uploads"
    img_dir.mkdir(exist_ok=True)
    img_file = img_dir / "4326d0f9_4709.jpg"
    img = np.full((600, 800, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(img_file), img)

    conn = get_connection()
    guest_tok = "tok_events_no_leak"
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO analyses (filename, status, company, barangay, street, guest_session_id, is_public) VALUES ('4326d0f9_4709.jpg', 'queued', 'Meralco', 'San Antonio', 'Perea St', ?, 1)",
        (_hash_token(guest_tok),),
    )
    a_id = cur.lastrowid
    # Also insert a legacy event row containing a Windows server path to verify API sanitization
    cur.execute(
        "INSERT INTO pipeline_events (analysis_id, state, detail) VALUES (?, 'ImageAcquired', ?)",
        (a_id, r"Loaded uploads\4326d0f9_4709.jpg"),
    )
    conn.commit()

    execute_pipeline(a_id, str(img_file), conn, model=None, batch_id=None)
    conn.close()

    client_unauth.cookies.set("guest_session", guest_tok)
    res = client_unauth.get(f"/api/analysis/{a_id}/events")
    assert res.status_code == 200
    events = res.json()
    assert len(events) > 0

    acquired_details = [ev["detail"] for ev in events if ev.get("state") == "ImageAcquired"]
    assert acquired_details, "Expected at least one ImageAcquired event"
    for detail in acquired_details:
        assert detail == "Loaded 4709.jpg"

    for ev in events:
        detail = ev.get("detail") or ""
        assert "uploads" not in detail.lower(), f"Event leaked 'uploads': {detail}"
        assert "\\" not in detail, f"Event leaked backslash: {detail}"
        assert not detail.startswith("Loaded /"), f"Event leaked absolute POSIX path: {detail}"
        assert ":" not in detail, f"Event leaked drive/absolute path: {detail}"
        assert str(tmp_path) not in detail, f"Event leaked tmp_path: {detail}"


def test_gps_upload_single_and_batch_storage_and_validation(client_unauth, monkeypatch):
    """Task 8: single + batch with gps_lat=14.42, gps_lng=120.94 -> stored; empty strings -> NULL, no 422; lat=999 -> 422."""
    import io
    import pytest
    from PIL import Image
    from backend.database import get_connection

    monkeypatch.setattr("backend.routes.analysis.enqueue_batch", lambda batch_id: None)
    monkeypatch.setattr("backend.routes.batches.enqueue_batch", lambda batch_id: None)

    def make_png_bytes() -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(120, 140, 160)).save(buf, format="PNG")
        return buf.getvalue()

    img_bytes = make_png_bytes()
    csrf_headers = {"X-Requested-With": "XMLHttpRequest"}

    # 1. Single upload with gps_lat=14.42, gps_lng=120.94 -> stored in DB
    res_single = client_unauth.post(
        "/api/analysis/upload",
        data={
            "company": "Meralco",
            "barangay": "San Antonio",
            "street": "Pearl Drive",
            "gps_lat": "14.42",
            "gps_lng": "120.94",
        },
        files={"file": ("single_gps.png", img_bytes, "image/png")},
        headers=csrf_headers,
    )
    assert res_single.status_code == 200, res_single.text
    single_id = res_single.json()["analysis_id"]

    conn = get_connection()
    row_single = conn.execute("SELECT gps_lat, gps_lng, gps_source FROM analyses WHERE id = ?", (single_id,)).fetchone()
    conn.close()
    assert row_single["gps_lat"] == pytest.approx(14.42)
    assert row_single["gps_lng"] == pytest.approx(120.94)
    assert row_single["gps_source"] == "manual"

    # 2. Batch upload with gps_lat=14.42, gps_lng=120.94 -> stored in DB for all batch items
    res_batch = client_unauth.post(
        "/api/batches",
        data={
            "company": "Meralco",
            "barangay": "San Antonio",
            "street": "Pearl Drive",
            "gps_lat": "14.42",
            "gps_lng": "120.94",
        },
        files=[
            ("files", ("batch_gps_1.png", img_bytes, "image/png")),
            ("files", ("batch_gps_2.png", img_bytes, "image/png")),
        ],
        headers=csrf_headers,
    )
    assert res_batch.status_code == 200, res_batch.text
    batch_id = res_batch.json()["batch_id"]

    conn = get_connection()
    batch_rows = conn.execute("SELECT gps_lat, gps_lng, gps_source FROM analyses WHERE batch_id = ?", (batch_id,)).fetchall()
    conn.close()
    assert len(batch_rows) == 2
    for r in batch_rows:
        assert r["gps_lat"] == pytest.approx(14.42)
        assert r["gps_lng"] == pytest.approx(120.94)
        assert r["gps_source"] == "manual"

    # 3. Empty strings -> NULL in DB, no 422 (single + batch)
    res_empty_single = client_unauth.post(
        "/api/analysis/upload",
        data={
            "company": "Meralco",
            "barangay": "San Antonio",
            "street": "Pearl Drive",
            "gps_lat": "",
            "gps_lng": "",
        },
        files={"file": ("empty_gps.png", img_bytes, "image/png")},
        headers=csrf_headers,
    )
    assert res_empty_single.status_code == 200, res_empty_single.text
    empty_single_id = res_empty_single.json()["analysis_id"]

    res_empty_batch = client_unauth.post(
        "/api/batches",
        data={
            "company": "Meralco",
            "barangay": "San Antonio",
            "street": "Pearl Drive",
            "gps_lat": "   ",
            "gps_lng": "",
        },
        files=[("files", ("empty_batch.png", img_bytes, "image/png"))],
        headers=csrf_headers,
    )
    assert res_empty_batch.status_code == 200, res_empty_batch.text
    empty_batch_id = res_empty_batch.json()["batch_id"]

    conn = get_connection()
    row_empty_s = conn.execute("SELECT gps_lat, gps_lng FROM analyses WHERE id = ?", (empty_single_id,)).fetchone()
    row_empty_b = conn.execute("SELECT gps_lat, gps_lng FROM analyses WHERE batch_id = ?", (empty_batch_id,)).fetchone()
    conn.close()
    assert row_empty_s["gps_lat"] is None and row_empty_s["gps_lng"] is None
    assert row_empty_b["gps_lat"] is None and row_empty_b["gps_lng"] is None

    # 4. Out-of-range lat=999 -> 422 (single + batch)
    res_bad_single = client_unauth.post(
        "/api/analysis/upload",
        data={
            "company": "Meralco",
            "barangay": "San Antonio",
            "street": "Pearl Drive",
            "gps_lat": "999",
            "gps_lng": "120.94",
        },
        files={"file": ("bad_gps.png", img_bytes, "image/png")},
        headers=csrf_headers,
    )
    assert res_bad_single.status_code == 422

    res_bad_batch = client_unauth.post(
        "/api/batches",
        data={
            "company": "Meralco",
            "barangay": "San Antonio",
            "street": "Pearl Drive",
            "gps_lat": "999",
            "gps_lng": "120.94",
        },
        files=[("files", ("bad_batch.png", img_bytes, "image/png"))],
        headers=csrf_headers,
    )
    assert res_bad_batch.status_code == 422


def test_logout_revokes_server_session_in_db(_init_settings):
    from fastapi.testclient import TestClient
    from main import app
    from backend.auth import hash_password

    h, s = hash_password("LogoutTest123!")
    _init_settings.execute(
        "INSERT INTO users (username, password_hash, salt, role, active) VALUES (?, ?, ?, 'admin', 1)",
        ("logout_test_user", h, s),
    )
    _init_settings.commit()

    client = TestClient(app)
    csrf_headers = {"X-Requested-With": "XMLHttpRequest"}
    login_res = client.post(
        "/api/auth/login",
        json={"username": "logout_test_user", "password": "LogoutTest123!"},
        headers=csrf_headers,
    )
    assert login_res.status_code == 200
    session_token = client.cookies.get("session")
    assert session_token

    me_before = client.get("/api/auth/me")
    assert me_before.status_code == 200

    logout_res = client.post("/api/auth/logout", headers=csrf_headers)
    assert logout_res.status_code == 200

    # Replaying the captured session cookie after logout must be rejected (401)
    replay_client = TestClient(app, cookies={"session": session_token})
    me_after = replay_client.get("/api/auth/me")
    assert me_after.status_code == 401


def test_batch_upload_persists_notes_and_cleans_up_partial_files_on_failure(client_unauth, monkeypatch, tmp_path):
    import io
    from PIL import Image
    from backend.database import get_connection
    import backend.routes.batches as batches_route

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(batches_route, "enqueue_batch", lambda *args, **kwargs: None)

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(80, 120, 160)).save(buf, format="PNG")
    valid_png = buf.getvalue()
    csrf_headers = {"X-Requested-With": "XMLHttpRequest"}

    # 1. Batch upload with notes stores notes on each created analysis row
    res_ok = client_unauth.post(
        "/api/batches",
        data={
            "company": "Meralco",
            "barangay": "San Antonio",
            "street": "Pearl Drive",
            "notes": "Batch inspection note #42",
        },
        files=[
            ("files", ("batch_note_1.png", valid_png, "image/png")),
            ("files", ("batch_note_2.png", valid_png, "image/png")),
        ],
        headers=csrf_headers,
    )
    assert res_ok.status_code == 200, res_ok.text
    batch_id = res_ok.json()["batch_id"]

    conn = get_connection()
    rows = conn.execute("SELECT notes FROM analyses WHERE batch_id = ?", (batch_id,)).fetchall()
    conn.close()
    assert len(rows) == 2
    assert all(r["notes"] == "Batch inspection note #42" for r in rows)

    # 2. If second file in batch fails validation, first file + thumbnail are cleaned up
    uploads_dir = tmp_path / "uploads"
    before_files = set(uploads_dir.iterdir()) if uploads_dir.exists() else set()

    res_fail = client_unauth.post(
        "/api/batches",
        data={
            "company": "Meralco",
            "barangay": "San Antonio",
            "street": "Pearl Drive",
        },
        files=[
            ("files", ("good_first.png", valid_png, "image/png")),
            ("files", ("corrupt_second.png", b"not-an-image", "image/png")),
        ],
        headers=csrf_headers,
    )
    assert res_fail.status_code == 400
    after_files = set(uploads_dir.iterdir()) if uploads_dir.exists() else set()
    assert after_files == before_files


def test_calibrate_uses_analysis_band_and_clearance_status(client_admin, _init_settings):
    import json
    from backend.utils.clearance_helpers import determine_overall_safety

    # 1. Pure helper check: High analysis_band escalates Safe wire to Unsafe
    safety = determine_overall_safety(
        ["Safe"],
        max_wire_band="Low",
        analysis_band="High",
        clearance_status="above",
    )
    assert safety == "Unsafe"

    # 2. Calibrate endpoint respects analysis.risk_score band (High risk_score -> at least Unsafe)
    cur = _init_settings.execute(
        """INSERT INTO analyses (
            filename, status, company, barangay, street, is_public, is_mock,
            width, height, pole_bbox_json, assumed_pole_height_m,
            risk_score, max_crs, clearance_available, clearance_ok
        ) VALUES (?, 'complete', 'Meralco', 'Brgy 1', 'Main St', 0, 1, 800, 600, ?, 10.0, 72.0, 20.0, 7.5, 1)""",
        ("calib_parity.jpg", json.dumps([100, 100, 150, 550])),
    )
    aid = cur.lastrowid
    _init_settings.execute(
        """INSERT INTO wires (
            analysis_id, class_name, confidence, sag_px, span_px, crs, band,
            clearance_m, clearance_status, safety_level, centerline_json, params_json
        ) VALUES (?, 'communication_cable', 0.9, 5.0, 200.0, 20.0, 'Low', 7.5, 'ok', 'Safe', ?, '{}')""",
        (aid, json.dumps([[100, 200], [300, 200]])),
    )
    _init_settings.commit()

    res_cal = client_admin.post(
        f"/api/analysis/{aid}/calibrate",
        json={"points": [{"x": 100, "y": 100}, {"x": 100, "y": 550}], "real_length_m": 9.0},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert res_cal.status_code == 200, res_cal.text
    assert res_cal.json()["safety_level"] == "Unsafe"


def test_geojson_export_skips_and_counts_no_gps_records(client_admin, _init_settings):
    cur1 = _init_settings.execute(
        """INSERT INTO analyses (filename, status, company, barangay, street, is_public, is_mock, gps_lat, gps_lng)
           VALUES ('with_gps.jpg', 'complete', 'Meralco', 'B1', 'S1', 0, 0, 14.55, 121.02)"""
    )
    id_with_gps = cur1.lastrowid

    cur2 = _init_settings.execute(
        """INSERT INTO analyses (filename, status, company, barangay, street, is_public, is_mock, gps_lat, gps_lng)
           VALUES ('no_gps.jpg', 'complete', 'Meralco', 'B1', 'S1', 0, 0, NULL, NULL)"""
    )
    id_no_gps = cur2.lastrowid
    _init_settings.commit()

    res = client_admin.post(
        "/api/reports/geojson",
        json={"analysis_ids": [id_with_gps, id_no_gps], "include_mock": False, "include_invalid": False},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert len(body["features"]) == 1
    assert body["features"][0]["properties"]["id"] == id_with_gps
    assert body["skipped_no_gps"] == 1
    assert body["skipped_invalid"] == 0
    assert res.headers.get("X-Skipped-No-Gps") == "1"
    assert res.headers.get("X-Skipped-Invalid") == "0"
    assert res.headers.get("X-Skipped-Count") == "1"





