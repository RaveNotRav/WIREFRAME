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


