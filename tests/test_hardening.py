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
