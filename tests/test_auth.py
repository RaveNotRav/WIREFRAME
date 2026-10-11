from backend.auth import hash_password, verify_password
from backend.database import get_connection


def test_password_hashing():
    # Should use configurable iterations from test env
    phash, salt = hash_password("mypassword")
    assert verify_password("mypassword", salt, phash)
    assert not verify_password("wrongpassword", salt, phash)

def test_login_success(client_unauth, _init_settings):
    # Seed user
    phash, salt = hash_password("securepassword")
    conn = get_connection()
    conn.execute("INSERT INTO users (username, password_hash, salt, role) VALUES ('testuser', ?, ?, 'field_inspector')", (phash, salt))
    conn.commit()
    conn.close()

    res = client_unauth.post("/api/auth/login", json={"username": "testuser", "password": "securepassword"}, headers={"X-Requested-With": "XMLHttpRequest"})
    assert res.status_code == 200
    assert "session=" in res.headers["set-cookie"]
    assert "HttpOnly" in res.headers["set-cookie"]

def test_login_failure(client_unauth, _init_settings):
    # Missing user dummy hash logic should take roughly similar time and return generic error
    res = client_unauth.post("/api/auth/login", json={"username": "nonexistent", "password": "securepassword"}, headers={"X-Requested-With": "XMLHttpRequest"})
    assert res.status_code == 401
    assert res.json()["detail"] == "Invalid username or password"

def test_csrf_enforcement(client_unauth):
    # POST without CSRF header should fail
    res = client_unauth.post("/api/auth/login", json={"username": "testuser", "password": "securepassword"})
    assert res.status_code == 403

def test_rate_limiting(client_unauth, _init_settings):
    # login limit is 5
    for _ in range(5):
        res = client_unauth.post("/api/auth/login", json={"username": "someuser", "password": "pw"}, headers={"X-Requested-With": "XMLHttpRequest"})
        assert res.status_code == 401
        
    res = client_unauth.post("/api/auth/login", json={"username": "someuser", "password": "pw"}, headers={"X-Requested-With": "XMLHttpRequest"})
    assert res.status_code == 429
