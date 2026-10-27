"""Tests for scripts/seed_demo_admin.py.

Verifies:
- Refuses to run without WIREFRAME_MOCK=1
- Creates exactly one admin on an empty DB
- No-op when users exist without --add
- --add creates demo_admin_2
- Password is never present in captured stdout or stderr
- Created user can log in through the API using the password read from the file
- File is written only to tmp_path during tests, never touching the repo root
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.seed_demo_admin import seed_demo_admin


def test_seed_refuses_without_mock_mode(tmp_path, monkeypatch):
    """TASK 3: Refuses to run (exit code 1) unless WIREFRAME_MOCK == '1'."""
    out_file = tmp_path / "passwords_nomock.txt"
    db_file = tmp_path / "nomock.db"

    # 1. Direct function call test with WIREFRAME_MOCK="0"
    monkeypatch.setenv("WIREFRAME_MOCK", "0")
    monkeypatch.setenv("WIREFRAME_DB_PATH", str(db_file))
    code = seed_demo_admin(output_file=str(out_file))
    assert code == 1
    assert not out_file.exists()

    # 2. Subprocess test with WIREFRAME_MOCK unset
    env = os.environ.copy()
    env.pop("WIREFRAME_MOCK", None)
    env["WIREFRAME_DB_PATH"] = str(db_file)

    res = subprocess.run(
        [sys.executable, "scripts/seed_demo_admin.py", "--output-file", str(out_file)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 1
    assert "ERROR" in res.stderr
    assert "WIREFRAME_MOCK='1'" in res.stderr
    assert not out_file.exists()


def test_seed_lifecycle_and_stdout_hygiene(tmp_path):
    """TASK 3: Creates 1 admin on empty DB, no-op when users exist, --add creates demo_admin_2, password not in stdout."""
    db_file = tmp_path / "lifecycle.db"
    out_file = tmp_path / "creds_lifecycle.txt"

    env = os.environ.copy()
    env["WIREFRAME_MOCK"] = "1"
    env["WIREFRAME_DB_PATH"] = str(db_file)

    # 1. Empty DB: creates exactly one admin 'demo_admin'
    res1 = subprocess.run(
        [sys.executable, "scripts/seed_demo_admin.py", "--output-file", str(out_file)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res1.returncode == 0
    assert "Created admin 'demo_admin'" in res1.stdout
    assert out_file.name in res1.stdout

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    users1 = conn.execute("SELECT * FROM users").fetchall()
    assert len(users1) == 1
    assert users1[0]["username"] == "demo_admin"
    assert users1[0]["role"] == "admin"
    conn.close()

    assert out_file.exists()
    content1 = out_file.read_text(encoding="utf-8")
    assert "username: demo_admin" in content1
    pw1 = [line.split(": ", 1)[1] for line in content1.splitlines() if line.startswith("password: ")][0]
    assert len(pw1) >= 16

    # Password must NEVER appear in captured stdout or stderr
    assert pw1 not in res1.stdout
    assert pw1 not in res1.stderr

    # 2. Re-run without --add: no-op when users exist
    res2 = subprocess.run(
        [sys.executable, "scripts/seed_demo_admin.py", "--output-file", str(out_file)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res2.returncode == 0
    assert "already exist" in res2.stdout
    assert "doing nothing" in res2.stdout

    conn = sqlite3.connect(str(db_file))
    users2 = conn.execute("SELECT * FROM users").fetchall()
    assert len(users2) == 1
    conn.close()
    assert pw1 not in res2.stdout
    assert pw1 not in res2.stderr

    # 3. Re-run with --add: creates demo_admin_2
    res3 = subprocess.run(
        [sys.executable, "scripts/seed_demo_admin.py", "--add", "--output-file", str(out_file)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res3.returncode == 0
    assert "Created admin 'demo_admin_2'" in res3.stdout
    assert out_file.name in res3.stdout

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    users3 = conn.execute("SELECT * FROM users ORDER BY id ASC").fetchall()
    assert len(users3) == 2
    assert users3[0]["username"] == "demo_admin"
    assert users3[1]["username"] == "demo_admin_2"
    assert users3[1]["role"] == "admin"
    conn.close()

    content3 = out_file.read_text(encoding="utf-8")
    assert "username: demo_admin_2" in content3
    pws = [line.split(": ", 1)[1] for line in content3.splitlines() if line.startswith("password: ")]
    assert len(pws) == 2
    pw2 = pws[1]
    assert len(pw2) >= 16
    assert pw2 != pw1

    # Neither password appears in stdout or stderr
    assert pw2 not in res3.stdout
    assert pw2 not in res3.stderr
    assert pw1 not in res3.stdout


def test_created_user_can_login_through_api(tmp_path, client_unauth, monkeypatch):
    """TASK 3: Created admin can log in through the API using the password read from file."""
    out_file = tmp_path / "login_creds.txt"

    # Use environment override to ensure no files are created in the repo root
    monkeypatch.setenv("WIREFRAME_MOCK", "1")
    monkeypatch.setenv("WIREFRAME_DEMO_ADMIN_FILE", str(out_file))

    code = seed_demo_admin()
    assert code == 0
    assert out_file.exists()

    content = out_file.read_text(encoding="utf-8")
    lines = dict(line.split(": ", 1) for line in content.strip().splitlines() if ": " in line)
    username = lines["username"]
    password = lines["password"]
    assert username == "demo_admin"

    # Log in through API
    login_res = client_unauth.post(
        "/api/auth/login",
        json={"username": username, "password": password},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert login_res.status_code == 200
    assert "session" in login_res.cookies

    # Authenticated request to /api/auth/me to verify admin role
    client_unauth.cookies.set("session", login_res.cookies["session"])
    me_res = client_unauth.get("/api/auth/me")
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["username"] == "demo_admin"
    assert me_data["role"] == "admin"


def test_seed_reset_absent_creates_admin(tmp_path):
    """TASK 4: --reset creates demo_admin when absent, never leaks password to stdout/stderr."""
    db_file = tmp_path / "reset_absent.db"
    out_file = tmp_path / "creds_absent.txt"

    env = os.environ.copy()
    env["WIREFRAME_MOCK"] = "1"
    env["WIREFRAME_DB_PATH"] = str(db_file)

    res = subprocess.run(
        [sys.executable, "scripts/seed_demo_admin.py", "--reset", "--output-file", str(out_file)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0
    assert "demo_admin reset; credentials written to" in res.stdout
    assert out_file.name in res.stdout

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    user = conn.execute("SELECT * FROM users WHERE username = 'demo_admin'").fetchone()
    assert user is not None
    assert user["role"] == "admin"
    assert user["active"] == 1
    salt = user["salt"]
    phash = user["password_hash"]
    conn.close()

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "username: demo_admin" in content
    pws = [line.split(": ", 1)[1] for line in content.splitlines() if line.startswith("password: ")]
    assert len(pws) == 1
    pw = pws[0]
    assert len(pw) >= 16

    from backend.auth import verify_password
    assert verify_password(pw, salt, phash)

    # Password and hash must never appear in stdout or stderr
    assert pw not in res.stdout
    assert pw not in res.stderr
    assert phash not in res.stdout


def test_seed_reset_existing_changes_hash_and_verifies(tmp_path):
    """TASK 4: --reset on existing demo_admin updates hash, writes valid password, never prints password."""
    db_file = tmp_path / "reset_existing.db"
    out_file = tmp_path / "creds_existing.txt"

    env = os.environ.copy()
    env["WIREFRAME_MOCK"] = "1"
    env["WIREFRAME_DB_PATH"] = str(db_file)

    # 1. First create demo_admin using default behavior
    res1 = subprocess.run(
        [sys.executable, "scripts/seed_demo_admin.py", "--output-file", str(out_file)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res1.returncode == 0
    content1 = out_file.read_text(encoding="utf-8")
    pw1 = [line.split(": ", 1)[1] for line in content1.splitlines() if line.startswith("password: ")][0]

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    user1 = conn.execute("SELECT * FROM users WHERE username = 'demo_admin'").fetchone()
    old_hash = user1["password_hash"]
    old_salt = user1["salt"]
    conn.close()

    from backend.auth import verify_password
    assert verify_password(pw1, old_salt, old_hash)

    # 2. Reset existing demo_admin with --reset
    res2 = subprocess.run(
        [sys.executable, "scripts/seed_demo_admin.py", "--reset", "--output-file", str(out_file)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res2.returncode == 0
    assert "demo_admin reset; credentials written to" in res2.stdout
    assert out_file.name in res2.stdout

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    user2 = conn.execute("SELECT * FROM users WHERE username = 'demo_admin'").fetchone()
    new_hash = user2["password_hash"]
    new_salt = user2["salt"]
    assert user2["active"] == 1
    assert user2["role"] == "admin"
    assert new_hash != old_hash
    conn.close()

    content2 = out_file.read_text(encoding="utf-8")
    assert "username: demo_admin" in content2
    pws2 = [line.split(": ", 1)[1] for line in content2.splitlines() if line.startswith("password: ")]
    assert len(pws2) == 1
    pw2 = pws2[0]
    assert len(pw2) >= 16
    assert pw2 != pw1

    assert verify_password(pw2, new_salt, new_hash)
    assert not verify_password(pw1, new_salt, new_hash)

    # Neither password nor hash appears in output
    assert pw2 not in res2.stdout
    assert pw2 not in res2.stderr
    assert new_hash not in res2.stdout


def test_reset_user_can_login_through_api(tmp_path, client_unauth, monkeypatch):
    """TASK 4: Admin reset via seed_demo_admin(reset=True) can log in through API."""
    out_file = tmp_path / "reset_login_creds.txt"

    monkeypatch.setenv("WIREFRAME_MOCK", "1")
    monkeypatch.setenv("WIREFRAME_DEMO_ADMIN_FILE", str(out_file))

    # Initial seed
    code1 = seed_demo_admin()
    assert code1 == 0

    # Reset
    code2 = seed_demo_admin(reset=True)
    assert code2 == 0

    content = out_file.read_text(encoding="utf-8")
    lines = dict(line.split(": ", 1) for line in content.strip().splitlines() if ": " in line)
    username = lines["username"]
    password = lines["password"]
    assert username == "demo_admin"

    # API login
    login_res = client_unauth.post(
        "/api/auth/login",
        json={"username": username, "password": password},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert login_res.status_code == 200
    assert "session" in login_res.cookies

    # Verify session
    client_unauth.cookies.set("session", login_res.cookies["session"])
    me_res = client_unauth.get("/api/auth/me")
    assert me_res.status_code == 200
    assert me_res.json()["username"] == "demo_admin"


def test_ensure_admin_creates_when_absent_and_noop_when_present(tmp_path):
    """--ensure creates demo_admin when absent, no-ops when present, never prints password."""
    db_file = tmp_path / "ensure.db"
    out_file = tmp_path / "creds_ensure.txt"

    env = os.environ.copy()
    env["WIREFRAME_MOCK"] = "1"
    env["WIREFRAME_DB_PATH"] = str(db_file)

    # 1. First run: demo_admin absent -> creates it
    res1 = subprocess.run(
        [sys.executable, "scripts/seed_demo_admin.py", "--ensure", "--output-file", str(out_file)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res1.returncode == 0
    assert "Created admin 'demo_admin'" in res1.stdout
    assert out_file.exists()
    content1 = out_file.read_text(encoding="utf-8")
    pw1 = [line.split(": ", 1)[1] for line in content1.splitlines() if line.startswith("password: ")][0]
    assert len(pw1) >= 16
    assert pw1 not in res1.stdout
    assert pw1 not in res1.stderr

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    user1 = conn.execute("SELECT * FROM users WHERE username = 'demo_admin'").fetchone()
    assert user1 is not None
    assert user1["role"] == "admin"
    hash1 = user1["password_hash"]
    conn.close()

    # 2. Second run: demo_admin exists -> no-op, prints guidance, does NOT rotate
    res2 = subprocess.run(
        [sys.executable, "scripts/seed_demo_admin.py", "--ensure", "--output-file", str(out_file)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res2.returncode == 0
    assert "demo_admin exists; use scripts\\seed_demo_admin.py --reset to rotate password" in res2.stdout
    assert pw1 not in res2.stdout
    assert pw1 not in res2.stderr

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    user2 = conn.execute("SELECT * FROM users WHERE username = 'demo_admin'").fetchone()
    assert user2["password_hash"] == hash1, "Password hash must not change on ensure"
    conn.close()


