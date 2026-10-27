#!/usr/bin/env python3
"""Seed a local admin user for mock-mode demos.

Only runs when WIREFRAME_MOCK=1.
Writes credentials to passwords_demo_admin.txt (or WIREFRAME_DEMO_ADMIN_FILE).
Never prints the generated password to stdout or logs.
"""

import argparse
import os
import secrets
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("WIREFRAME_DB_PATH", str(REPO_ROOT / "wireframe.db"))


def generate_admin_password(username: str) -> str:
    """Generate a random password meeting the policy in backend/routes/users.py."""
    common_passwords = {"password123", "qwertyuiop", "1234567890", "password1234"}
    while True:
        # token_urlsafe(24) produces ~32 characters, exceeding the 16-character minimum
        pw = secrets.token_urlsafe(24)
        if len(pw) >= 16 and pw != username and pw not in common_passwords:
            return pw


def choose_admin_username(conn) -> str:
    """Select 'demo_admin' or the next available 'demo_admin_N'."""
    username = "demo_admin"
    if conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
        n = 2
        while True:
            candidate = f"demo_admin_{n}"
            if not conn.execute("SELECT 1 FROM users WHERE username = ?", (candidate,)).fetchone():
                username = candidate
                break
            n += 1
    return username


def seed_demo_admin(
    add: bool = False,
    reset: bool = False,
    ensure: bool = False,
    output_file: str | None = None,
) -> int:
    """Seed or reset a demo admin user. Returns exit code (0 on success, 1 on refusal/error)."""
    if os.environ.get("WIREFRAME_MOCK") != "1":
        print(
            "ERROR: seed_demo_admin.py can only be run when WIREFRAME_MOCK='1'. Refusing to run.",
            file=sys.stderr,
        )
        return 1

    from backend.auth import hash_password
    from backend.database import get_connection, run_migrations

    conn = get_connection()
    try:
        run_migrations(conn)

        default_target = os.environ.get(
            "WIREFRAME_DEMO_ADMIN_FILE",
            str(REPO_ROOT / "passwords_demo_admin.txt"),
        )
        target_path = Path(output_file or default_target).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if ensure:
            row = conn.execute("SELECT id FROM users WHERE username = 'demo_admin'").fetchone()
            if row:
                print("demo_admin exists; use scripts\\seed_demo_admin.py --reset to rotate password")
                return 0

            username = "demo_admin"
            password = generate_admin_password(username)
            phash, salt = hash_password(password)
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, role, active) VALUES (?, ?, ?, 'admin', 1)",
                (username, phash, salt),
            )
            conn.commit()

            with open(target_path, "w", encoding="utf-8") as f:
                f.write(f"username: {username}\npassword: {password}\n")

            print(f"Created admin '{username}'. Credentials written to {target_path.name}")
            return 0

        if reset:
            username = "demo_admin"
            password = generate_admin_password(username)
            phash, salt = hash_password(password)

            row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE users SET password_hash = ?, salt = ?, role = 'admin', active = 1 WHERE username = ?",
                    (phash, salt, username),
                )
            else:
                conn.execute(
                    "INSERT INTO users (username, password_hash, salt, role, active) VALUES (?, ?, ?, 'admin', 1)",
                    (username, phash, salt),
                )
            conn.commit()

            with open(target_path, "w", encoding="utf-8") as f:
                f.write(f"username: {username}\npassword: {password}\n")

            print(f"demo_admin reset; credentials written to {target_path.name}")
            return 0

        row = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
        user_count = row["c"] if row else 0

        if user_count > 0 and not add:
            print("User(s) already exist in database; doing nothing. Use --add to create another admin.")
            return 0

        username = choose_admin_username(conn)
        password = generate_admin_password(username)
        phash, salt = hash_password(password)

        conn.execute(
            "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, 'admin')",
            (username, phash, salt),
        )
        conn.commit()

        mode = "a" if add and target_path.exists() else "w"
        with open(target_path, mode, encoding="utf-8") as f:
            if mode == "a":
                f.write("\n")
            f.write(f"username: {username}\npassword: {password}\n")

        print(f"Created admin '{username}'. Credentials written to {target_path.name}")
        return 0
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed demo admin user (mock mode only).")
    parser.add_argument(
        "--add",
        action="store_true",
        help="Add another admin even if users already exist in the database.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset demo_admin password (or create if absent).",
    )
    parser.add_argument(
        "--ensure",
        action="store_true",
        help="Ensure demo_admin exists (creates if absent, no-op if present).",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Custom output file path for credentials (default: passwords_demo_admin.txt).",
    )
    args = parser.parse_args()
    return seed_demo_admin(add=args.add, reset=args.reset, ensure=args.ensure, output_file=args.output_file)


if __name__ == "__main__":
    sys.exit(main())
