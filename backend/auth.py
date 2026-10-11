import hashlib
import hmac
import os
import secrets
import threading
import time
from datetime import datetime, timedelta

from fastapi import Cookie, Depends, HTTPException, Request
from pydantic import BaseModel

from backend.database import get_connection

# PBKDF2 Iterations - tests can lower this if WIREFRAME_ENV=test
PBKDF2_ITERATIONS = 1000 if os.environ.get("WIREFRAME_ENV") == "test" else 600000

class User(BaseModel):
    id: int
    username: str
    role: str
    active: bool

def hash_password(password: str, salt: bytes = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_bytes(16)
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        PBKDF2_ITERATIONS
    )
    return hashed.hex(), salt.hex()

def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(hashed.hex(), hash_hex)

def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()

def create_session(conn, user_id: int, expiry_hours: int = 8) -> str:
    token = secrets.token_urlsafe(32)
    token_hash = _hash_token(token)
    
    expires_at = (datetime.utcnow() + timedelta(hours=expiry_hours)).isoformat()
    
    conn.execute(
        "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
        (token_hash, user_id, expires_at)
    )
    conn.commit()
    return token

def get_current_user(session: str | None = Cookie(None)) -> User:
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    token_hash = _hash_token(session)
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT u.id, u.username, u.role, u.active, s.expires_at 
               FROM sessions s 
               JOIN users u ON s.user_id = u.id 
               WHERE s.token_hash = ?""",
            (token_hash,)
        ).fetchone()
        
        if not row:
            raise HTTPException(status_code=401, detail="Invalid session")
            
        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.utcnow() > expires_at:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
            conn.commit()
            raise HTTPException(status_code=401, detail="Session expired")
            
        if not row["active"]:
            raise HTTPException(status_code=401, detail="User is inactive")
            
        return User(id=row["id"], username=row["username"], role=row["role"], active=bool(row["active"]))
    finally:
        conn.close()


def get_optional_user(session: str | None = Cookie(None)) -> User | None:
    if not session:
        return None
    token_hash = _hash_token(session)
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT u.id, u.username, u.role, u.active, s.expires_at "
            "FROM sessions s JOIN users u ON s.user_id = u.id "
            "WHERE s.token_hash = ?",
            (token_hash,)
        ).fetchone()
        
        if not row:
            return None
        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.utcnow() > expires_at:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
            conn.commit()
            return None
        if not row["active"]:
            return None
        return User(id=row["id"], username=row["username"], role=row["role"], active=bool(row["active"]))
    finally:
        conn.close()

def verify_analysis_access(row: dict, user: User | None, request: Request):
    if dict(row).get("created_by") is not None:
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")
        if user.role == "field_inspector" and row["created_by"] != user.id:
            raise HTTPException(status_code=403, detail="Forbidden")
    else:
        # Public submission
        if user and user.role in ["lgu_engineer", "admin"]:
            return
            
        guest_token = request.cookies.get("guest_session")
        if not guest_token:
            raise HTTPException(status_code=403, detail="Forbidden - Missing guest token")
        if _hash_token(guest_token) != dict(row).get("guest_session_id"):
            raise HTTPException(status_code=403, detail="Forbidden - Invalid guest token")
    return True

def require_role(allowed_roles: list[str]):
    def role_checker(user: User = Depends(get_current_user)):
        if user.role not in allowed_roles and user.role != "admin":
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return role_checker

def verify_csrf(request: Request):
    if request.method == "GET":
        return
        
    # Check X-Requested-With or Origin
    requested_with = request.headers.get("X-Requested-With")
    if requested_with == "XMLHttpRequest":
        return
        
    origin = request.headers.get("origin")
    if origin:
        # Simple validation, a real app would check against allowed origins
        return
        
    raise HTTPException(status_code=403, detail="CSRF check failed")

# --- Rate Limiter ---

_rate_limit_lock = threading.Lock()
_rate_limits = {}

def check_rate_limit(request: Request, action: str, username: str = "", limit: int = 5, window: int = 900):
    ip = request.client.host
    if os.environ.get("TRUSTED_PROXY") == "1":
        ip = request.headers.get("X-Forwarded-For", ip).split(",")[0].strip()
        
    key = f"{action}:{ip}:{username}"
    now = time.time()
    
    with _rate_limit_lock:
        # Evict old entries periodically
        if len(_rate_limits) > 10000:
            for k in list(_rate_limits.keys()):
                _rate_limits[k] = [t for t in _rate_limits[k] if now - t < 3600]
                if not _rate_limits[k]:
                    del _rate_limits[k]
                    
        history = _rate_limits.get(key, [])
        history = [t for t in history if now - t < window]
        
        if len(history) >= limit:
            raise HTTPException(status_code=429, detail="Too many requests")
            
        history.append(now)
        _rate_limits[key] = history

def reset_rate_limits():
    with _rate_limit_lock:
        _rate_limits.clear()


def check_anon_rate_limit(request: Request, limit: int = 3, window: int = 900):
    ip = request.client.host
    if os.environ.get("TRUSTED_PROXY") == "1":
        ip = request.headers.get("X-Forwarded-For", ip).split(",")[0].strip()
    
    key = f"anon_upload:{ip}"
    now = time.time()
    
    with _rate_limit_lock:
        if len(_rate_limits) > 10000:
            for k in list(_rate_limits.keys()):
                _rate_limits[k] = [t for t in _rate_limits[k] if now - t < 3600]
                if not _rate_limits[k]:
                    del _rate_limits[k]
                    
        history = _rate_limits.get(key, [])
        history = [t for t in history if now - t < window]
        
        if len(history) >= limit:
            raise HTTPException(status_code=429, detail="You have reached the maximum number of public uploads (3 per 15 minutes). Please try again later or log in.")
            
        history.append(now)
        _rate_limits[key] = history
