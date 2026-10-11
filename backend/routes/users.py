from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth import hash_password, require_role, verify_csrf
from backend.database import get_connection

router = APIRouter(prefix="/api/users", tags=["users"])

class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str

@router.post("", dependencies=[Depends(verify_csrf), Depends(require_role(["admin"]))])
async def create_user(payload: CreateUserRequest):
    if len(payload.password) < 10:
        raise HTTPException(status_code=400, detail="Password must be at least 10 characters long")
    if payload.password == payload.username:
        raise HTTPException(status_code=400, detail="Password cannot be the same as username")
        
    common_passwords = ["password123", "qwertyuiop", "1234567890", "password1234"]
    if payload.password in common_passwords:
        raise HTTPException(status_code=400, detail="Password is too common")
        
    if payload.role not in ["admin", "lgu_engineer", "field_inspector"]:
        raise HTTPException(status_code=400, detail="Invalid role")
        
    phash, salt = hash_password(payload.password)
    
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
            (payload.username, phash, salt, payload.role)
        )
        conn.commit()
        return {"status": "success", "username": payload.username}
    except Exception as e:
        if "UNIQUE" in str(e):
            raise HTTPException(status_code=400, detail="Username already exists")
        raise
    finally:
        conn.close()

# Note: Deactivation logic would also go here.
