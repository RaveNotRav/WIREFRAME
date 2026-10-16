from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from backend.auth import (
    User,
    check_rate_limit,
    clear_rate_limit,
    create_session,
    get_current_user,
    hash_password,
    verify_csrf,
    verify_password,
)
from backend.database import get_connection

router = APIRouter(prefix="/api/auth", tags=["auth"])

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login", dependencies=[Depends(verify_csrf)])
async def login(req: Request, res: Response, payload: LoginRequest):
    check_rate_limit(req, "login", payload.username, limit=5, window=900)
    
    conn = get_connection()
    try:
        user = conn.execute("SELECT * FROM users WHERE username = ?", (payload.username,)).fetchone()
        
        valid = False
        if user:
            valid = verify_password(payload.password, user["salt"], user["password_hash"])
        else:
            # Dummy hash to prevent timing attacks
            hash_password(payload.password)
            
        if not valid or not user or not user["active"]:
            raise HTTPException(status_code=401, detail="Invalid username or password")
            
        clear_rate_limit(req, "login", payload.username)

        # Clear existing expired sessions
        conn.execute("DELETE FROM sessions WHERE expires_at < datetime('now') OR user_id = ?", (user["id"],))
        
        token = create_session(conn, user["id"])
        
        # Set cookie
        res.delete_cookie("guest_session", path="/")
        res.set_cookie(
            key="session",
            value=token,
            httponly=True,
            samesite="lax",
            secure=(req.url.scheme == "https"),
            max_age=8 * 3600,
            path="/",
        )
        return {"status": "success"}
    finally:
        conn.close()

@router.post("/logout", dependencies=[Depends(verify_csrf)])
async def logout(res: Response, user: User = Depends(get_current_user)):
    # Simple logout: clear cookie. We could also explicitly delete the session from DB.
    # To delete from DB, we'd need the token. get_current_user could return it, but simple clearing works.
    res.delete_cookie("session", path="/")
    return {"status": "success"}

@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    return user
