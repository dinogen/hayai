import hmac

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from api.auth import SESSION_KEY, is_authenticated
from app.config import settings

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/auth/login")
def login(payload: LoginRequest, request: Request):
    user_ok = hmac.compare_digest(payload.username, settings.AUTH_USERNAME)
    pass_ok = hmac.compare_digest(payload.password, settings.AUTH_PASSWORD)
    if not (user_ok and pass_ok):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    request.session[SESSION_KEY] = True
    return {"authenticated": True}


@router.post("/auth/logout")
def logout(request: Request):
    request.session.clear()
    return {"authenticated": False}


@router.get("/auth/me")
def session_status(request: Request):
    return {"authenticated": is_authenticated(request)}
