"""Authentication helpers for the HAYAI v2 API (cookie-based session).

A single user (username/password from .env) authenticates via
`POST /api/auth/login`, which sets a signed session cookie managed by
Starlette's SessionMiddleware. The `require_auth` dependency rejects any
request without a valid session.
"""
from fastapi import HTTPException, Request

from app.config import settings

SESSION_KEY = "authenticated"
SESSION_COOKIE_NAME = "hayai_session"

if not (settings.AUTH_USERNAME and settings.AUTH_PASSWORD and settings.AUTH_SESSION_SECRET):
    raise RuntimeError(
        "AUTH_USERNAME, AUTH_PASSWORD and AUTH_SESSION_SECRET must be set in .env "
        "to enable the cookie-based session authentication."
    )


def is_authenticated(request: Request) -> bool:
    return bool(request.session.get(SESSION_KEY))


def require_auth(request: Request) -> None:
    """FastAPI dependency: reject the request unless an authenticated session exists."""
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
