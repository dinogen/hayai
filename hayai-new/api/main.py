from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from api.auth import SESSION_COOKIE_NAME, require_auth
from api.routers import auth, portfolios, config, holdings, markets, instruments
from app.config import settings
from app.db import execute_query

app = FastAPI(
    title="HAYAI v2 API",
    description="Personal Quant & AI Decision Support System API (Read-Only, with manual portfolio management)",
    version="2.0.0"
)

# Signed cookie session: single user authenticates via POST /api/auth/login.
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.AUTH_SESSION_SECRET,
    session_cookie=SESSION_COOKIE_NAME,
    max_age=settings.AUTH_SESSION_MAX_AGE,
    same_site="lax",
    https_only=False,
)

# Credentialed requests (cookies) require explicit origins, not "*".
DEV_ORIGINS = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Public: authentication endpoints only.
app.include_router(auth.router, prefix="/api", tags=["Authentication"])

# All business endpoints require an authenticated session.
app.include_router(portfolios.router, prefix="/api", tags=["Portfolios"], dependencies=[Depends(require_auth)])
app.include_router(config.router, prefix="/api", tags=["Configuration"], dependencies=[Depends(require_auth)])
app.include_router(holdings.router, prefix="/api", tags=["Holdings"], dependencies=[Depends(require_auth)])
app.include_router(markets.router, prefix="/api", tags=["Markets"], dependencies=[Depends(require_auth)])
app.include_router(instruments.router, prefix="/api", tags=["Instruments"], dependencies=[Depends(require_auth)])

@app.get("/api/health")
def health_check():
    try:
        # Check DB connection & last successful job run
        last_jobs = execute_query("SELECT job_name, status, finished_at FROM job_run ORDER BY id DESC LIMIT 5")
        return {
            "status": "healthy",
            "database": "connected",
            "recent_jobs": last_jobs
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": str(e)
        }
