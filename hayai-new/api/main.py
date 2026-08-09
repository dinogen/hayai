from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db import execute_query
from api.routers import portfolios, config, holdings

app = FastAPI(
    title="HAYAI v2 API",
    description="Personal Quant & AI Decision Support System API (Read-Only, with manual portfolio management)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(portfolios.router, prefix="/api", tags=["Portfolios"])
app.include_router(config.router, prefix="/api", tags=["Configuration"])
app.include_router(holdings.router, prefix="/api", tags=["Holdings"])

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
