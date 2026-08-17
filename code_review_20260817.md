# Code Review Report - HAYAI v2
**Date:** August 17, 2026  
**Scope:** FastAPI Backend & API Architecture (`hayai-new/api/`, `hayai-new/app/`)

---

## 🔴 Critical Issues

1. **Database Connection Context Manager Transaction Rollback & Autocommit Risk**
   - **File / Line Reference:** `hayai-new/app/db.py`, lines 5–24 (`get_db_connection`)
   - **Explanation:** The database connection is created with `autocommit=False`. While `commit()` is called upon normal execution and `rollback()` on exception, read-only queries (e.g. `SELECT`) that execute inside `execute_query` leave transactions open until closed, which can lock resources or hold transaction locks in InnoDB depending on isolation levels if not explicitly committed or managed. Furthermore, connections returned to the connection pool/closed rely on the context manager; however, `pymysql.connect` does not inherently pool connections unless wrapped in a pool. Each request opens a brand-new TCP socket and connection to MariaDB, introducing severe latency and overhead under high request volume.
   - **Suggested Solution:**
     ```python
     connection = pymysql.connect(
         host=settings.DB_HOST,
         port=settings.DB_PORT,
         user=settings.DB_USER,
         password=settings.DB_PASSWORD,
         database=settings.DB_NAME,
         charset='utf8mb4',
         cursorclass=pymysql.cursors.DictCursor,
         autocommit=True  # For safety in read-heavy quant APIs, or manage explicit transactions for writes.
     )
     ```
   - **Rationale:** Ensures optimal connection lifecycle management, avoids hanging transaction locks, and prevents resource exhaustion under batch and API load.

2. **Hardcoded / Missing Validation on User Session Secret & Environment Fallbacks**
   - **File / Line Reference:** `hayai-new/api/auth.py`, lines 15–19; `hayai-new/app/config.py`
   - **Explanation:** While `auth.py` raises a `RuntimeError` if `AUTH_USERNAME`, `AUTH_PASSWORD`, or `AUTH_SESSION_SECRET` are missing, if `AUTH_SESSION_SECRET` is left as a weak default or placeholder in `.env.example`, Starlette's `SessionMiddleware` signed cookies can be forged or compromised.
   - **Suggested Solution:** Add strict length and entropy validation for `AUTH_SESSION_SECRET` upon startup:
     ```python
     if len(settings.AUTH_SESSION_SECRET) < 32:
         raise RuntimeError("AUTH_SESSION_SECRET must be at least 32 characters long for secure cookie signing.")
     ```
   - **Rationale:** Protects session integrity against cryptographic tampering and session hijacking.

---

## 🟡 Suggestions

1. **Explicit Pagination & Query Limits on News and Time-Series Data**
   - **File / Line Reference:** `hayai-new/api/routers/portfolios.py`, lines 172–208 (`get_portfolio_news`)
   - **Explanation:** The `get_portfolio_news` endpoint allows fetching news up to `days=14` with a default `limit=50`. However, if the volume of news articles per instrument is high, joining across `news`, `instrument`, `portfolio_instrument`, and `news_sentiment` without a composite index on `(portfolio_id, published_at)` can degrade query performance.
   - **Suggested Solution:** Ensure database indexes exist for foreign keys and timestamp filters:
     ```sql
     CREATE INDEX idx_news_published_instrument ON news(instrument_id, published_at DESC);
     ```
   - **Rationale:** Keeps API response times sub-100ms even as historical news data accumulates over months.

2. **Type Hinting and Error Handling for JSON Parsing in Signals**
   - **File / Line Reference:** `hayai-new/api/routers/portfolios.py`, lines 163–169
   - **Explanation:** `sentiment_breakdown` JSON fields are decoded inline with a broad `except (TypeError, ValueError):`. If the database column contains malformed JSON or empty strings, it silently sets it to `None` without logging the warning.
   - **Suggested Solution:**
     ```python
     import logging
     logger = logging.getLogger(__name__)

     for s in signals:
         if s.get('sentiment_breakdown'):
             try:
                 s['sentiment_breakdown'] = json.loads(s['sentiment_breakdown'])
             except (TypeError, ValueError) as e:
                 logger.warning(f"Failed to parse sentiment_breakdown JSON: {e}")
                 s['sentiment_breakdown'] = None
     ```
   - **Rationale:** Improves observability and aids debugging when upstream LLM ingestion stores invalid JSON payloads.

3. **CORS Middleware Origins Configuration**
   - **File / Line Reference:** `hayai-new/api/main.py`, lines 27–37
   - **Explanation:** `DEV_ORIGINS` explicitly lists `http://localhost:4200` and `http://127.0.0.1:4200`. In production deployment behind Nginx (as specified in `doc-new-app`), frontend and backend typically share the same origin or domain, but explicit production origins should be configurable via environment variables.
   - **Suggested Solution:**
     ```python
     cors_origins = settings.CORS_ORIGINS.split(",") if hasattr(settings, "CORS_ORIGINS") else ["http://localhost:4200"]
     app.add_middleware(
         CORSMiddleware,
         allow_origins=cors_origins,
         allow_credentials=True,
         allow_methods=["*"],
         allow_headers=["*"],
     )
     ```
   - **Rationale:** Prevents CORS misconfigurations when deploying the Angular SPA and FastAPI backend behind Nginx in production.

---

## ✅ Good Practices

1. **Secure Credential Comparison (`hmac.compare_digest`)**
   - **File / Line Reference:** `hayai-new/api/routers/auth.py`, lines 19–20
   - **What's done well:** Using `hmac.compare_digest()` for username and password verification successfully prevents timing attacks during authentication checks.

2. **Clean Separation of Concerns & Router Architecture**
   - **File / Line Reference:** `hayai-new/api/main.py`, lines 40–47
   - **What's done well:** FastAPI routers are cleanly modularized by domain (`auth`, `portfolios`, `config`, `holdings`, `markets`, `instruments`) with dependency injection (`Depends(require_auth)`) securing business endpoints uniformly.

3. **Robust Cookie-Based Authentication with Starlette SessionMiddleware**
   - **File / Line Reference:** `hayai-new/api/main.py`, lines 17–24; `hayai-new/api/auth.py`
   - **What's done well:** Utilizing signed HTTP-only / SameSite session cookies rather than stateless JWTs stored in localStorage avoids common client-side XSS token exfiltration risks for single-user administrative dashboards.
