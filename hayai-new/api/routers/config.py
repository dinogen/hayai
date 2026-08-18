from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.config import get_news_llm_enabled, set_news_llm_enabled
from app.db import execute_query, get_db_connection

router = APIRouter()

class ResetRequest(BaseModel):
    initial_capital: float

class ConfigUpdateRequest(BaseModel):
    max_assets: int | None = None
    rebalance_threshold_eur: float | None = None

class NewsLlmUpdateRequest(BaseModel):
    news_llm_enabled: bool

@router.get("/config/news-llm")
def get_news_llm_flag():
    return {"news_llm_enabled": get_news_llm_enabled()}

@router.put("/config/news-llm")
def update_news_llm_flag(payload: NewsLlmUpdateRequest):
    return {"news_llm_enabled": set_news_llm_enabled(payload.news_llm_enabled)}

@router.get("/portfolios/{code}/config")
def get_portfolio_config(code: str):
    port = execute_query("""
        SELECT code, name, active, n_long, n_short, max_assets, risk_percentage, initial_capital, rebalance_threshold_eur
        FROM portfolio WHERE code = %s
    """, (code,))
    if not port:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return port[0]

@router.post("/portfolios/{code}/config")
def update_portfolio_config(code: str, payload: ConfigUpdateRequest):
    port = execute_query("SELECT id FROM portfolio WHERE code = %s", (code,))
    if not port:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    portfolio_id = port[0]['id']

    updates = []
    params = []
    if payload.max_assets is not None:
        if payload.max_assets < 1:
            raise HTTPException(status_code=422, detail="max_assets must be a positive integer")
        updates.append("max_assets = %s")
        params.append(payload.max_assets)

    if payload.rebalance_threshold_eur is not None:
        if payload.rebalance_threshold_eur < 0:
            raise HTTPException(status_code=422, detail="rebalance_threshold_eur cannot be negative")
        updates.append("rebalance_threshold_eur = %s")
        params.append(payload.rebalance_threshold_eur)

    if updates:
        params.append(portfolio_id)
        execute_query(f"UPDATE portfolio SET {', '.join(updates)} WHERE id = %s", tuple(params), fetch=False)

    updated = execute_query("""
        SELECT code, name, active, n_long, n_short, max_assets, risk_percentage, initial_capital, rebalance_threshold_eur
        FROM portfolio WHERE id = %s
    """, (portfolio_id,))
    return updated[0]

@router.post("/portfolios/{code}/reset")
def reset_portfolio(code: str, payload: ResetRequest):
    capital = payload.initial_capital
    if capital <= 0:
        raise HTTPException(status_code=422, detail="initial_capital must be a positive number")

    port = execute_query("SELECT id FROM portfolio WHERE code = %s", (code,))
    if not port:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    portfolio_id = port[0]['id']

    # Transaction: update capital, clear portfolio state, set cash back to new capital.
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE portfolio SET initial_capital = %s WHERE id = %s", (capital, portfolio_id))
            cursor.execute("DELETE FROM portfolio_position WHERE portfolio_id = %s", (portfolio_id,))
            cursor.execute("DELETE FROM portfolio_cash WHERE portfolio_id = %s", (portfolio_id,))
            cursor.execute("DELETE FROM portfolio_trade WHERE portfolio_id = %s", (portfolio_id,))
            cursor.execute("DELETE FROM portfolio_recommendation WHERE portfolio_id = %s", (portfolio_id,))
            cursor.execute(
                "INSERT INTO portfolio_cash (portfolio_id, cash_date, balance) VALUES (%s, CURDATE(), %s)",
                (portfolio_id, capital)
            )

    return {
        "portfolio_code": code,
        "initial_capital": capital,
        "reset": True,
        "message": "Portfolio state reset. The next nightly run will recompute recommendations."
    }
