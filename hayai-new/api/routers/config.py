from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.db import execute_query, get_db_connection

router = APIRouter()

class ResetRequest(BaseModel):
    initial_capital: float

@router.get("/portfolios/{code}/config")
def get_portfolio_config(code: str):
    port = execute_query("""
        SELECT code, name, active, n_long, n_short, risk_percentage, initial_capital
        FROM portfolio WHERE code = %s
    """, (code,))
    if not port:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return port[0]

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
