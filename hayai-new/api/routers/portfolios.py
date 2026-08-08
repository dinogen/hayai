from fastapi import APIRouter, HTTPException
from app.db import execute_query

router = APIRouter()

@router.get("/portfolios")
def get_portfolios():
    portfolios = execute_query("SELECT id, code, name, active, n_long, n_short, risk_percentage, initial_capital FROM portfolio WHERE active = 1")
    return portfolios

@router.get("/portfolios/{code}")
def get_portfolio_detail(code: str):
    port = execute_query("SELECT * FROM portfolio WHERE code = %s", (code,))
    if not port:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    instruments = execute_query("""
        SELECT i.id, i.symbol, i.name, i.instrument_type, i.currency
        FROM instrument i
        JOIN portfolio_instrument pi ON i.id = pi.instrument_id
        JOIN portfolio p ON pi.portfolio_id = p.id
        WHERE p.code = %s AND i.active = 1
    """, (code,))

    return {
        "portfolio": port[0],
        "instruments": instruments
    }

@router.get("/portfolios/{code}/recommendations/latest")
def get_latest_recommendations(code: str):
    port = execute_query("SELECT id, initial_capital, risk_percentage FROM portfolio WHERE code = %s", (code,))
    if not port:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    portfolio_id = port[0]['id']
    equity_indicativa = float(port[0]['initial_capital'])
    risk_pct = float(port[0]['risk_percentage'])

    recs = execute_query("""
        SELECT pr.rec_date, pr.weight, pr.side, pr.target_amount, pr.target_qty, pr.prev_weight,
        i.symbol, i.name, i.instrument_type, i.currency,
        ps.quant_score, ps.llm_sentiment_modifier, ps.final_signal, ps.ai_rationale,
        pd.close as current_price
        FROM portfolio_recommendation pr
        JOIN instrument i ON pr.instrument_id = i.id
        LEFT JOIN portfolio_signal ps ON pr.portfolio_id = ps.portfolio_id AND pr.instrument_id = ps.instrument_id AND pr.rec_date = ps.signal_date
        LEFT JOIN price_daily pd ON pr.instrument_id = pd.instrument_id AND pd.trade_date = pr.rec_date
        WHERE pr.portfolio_id = %s
        AND pr.rec_date = (SELECT MAX(rec_date) FROM portfolio_recommendation WHERE portfolio_id = %s)
        ORDER BY ABS(pr.weight) DESC
    """, (portfolio_id, portfolio_id))

    rec_date = recs[0]['rec_date'] if recs else None

    return {
        "portfolio_code": code,
        "rec_date": rec_date,
        "equity_indicativa": equity_indicativa,
        "risk_percentage": risk_pct,
        "items": recs
    }

@router.get("/portfolios/{code}/signals")
def get_portfolio_signals(code: str):
    port = execute_query("SELECT id FROM portfolio WHERE code = %s", (code,))
    if not port:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    portfolio_id = port[0]['id']
    signals = execute_query("""
        SELECT ps.signal_date, ps.quant_score, ps.llm_sentiment_modifier, ps.final_signal, ps.ai_rationale,
        i.symbol, i.name, i.instrument_type
        FROM portfolio_signal ps
        JOIN instrument i ON ps.instrument_id = i.id
        WHERE ps.portfolio_id = %s
        AND ps.signal_date = (SELECT MAX(signal_date) FROM portfolio_signal WHERE portfolio_id = %s)
        ORDER BY ps.final_signal DESC
    """, (portfolio_id, portfolio_id))

    return signals

@router.get("/portfolios/{code}/news")
def get_portfolio_news(code: str):
    port = execute_query("SELECT id FROM portfolio WHERE code = %s", (code,))
    if not port:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    portfolio_id = port[0]['id']
    news = execute_query("""
        SELECT n.id, n.title, n.publisher, n.link, n.published_at, n.summary,
        i.symbol, ns.sentiment, ns.confidence, ns.catalyst, ns.rationale as ai_rationale
        FROM news n
        JOIN instrument i ON n.instrument_id = i.id
        JOIN portfolio_instrument pi ON i.id = pi.instrument_id
        LEFT JOIN news_sentiment ns ON n.id = ns.news_id
        WHERE pi.portfolio_id = %s
        ORDER BY n.published_at DESC
        LIMIT 50
    """, (portfolio_id,))

    return news

@router.get("/portfolios/{code}/summaries/latest")
def get_latest_summary(code: str):
    port = execute_query("SELECT id FROM portfolio WHERE code = %s", (code,))
    if not port:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    portfolio_id = port[0]['id']
    summary = execute_query("""
        SELECT summary_date, markdown
        FROM news_summary
        WHERE portfolio_id = %s
        ORDER BY summary_date DESC
        LIMIT 1
    """, (portfolio_id,))

    if not summary:
        return {"portfolio_code": code, "summary_date": None, "markdown": "Nessun riassunto disponibile."}
    
    return summary[0]
