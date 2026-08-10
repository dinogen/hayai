from fastapi import APIRouter, HTTPException
from app.db import execute_query

router = APIRouter()

MAX_DAYS = 750


def _instrument_by_symbol(symbol: str) -> dict | None:
    rows = execute_query("""
        SELECT id, symbol, name, instrument_type, currency, sector, country, area, metadata_date
        FROM instrument
        WHERE symbol = %s
    """, (symbol,))
    return rows[0] if rows else None


def _resolve_model_id() -> int | None:
    rows = execute_query("SELECT model_id FROM portfolio WHERE code = 'main' AND active = 1")
    model_id = rows[0]['model_id'] if rows else None
    if not model_id:
        rows = execute_query("SELECT id FROM model_registry WHERE status = 'active' ORDER BY id LIMIT 1")
        model_id = rows[0]['id'] if rows else None
    return model_id


@router.get("/instruments/{symbol}")
def get_instrument_detail(symbol: str, days: int = 250):
    days = max(1, min(days, MAX_DAYS))

    ins = _instrument_by_symbol(symbol)
    if not ins:
        raise HTTPException(status_code=404, detail=f"Instrument {symbol} not found")

    inst_id = int(ins['id'])

    model_id = _resolve_model_id()
    if model_id:
        sig_rows = execute_query("""
            SELECT ps.quant_score, ps.llm_sentiment_modifier, ps.final_signal, ps.signal_date,
                   mp.vol_20
            FROM portfolio_signal ps
            LEFT JOIN model_prediction mp ON mp.instrument_id = ps.instrument_id AND mp.model_id = %s
                AND mp.as_of_date = (SELECT MAX(as_of_date) FROM model_prediction
                                     WHERE model_id = %s AND instrument_id = ps.instrument_id)
            WHERE ps.instrument_id = %s
            AND ps.signal_date = (SELECT MAX(signal_date) FROM portfolio_signal WHERE instrument_id = %s)
            ORDER BY ps.signal_date DESC
            LIMIT 1
        """, (model_id, model_id, inst_id, inst_id))
    else:
        sig_rows = execute_query("""
            SELECT ps.quant_score, ps.llm_sentiment_modifier, ps.final_signal, ps.signal_date
            FROM portfolio_signal ps
            WHERE ps.instrument_id = %s
            AND ps.signal_date = (SELECT MAX(signal_date) FROM portfolio_signal WHERE instrument_id = %s)
            ORDER BY ps.signal_date DESC
            LIMIT 1
        """, (inst_id, inst_id))

    latest_signal = None
    if sig_rows:
        s = sig_rows[0]
        latest_signal = {
            "signal_date": s['signal_date'].isoformat() if s['signal_date'] else None,
            "quant_score": round(float(s['quant_score']), 6) if s['quant_score'] is not None else None,
            "llm_sentiment_modifier": round(float(s['llm_sentiment_modifier']), 4) if s['llm_sentiment_modifier'] is not None else None,
            "final_signal": round(float(s['final_signal']), 6) if s['final_signal'] is not None else None,
            "vol_20": round(float(s['vol_20']), 4) if s['vol_20'] is not None else None,
        }

    prices = execute_query("""
        SELECT trade_date, open, high, low, close, volume
        FROM price_daily
        WHERE instrument_id = %s
        ORDER BY trade_date DESC
        LIMIT %s
    """, (inst_id, days))
    prices = [{
        "trade_date": p['trade_date'].isoformat(),
        "open": float(p['open']) if p['open'] is not None else None,
        "high": float(p['high']) if p['high'] is not None else None,
        "low": float(p['low']) if p['low'] is not None else None,
        "close": float(p['close']) if p['close'] is not None else None,
        "volume": int(p['volume']) if p['volume'] is not None else 0,
    } for p in prices]
    prices.reverse()

    news = execute_query("""
        SELECT n.id, n.title, n.publisher, n.published_at,
               ns.impact_score, ns.impact_duration, ns.confidence, ns.catalyst
        FROM news n
        LEFT JOIN news_sentiment ns ON n.id = ns.news_id
        WHERE n.instrument_id = %s
        ORDER BY n.published_at DESC
        LIMIT 10
    """, (inst_id,))
    news = [{
        "id": int(n['id']),
        "title": n['title'],
        "publisher": n['publisher'],
        "published_at": n['published_at'].isoformat() if n['published_at'] else None,
        "impact_score": float(n['impact_score']) if n['impact_score'] is not None else None,
        "impact_duration": n['impact_duration'],
        "confidence": float(n['confidence']) if n['confidence'] is not None else None,
        "catalyst": n['catalyst'],
    } for n in news]

    return {
        "instrument": {
            "id": inst_id,
            "symbol": ins['symbol'],
            "name": ins['name'],
            "instrument_type": ins['instrument_type'],
            "currency": ins['currency'],
            "sector": ins['sector'],
            "country": ins['country'],
            "area": ins['area'],
            "metadata_date": ins['metadata_date'].isoformat() if ins['metadata_date'] else None,
        },
        "latest_signal": latest_signal,
        "prices": prices,
        "news": news,
    }
