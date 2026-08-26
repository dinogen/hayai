from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from datetime import date, datetime
import requests
from app.db import execute_query, get_db_connection
from app.math_utils import round_short_qty
from app.portfolio_rebalance import build_trades, apply_trades
from app.jobs.universe import _fetch_instrument_meta

router = APIRouter()


class PositionTarget(BaseModel):
    instrument_id: int
    side: str  # 'long' | 'short'
    qty: float
    avg_price: float | None = None


class HoldingsSaveRequest(BaseModel):
    positions: list[PositionTarget]


class WatchlistAddRequest(BaseModel):
    instrument_id: int


class UniverseAddRequest(BaseModel):
    symbol: str


def _portfolio(code: str) -> dict:
    rows = execute_query("SELECT id, initial_capital FROM portfolio WHERE code = %s", (code,))
    if not rows:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return rows[0]


def _active_model_id(portfolio_id: int) -> int | None:
    """Resolve the model used for signals: the portfolio's linked model or the
    active fallback (same logic as app.jobs.signal)."""
    rows = execute_query("SELECT model_id FROM portfolio WHERE id = %s", (portfolio_id,))
    model_id = rows[0]['model_id'] if rows else None
    if not model_id:
        rows = execute_query("SELECT id FROM model_registry WHERE status = 'active' ORDER BY id LIMIT 1")
        model_id = rows[0]['id'] if rows else None
    return model_id


def _watchlist_rows(portfolio_id: int) -> list:
    """Watchlist instruments with latest close price, latest hybrid signal,
    latest model volatility (vol_20) and whether an open position is held.
    Instruments without a signal yield NULLs."""
    model_id = _active_model_id(portfolio_id)
    return execute_query("""
        SELECT i.id AS instrument_id, i.symbol, i.name, i.instrument_type, i.area, i.sector,
               pd.close AS current_price,
               ps.quant_score, ps.llm_sentiment_modifier, ps.final_signal, ps.signal_date,
               mp.vol_20,
               EXISTS (
                   SELECT 1 FROM portfolio_position pp
                   WHERE pp.portfolio_id = pi.portfolio_id AND pp.instrument_id = i.id
                     AND pp.qty != 0
                     AND pp.pos_date = (SELECT MAX(pos_date) FROM portfolio_position
                                        WHERE portfolio_id = pi.portfolio_id AND instrument_id = i.id)
               ) AS has_open_position
        FROM portfolio_instrument pi
        JOIN instrument i ON pi.instrument_id = i.id
        LEFT JOIN (
            SELECT instrument_id, MAX(trade_date) AS max_date
            FROM price_daily WHERE close IS NOT NULL
            GROUP BY instrument_id
        ) mx ON mx.instrument_id = i.id
        LEFT JOIN price_daily pd ON pd.instrument_id = mx.instrument_id AND pd.trade_date = mx.max_date
        LEFT JOIN portfolio_signal ps ON ps.portfolio_id = pi.portfolio_id AND ps.instrument_id = pi.instrument_id
            AND ps.signal_date = (SELECT MAX(signal_date) FROM portfolio_signal WHERE portfolio_id = pi.portfolio_id)
        LEFT JOIN model_prediction mp ON mp.instrument_id = pi.instrument_id AND mp.model_id = %s
            AND mp.as_of_date = (SELECT MAX(as_of_date) FROM model_prediction
                                 WHERE model_id = %s AND instrument_id = pi.instrument_id)
        WHERE pi.portfolio_id = %s AND i.active = 1
        ORDER BY i.area, i.symbol
    """, (model_id, model_id, portfolio_id))


def _serialize_watchlist_row(r: dict) -> dict:
    return {
        "instrument_id": int(r['instrument_id']),
        "symbol": r['symbol'],
        "name": r['name'],
        "instrument_type": r['instrument_type'],
        "area": r['area'],
        "sector": r['sector'],
        "current_price": round(float(r['current_price']), 6) if r['current_price'] else None,
        "signal_date": r['signal_date'].isoformat() if r['signal_date'] else None,
        "quant_score": round(float(r['quant_score']), 6) if r['quant_score'] is not None else None,
        "llm_sentiment_modifier": round(float(r['llm_sentiment_modifier']), 4) if r['llm_sentiment_modifier'] is not None else None,
        "final_signal": round(float(r['final_signal']), 6) if r['final_signal'] is not None else None,
        "vol_20": round(float(r['vol_20']), 4) if r['vol_20'] is not None else None,
        "has_open_position": bool(r['has_open_position']),
    }


def _latest_close_map() -> dict:
    rows = execute_query("""
        SELECT pd.instrument_id, pd.close
        FROM price_daily pd
        JOIN (
            SELECT instrument_id, MAX(trade_date) AS max_date
            FROM price_daily WHERE close IS NOT NULL
            GROUP BY instrument_id
        ) mx ON mx.instrument_id = pd.instrument_id AND mx.max_date = pd.trade_date
    """)
    return {int(r['instrument_id']): float(r['close']) for r in rows}


def _current_positions(portfolio_id: int) -> list:
    return execute_query("""
        SELECT pp.instrument_id, pp.qty, pp.avg_price, pp.market_value, pp.pos_date,
               i.symbol, i.name, i.instrument_type,
               pd.close AS current_price
        FROM portfolio_position pp
        JOIN instrument i ON pp.instrument_id = i.id
        JOIN (
            SELECT instrument_id, MAX(pos_date) AS max_date
            FROM portfolio_position WHERE portfolio_id = %s
            GROUP BY instrument_id
        ) cur ON cur.instrument_id = pp.instrument_id AND cur.max_date = pp.pos_date
        LEFT JOIN (
            SELECT instrument_id, MAX(trade_date) AS max_date
            FROM price_daily WHERE close IS NOT NULL
            GROUP BY instrument_id
        ) mx ON mx.instrument_id = pp.instrument_id
        LEFT JOIN price_daily pd ON pd.instrument_id = mx.instrument_id AND pd.trade_date = mx.max_date
        WHERE pp.portfolio_id = %s AND pp.qty != 0
    """, (portfolio_id, portfolio_id))


@router.get("/portfolios/{code}/holdings")
def get_holdings(code: str):
    port = _portfolio(code)
    portfolio_id = port['id']
    initial_capital = float(port['initial_capital'])

    pos_rows = _current_positions(portfolio_id)

    positions = []
    for r in pos_rows:
        qty = float(r['qty'])
        avg_price = float(r['avg_price']) if r['avg_price'] else 0.0
        close = float(r['current_price']) if r['current_price'] else avg_price
        side = 'long' if qty > 0 else 'short'
        market_value = round(qty * close, 2)
        pnl = round(qty * (close - avg_price), 2)
        pnl_pct = round(pnl / (abs(qty) * avg_price) * 100, 2) if avg_price else 0.0
        positions.append({
            "instrument_id": int(r['instrument_id']),
            "symbol": r['symbol'],
            "name": r['name'],
            "instrument_type": r['instrument_type'],
            "side": side,
            "qty": abs(qty),
            "avg_price": round(avg_price, 6),
            "current_price": round(close, 6),
            "market_value": market_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })

    cash_rows = execute_query("""
        SELECT balance FROM portfolio_cash
        WHERE portfolio_id = %s ORDER BY cash_date DESC LIMIT 1
    """, (portfolio_id,))
    cash = float(cash_rows[0]['balance']) if cash_rows else initial_capital
    positions_value = round(sum(p['market_value'] for p in positions), 2)
    nav = round(cash + positions_value, 2)

    watchlist_rows = _watchlist_rows(portfolio_id)
    watchlist = [_serialize_watchlist_row(r) for r in watchlist_rows]

    recs = execute_query("""
        SELECT pr.instrument_id, pr.side, pr.target_qty, pr.rec_date, i.symbol
        FROM portfolio_recommendation pr
        JOIN instrument i ON pr.instrument_id = i.id
        WHERE pr.portfolio_id = %s
        AND pr.rec_date = (SELECT MAX(rec_date) FROM portfolio_recommendation WHERE portfolio_id = %s)
    """, (portfolio_id, portfolio_id))
    rec_date = recs[0]['rec_date'] if recs else None

    return {
        "portfolio_code": code,
        "as_of_date": date.today().isoformat(),
        "nav": nav,
        "cash_balance": round(cash, 2),
        "positions_value": positions_value,
        "initial_capital": round(initial_capital, 2),
        "positions": positions,
        "watchlist": watchlist,
        "latest_recommendations": {
            "rec_date": rec_date,
            "items": [
                {"instrument_id": int(r['instrument_id']), "symbol": r['symbol'],
                 "side": r['side'], "target_qty": float(r['target_qty'] or 0)}
                for r in recs
            ],
        },
    }


@router.get("/portfolios/{code}/holdings/report.md")
def get_holdings_report(code: str):
    """Markdown report of the current open positions (purchase price and date)."""
    port = _portfolio(code)
    portfolio_id = port['id']
    initial_capital = float(port['initial_capital'])

    pos_rows = _current_positions(portfolio_id)

    purchase_rows = execute_query("""
        SELECT pt.instrument_id, MAX(pt.trade_date) AS purchase_date
        FROM portfolio_trade pt
        WHERE pt.portfolio_id = %s AND pt.side IN ('buy', 'short')
        GROUP BY pt.instrument_id
    """, (portfolio_id,))
    purchase_map = {int(r['instrument_id']): r['purchase_date'].isoformat() for r in purchase_rows}

    positions = []
    for r in pos_rows:
        qty = float(r['qty'])
        avg_price = float(r['avg_price']) if r['avg_price'] else 0.0
        close = float(r['current_price']) if r['current_price'] else avg_price
        side = 'LONG' if qty > 0 else 'SHORT'
        market_value = round(qty * close, 2)
        pnl = round(qty * (close - avg_price), 2)
        purchase_date = purchase_map.get(int(r['instrument_id']), r['pos_date'].isoformat())
        positions.append({
            "symbol": r['symbol'],
            "name": r['name'],
            "side": side,
            "qty": abs(qty),
            "avg_price": avg_price,
            "purchase_date": purchase_date,
            "current_price": close,
            "market_value": market_value,
            "pnl": pnl,
        })

    cash_rows = execute_query("""
        SELECT balance FROM portfolio_cash
        WHERE portfolio_id = %s ORDER BY cash_date DESC LIMIT 1
    """, (portfolio_id,))
    cash = float(cash_rows[0]['balance']) if cash_rows else initial_capital
    positions_value = round(sum(p['market_value'] for p in positions), 2)
    nav = round(cash + positions_value, 2)

    def _eur(v: float) -> str:
        return f"€{v:,.2f}"

    md_lines = []
    md_lines.append(f"# Report Portafoglio — {code}")
    md_lines.append("")
    md_lines.append(f"Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")
    md_lines.append("## Riepilogo")
    md_lines.append("")
    md_lines.append("| Voce | Importo |")
    md_lines.append("|---|---|")
    md_lines.append(f"| Capitale iniziale | {_eur(initial_capital)} |")
    md_lines.append(f"| NAV | {_eur(nav)} |")
    md_lines.append(f"| Liquidità (cash) | {_eur(cash)} |")
    md_lines.append(f"| Valore posizioni | {_eur(positions_value)} |")
    md_lines.append("")
    md_lines.append("## Posizioni correnti")
    md_lines.append("")
    md_lines.append("| Simbolo | Nome | Side | Qty | Prezzo carico | Data acquisto | Prezzo attuale | Valore | P&L |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    for p in positions:
        md_lines.append(
            f"| {p['symbol']} | {p['name'] or p['symbol']} | {p['side']} "
            f"| {p['qty']:,.4f} | {p['avg_price']:,.4f} | {p['purchase_date']} "
            f"| {p['current_price']:,.4f} | {_eur(p['market_value'])} | {_eur(p['pnl'])} |"
        )

    md = "\n".join(md_lines) + "\n"
    headers = {"Content-Disposition": f'attachment; filename="report-{code}.md"'}
    return Response(content=md, media_type="text/markdown; charset=utf-8", headers=headers)


@router.get("/portfolios/{code}/watchlist")
def get_watchlist(code: str):
    port = _portfolio(code)
    rows = _watchlist_rows(port['id'])
    return [_serialize_watchlist_row(r) for r in rows]


@router.get("/portfolios/{code}/universe")
def get_universe(code: str):
    """Investment universe: active instruments NOT linked to the portfolio,
    usable as candidates to add to the watchlist."""
    port = _portfolio(code)
    portfolio_id = port['id']
    rows = execute_query("""
        SELECT i.id AS instrument_id, i.symbol, i.name, i.instrument_type, i.area, i.sector,
               pd.close AS current_price
        FROM instrument i
        LEFT JOIN portfolio_instrument pi ON pi.instrument_id = i.id AND pi.portfolio_id = %s
        LEFT JOIN (
            SELECT instrument_id, MAX(trade_date) AS max_date
            FROM price_daily WHERE close IS NOT NULL
            GROUP BY instrument_id
        ) mx ON mx.instrument_id = i.id
        LEFT JOIN price_daily pd ON pd.instrument_id = mx.instrument_id AND pd.trade_date = mx.max_date
        WHERE i.active = 1 AND pi.portfolio_id IS NULL
        ORDER BY i.area, i.symbol
    """, (portfolio_id,))
    return [{
        "instrument_id": int(r['instrument_id']),
        "symbol": r['symbol'],
        "name": r['name'],
        "instrument_type": r['instrument_type'],
        "area": r['area'],
        "sector": r['sector'],
        "current_price": round(float(r['current_price']), 6) if r['current_price'] else None,
    } for r in rows]


@router.post("/portfolios/{code}/universe")
def add_to_universe(code: str, payload: UniverseAddRequest):
    """Add a brand-new ticker to the investment universe (active, NOT linked
    to the watchlist). Fetches metadata from yfinance as a best effort."""
    port = _portfolio(code)
    portfolio_id = port['id']
    symbol = (payload.symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=422, detail="Symbol is required")

    linked = execute_query("""
        SELECT i.id FROM instrument i
        JOIN portfolio_instrument pi ON pi.instrument_id = i.id AND pi.portfolio_id = %s
        WHERE i.symbol = %s
    """, (portfolio_id, symbol))
    if linked:
        raise HTTPException(status_code=409, detail=f"{symbol} is already in the watchlist")

    ins = execute_query(
        "SELECT id, symbol, name, instrument_type, currency, area, sector, active FROM instrument WHERE symbol = %s",
        (symbol,),
    )

    if ins:
        row = ins[0]
        added = False
        if not row['active']:
            execute_query("UPDATE instrument SET active = 1 WHERE id = %s", (row['id'],), fetch=False)
        inst_id = int(row['id'])
        return {
            "instrument_id": inst_id,
            "symbol": row['symbol'],
            "name": row['name'],
            "instrument_type": row['instrument_type'],
            "currency": row['currency'],
            "area": row['area'],
            "sector": row['sector'],
            "added": added,
            "already_in_universe": True,
            "message": f"{symbol} era già nell'universo (ora attivo).",
        }

    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    name, inst_type, currency = _fetch_instrument_meta(symbol, session)

    execute_query(
        "INSERT INTO instrument (symbol, name, instrument_type, currency, active) VALUES (%s, %s, %s, %s, 1)",
        (symbol, name, inst_type, currency),
        fetch=False,
    )
    new_row = execute_query(
        "SELECT id, symbol, name, instrument_type, currency, area, sector FROM instrument WHERE symbol = %s",
        (symbol,),
    )[0]
    return {
        "instrument_id": int(new_row['id']),
        "symbol": new_row['symbol'],
        "name": new_row['name'],
        "instrument_type": new_row['instrument_type'],
        "currency": new_row['currency'],
        "area": new_row['area'],
        "sector": new_row['sector'],
        "added": True,
        "already_in_universe": False,
        "message": f"{symbol} aggiunto all'universo.",
    }


@router.post("/portfolios/{code}/watchlist")
def add_to_watchlist(code: str, payload: WatchlistAddRequest):
    port = _portfolio(code)
    portfolio_id = port['id']
    ins = execute_query(
        "SELECT id, symbol, name, instrument_type, area, sector FROM instrument WHERE id = %s AND active = 1",
        (payload.instrument_id,),
    )
    if not ins:
        raise HTTPException(status_code=404, detail=f"Instrument {payload.instrument_id} not found or inactive")
    row = ins[0]
    linked = execute_query(
        "SELECT 1 FROM portfolio_instrument WHERE portfolio_id = %s AND instrument_id = %s",
        (portfolio_id, payload.instrument_id),
    )
    added = False
    if not linked:
        execute_query(
            "INSERT INTO portfolio_instrument (portfolio_id, instrument_id) VALUES (%s, %s)",
            (portfolio_id, payload.instrument_id),
            fetch=False,
        )
        added = True
    return {
        "instrument_id": int(row['id']),
        "symbol": row['symbol'],
        "name": row['name'],
        "instrument_type": row['instrument_type'],
        "area": row['area'],
        "sector": row['sector'],
        "added": added,
    }


@router.delete("/portfolios/{code}/watchlist/{instrument_id}")
def remove_from_watchlist(code: str, instrument_id: int):
    port = _portfolio(code)
    portfolio_id = port['id']
    ins = execute_query("SELECT id, symbol FROM instrument WHERE id = %s", (instrument_id,))
    if not ins:
        raise HTTPException(status_code=404, detail=f"Instrument {instrument_id} not found")
    symbol = ins[0]['symbol']

    linked = execute_query(
        "SELECT 1 FROM portfolio_instrument WHERE portfolio_id = %s AND instrument_id = %s",
        (portfolio_id, instrument_id),
    )
    if not linked:
        return {
            "instrument_id": instrument_id,
            "symbol": symbol,
            "removed": False,
            "message": "Instrument was not in the watchlist.",
        }

    open_pos = execute_query("""
        SELECT 1 FROM portfolio_position pp
        WHERE pp.portfolio_id = %s AND pp.instrument_id = %s AND pp.qty != 0
          AND pp.pos_date = (SELECT MAX(pos_date) FROM portfolio_position
                             WHERE portfolio_id = %s AND instrument_id = %s)
    """, (portfolio_id, instrument_id, portfolio_id, instrument_id))
    if open_pos:
        raise HTTPException(
            status_code=422,
            detail=f"Instrument {symbol} has an open position: close it first in 'Portafoglio Attuale'.",
        )

    execute_query(
        "DELETE FROM portfolio_instrument WHERE portfolio_id = %s AND instrument_id = %s",
        (portfolio_id, instrument_id),
        fetch=False,
    )
    return {
        "instrument_id": instrument_id,
        "symbol": symbol,
        "removed": True,
        "message": "Instrument removed from the watchlist. It stays available in the universe.",
    }


@router.post("/portfolios/{code}/holdings/save")
def save_holdings(code: str, payload: HoldingsSaveRequest):
    port = _portfolio(code)
    portfolio_id = port['id']
    initial_capital = float(port['initial_capital'])

    # Normalize short quantities to whole units (arithmetic rounding). A short
    # that rounds to zero is treated as closed (dropped from the target).
    normalized: list[PositionTarget] = []
    for p in payload.positions:
        if p.side == 'short':
            qty = round_short_qty(p.qty)
            if qty == 0:
                continue
            normalized.append(PositionTarget(instrument_id=p.instrument_id, side=p.side, qty=qty, avg_price=p.avg_price))
        else:
            normalized.append(p)

    # Validation
    seen = set()
    for p in normalized:
        if p.side not in ('long', 'short'):
            raise HTTPException(status_code=422, detail=f"side must be 'long' or 'short' (instrument {p.instrument_id})")
        if p.qty <= 0:
            raise HTTPException(status_code=422, detail=f"qty must be positive (instrument {p.instrument_id})")
        if p.instrument_id in seen:
            raise HTTPException(status_code=422, detail=f"duplicate instrument_id {p.instrument_id}")
        seen.add(p.instrument_id)
        wl = execute_query(
            "SELECT 1 FROM portfolio_instrument WHERE portfolio_id = %s AND instrument_id = %s",
            (portfolio_id, p.instrument_id),
        )
        if not wl:
            raise HTTPException(status_code=422, detail=f"instrument {p.instrument_id} is not in the portfolio watchlist")

    target = {
        p.instrument_id: {"side": p.side, "qty": p.qty, "avg_price": p.avg_price}
        for p in normalized
    }

    cur_rows = execute_query("""
        SELECT pp.instrument_id, pp.qty, pp.avg_price
        FROM portfolio_position pp
        JOIN (
            SELECT instrument_id, MAX(pos_date) AS max_date
            FROM portfolio_position WHERE portfolio_id = %s
            GROUP BY instrument_id
        ) cur ON cur.instrument_id = pp.instrument_id AND cur.max_date = pp.pos_date
        WHERE pp.portfolio_id = %s
    """, (portfolio_id, portfolio_id))
    current = {
        int(r['instrument_id']): {
            "qty": float(r['qty']),
            "avg_price": float(r['avg_price']) if r['avg_price'] else None,
        }
        for r in cur_rows
    }

    close_map = _latest_close_map()

    # Build the trades to move the current positions towards the target and apply
    # them atomically (shared logic used also by the weekly 'align' batch job).
    trades, desired, snapshot_avg = build_trades(current, target, close_map, threshold_eur=None)

    trade_date = date.today().isoformat()

    with get_db_connection(autocommit=False) as conn:
        with conn.cursor() as cursor:
            summary = apply_trades(
                conn, cursor, portfolio_id, trade_date, trades, desired, current,
                initial_capital, close_map, snapshot_avg=snapshot_avg,
            )

    return {
        "portfolio_code": code,
        "as_of_date": trade_date,
        "nav": summary["nav"],
        "cash_balance": summary["cash_balance"],
        "positions_value": summary["positions_value"],
        "positions_saved": len(desired),
        "trades_executed": summary["trades_executed"],
        "message": "Holdings saved successfully.",
    }
