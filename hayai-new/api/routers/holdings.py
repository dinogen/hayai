from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import date
from app.db import execute_query, get_db_connection
from app.math_utils import round_short_qty

router = APIRouter()


class PositionTarget(BaseModel):
    instrument_id: int
    side: str  # 'long' | 'short'
    qty: float
    avg_price: float | None = None


class HoldingsSaveRequest(BaseModel):
    positions: list[PositionTarget]


def _portfolio(code: str) -> dict:
    rows = execute_query("SELECT id, initial_capital FROM portfolio WHERE code = %s", (code,))
    if not rows:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return rows[0]


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
        SELECT pp.instrument_id, pp.qty, pp.avg_price, pp.market_value,
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

    watchlist_rows = execute_query("""
        SELECT i.id AS instrument_id, i.symbol, i.name, i.instrument_type,
               pd.close AS current_price
        FROM portfolio_instrument pi
        JOIN instrument i ON pi.instrument_id = i.id
        LEFT JOIN (
            SELECT instrument_id, MAX(trade_date) AS max_date
            FROM price_daily WHERE close IS NOT NULL
            GROUP BY instrument_id
        ) mx ON mx.instrument_id = i.id
        LEFT JOIN price_daily pd ON pd.instrument_id = mx.instrument_id AND pd.trade_date = mx.max_date
        WHERE pi.portfolio_id = %s AND i.active = 1
    """, (portfolio_id,))
    watchlist = [
        {
            "instrument_id": int(r['instrument_id']),
            "symbol": r['symbol'],
            "name": r['name'],
            "instrument_type": r['instrument_type'],
            "current_price": round(float(r['current_price']), 6) if r['current_price'] else None,
        }
        for r in watchlist_rows
    ]

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

    # Pre-existing cash effect from the trade log (committed data).
    sum_rows = execute_query("""
        SELECT COALESCE(SUM(amount), 0) AS total FROM portfolio_trade WHERE portfolio_id = %s
    """, (portfolio_id,))
    cash_from_log = float(sum_rows[0]['total'])

    trade_date = date.today().isoformat()

    def _close_price(inst_id: int, fallback: float) -> float:
        return close_map.get(inst_id, fallback)

    def _amount(side: str, qty: float, price: float) -> float:
        # buy / cover consume cash (negative), sell / short generate proceeds (positive).
        return round(qty * price, 2) if side in ('sell', 'short') else round(-qty * price, 2)

    trades = []
    all_ids = set(current.keys()) | set(target.keys())

    for inst_id in all_ids:
        cur_signed = current.get(inst_id, {}).get('qty', 0.0)
        cur_avg = current.get(inst_id, {}).get('avg_price')
        t = target.get(inst_id)
        tgt_signed = t['qty'] if t and t['side'] == 'long' else (-t['qty'] if t else 0.0)

        if cur_signed == tgt_signed:
            continue

        close = _close_price(inst_id, cur_avg or 0.0)

        # Full close of an existing position (opposite sign or target zero).
        if cur_signed != 0 and (tgt_signed == 0 or (cur_signed > 0) != (tgt_signed > 0)):
            side = 'sell' if cur_signed > 0 else 'cover'
            qty = abs(cur_signed)
            trades.append((inst_id, side, qty, round(close, 6), _amount(side, qty, close)))

        # Open / increase / reduce.
        if tgt_signed != 0:
            same_direction = cur_signed != 0 and (cur_signed > 0) == (tgt_signed > 0)
            open_qty = abs(tgt_signed) if not same_direction else abs(tgt_signed - cur_signed)
            if open_qty > 0:
                side = 'buy' if tgt_signed > 0 else 'short'
                price = (t['avg_price'] if t and t['avg_price'] and t['avg_price'] > 0 else close)
                trades.append((inst_id, side, open_qty, round(float(price), 6), _amount(side, open_qty, float(price))))

    # Recompute cash from initial capital + full trade log.
    cash_total = initial_capital + cash_from_log + sum(a for (_, _, _, _, a) in trades)

    upsert_pos = """
        INSERT INTO portfolio_position (portfolio_id, instrument_id, pos_date, qty, avg_price, market_value)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            qty = VALUES(qty),
            avg_price = VALUES(avg_price),
            market_value = VALUES(market_value)
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for inst_id, side, qty, price, amount in trades:
                cursor.execute("""
                    INSERT INTO portfolio_trade (portfolio_id, instrument_id, trade_date, side, qty, price, amount)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (portfolio_id, inst_id, trade_date, side, qty, price, amount))

            # Snapshot of open target positions.
            for inst_id, t in target.items():
                qty_signed = t['qty'] if t['side'] == 'long' else -t['qty']
                avg_price = float(t['avg_price']) if t['avg_price'] and t['avg_price'] > 0 else close_map.get(inst_id, 0.0)
                close = close_map.get(inst_id, avg_price)
                market_value = round(qty_signed * close, 2)
                cursor.execute(upsert_pos, (portfolio_id, inst_id, trade_date, qty_signed, round(avg_price, 6), market_value))

            # Closed positions get a qty=0 snapshot so the latest state reflects the closure.
            for inst_id in current.keys():
                if inst_id not in target:
                    cursor.execute(upsert_pos, (portfolio_id, inst_id, trade_date, 0, 0, 0))

            cursor.execute("""
                INSERT INTO portfolio_cash (portfolio_id, cash_date, balance)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE balance = VALUES(balance)
            """, (portfolio_id, trade_date, round(cash_total, 2)))

    positions_value = 0.0
    for inst_id, t in target.items():
        qty_signed = t['qty'] if t['side'] == 'long' else -t['qty']
        close = close_map.get(inst_id, 0.0)
        positions_value += qty_signed * close
    nav = round(cash_total + positions_value, 2)

    return {
        "portfolio_code": code,
        "as_of_date": trade_date,
        "nav": nav,
        "cash_balance": round(cash_total, 2),
        "positions_value": round(positions_value, 2),
        "positions_saved": len(target),
        "trades_executed": len(trades),
        "message": "Holdings saved successfully.",
    }
