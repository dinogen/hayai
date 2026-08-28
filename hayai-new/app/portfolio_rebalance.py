from datetime import date

from app.db import execute_query, get_db_connection
from app.logging_setup import setup_logger
from app.math_utils import round_short_qty

logger = setup_logger("app.portfolio_rebalance")

DEFAULT_STALE_DAYS = 4


def _amount(side: str, qty: float, price: float) -> float:
    # buy / cover consume cash (negative), sell / short generate proceeds (positive).
    return round(qty * price, 2) if side in ("sell", "short") else round(-qty * price, 2)


def build_trades(current: dict, target: dict, close_map: dict, threshold_eur: float | None = None):
    """Compute the trades needed to move `current` positions towards `target`.

    Args:
        current: {instrument_id: {"qty": signed float, "avg_price": float|None}} open positions.
        target: {instrument_id: {"side": "long"|"short", "qty": positive float, "avg_price": float|None}}
                desired composition.
        close_map: {instrument_id: latest close price}.
        threshold_eur: optional tolerance. Only same-direction adjustments below the threshold
                       are skipped (hold); opening/closing always execute.

    Returns:
        (trades, desired, snapshot_avg):
        - trades: list of (instrument_id, side, qty, price, amount) to execute.
        - desired: {instrument_id: signed qty} of open positions after the trades.
        - snapshot_avg: {instrument_id: avg_price} for the post-trade position snapshots.
    """
    trades = []
    desired = {}
    snapshot_avg = {}

    all_ids = set(current.keys()) | set(target.keys())

    for inst_id in all_ids:
        cur = current.get(inst_id, {})
        cur_signed = float(cur.get("qty") or 0.0)
        cur_avg = float(cur["avg_price"]) if cur.get("avg_price") else None
        t = target.get(inst_id)
        tgt_signed = t["qty"] if t and t["side"] == "long" else (-t["qty"] if t else 0.0)
        t_avg = float(t["avg_price"]) if t and t.get("avg_price") and float(t["avg_price"]) > 0 else None

        if cur_signed == tgt_signed:
            if tgt_signed != 0:
                desired[inst_id] = tgt_signed
                snapshot_avg[inst_id] = t_avg or cur_avg or close_map.get(inst_id, 0.0)
            continue

        close = close_map.get(inst_id, cur_avg or 0.0)

        # Tolerance threshold: only gates same-direction adjustments; open/close always execute.
        if (
            threshold_eur is not None
            and cur_signed != 0
            and tgt_signed != 0
            and (cur_signed > 0) == (tgt_signed > 0)
        ):
            diff_eur = abs(tgt_signed - cur_signed) * close
            if diff_eur < threshold_eur:
                logger.info(
                    f"Instrument {inst_id}: variation €{diff_eur:.2f} below threshold "
                    f"€{threshold_eur:.2f} -> hold (position unchanged)"
                )
                desired[inst_id] = cur_signed
                snapshot_avg[inst_id] = cur_avg or close
                continue

        # Full close of an existing position (opposite sign or target zero).
        if cur_signed != 0 and (tgt_signed == 0 or (cur_signed > 0) != (tgt_signed > 0)):
            side = "sell" if cur_signed > 0 else "cover"
            qty = abs(cur_signed)
            trades.append((inst_id, side, qty, round(close, 6), _amount(side, qty, close)))

        # Open / increase / reduce.
        if tgt_signed != 0:
            if cur_signed != 0 and (cur_signed > 0) == (tgt_signed > 0):
                # Same direction: adjust by the signed difference (increase vs reduce).
                diff = tgt_signed - cur_signed
                if diff > 0:
                    side = "buy" if tgt_signed > 0 else "cover"
                elif diff < 0:
                    side = "sell" if tgt_signed > 0 else "short"
                else:
                    side = None
                open_qty = abs(diff)
            else:
                # Fresh open (no current position) or re-open after a side flip.
                side = "buy" if tgt_signed > 0 else "short"
                open_qty = abs(tgt_signed)
            if open_qty > 0 and side:
                price = t_avg if t_avg else close
                trades.append((inst_id, side, open_qty, round(float(price), 6), _amount(side, open_qty, float(price))))

        if tgt_signed != 0:
            desired[inst_id] = tgt_signed
            if cur_signed != 0 and (cur_signed > 0) == (tgt_signed > 0):
                # Same direction (increase/reduce): keep the existing cost basis.
                snapshot_avg[inst_id] = t_avg or cur_avg or close
            else:
                # Fresh open (or side flip): cost basis is the execution price.
                snapshot_avg[inst_id] = t_avg or close

    return trades, desired, snapshot_avg


def build_reconciliation(current: dict, target: dict, close_map: dict, threshold_eur: float | None = None) -> dict:
    """Map the trades that `build_trades` would execute to human-readable actions.

    This is the single source of truth for the reconciliation table: the UI always
    shows exactly what the execution (build_trades) will do, including side flips
    (close + re-open) which a naive absolute-quantity comparison would get wrong.

    Args:
        current: {instrument_id: {"qty": signed float, "avg_price": float|None}}
        target: {instrument_id: {"side": "long"|"short", "qty": positive float, "avg_price": float|None}}
        close_map: {instrument_id: latest close price}
        threshold_eur: optional tolerance (same semantics as build_trades).

    Returns:
        {instrument_id: {"action", "message", "diff_qty"}} with action in
        buy/sell/short/cover/flip/hold.
    """
    def _fmt_qty(q: float) -> str:
        return f"{q:.2f}"

    # Normalize short targets to whole units (same rule as align/save/execute);
    # a target that rounds to zero is treated as closed (dropped).
    normalized_target = {}
    for inst_id, t in target.items():
        qty = t["qty"]
        if t["side"] == "short":
            qty = round_short_qty(qty)
            if qty == 0:
                continue
        normalized_target[inst_id] = {**t, "qty": qty}
    target = normalized_target

    trades, desired, _ = build_trades(current, target, close_map, threshold_eur=threshold_eur)

    trades_by_inst: dict[int, list] = {}
    for inst_id, side, qty, _price, _amount in trades:
        trades_by_inst.setdefault(inst_id, []).append((side, qty))

    result: dict = {}
    for inst_id in set(current) | set(target):
        cur_signed = float(current.get(inst_id, {}).get("qty") or 0.0)
        t = target.get(inst_id)
        tgt_signed = t["qty"] if t and t["side"] == "long" else (-t["qty"] if t else 0.0)
        post_signed = desired.get(inst_id, 0.0)
        inst_trades = trades_by_inst.get(inst_id, [])

        if not inst_trades:
            if cur_signed == 0:
                continue  # nothing held, nothing recommended
            result[inst_id] = {
                "action": "hold",
                "message": f"mantieni {_fmt_qty(abs(cur_signed))} (invariato)",
                "diff_qty": 0.0,
            }
        elif len(inst_trades) == 1:
            side, qty = inst_trades[0]
            qty_str = _fmt_qty(qty)
            if side == "buy":
                action = "buy"
                message = f"apri long {qty_str}" if cur_signed == 0 else f"compra {qty_str}"
            elif side == "sell":
                action = "sell"
                message = f"chiudi long (vendi {qty_str})" if post_signed == 0 else f"vendi {qty_str}"
            elif side == "short":
                action = "short"
                message = f"apri short {qty_str}" if cur_signed == 0 else f"shorta {qty_str}"
            else:  # cover
                action = "cover"
                message = f"chiudi short (copri {qty_str})" if post_signed == 0 else f"copri {qty_str}"
            result[inst_id] = {"action": action, "message": message, "diff_qty": float(qty)}
        else:
            # Two trades: full close + re-open in the opposite direction (side flip).
            close_side, close_qty = inst_trades[0]
            open_side, open_qty = inst_trades[1]
            close_txt = "long" if close_side == "sell" else "short"
            open_txt = "long" if open_side == "buy" else "short"
            result[inst_id] = {
                "action": "flip",
                "message": f"chiudi {close_txt} e apri {open_txt} {_fmt_qty(open_qty)}",
                "diff_qty": float(close_qty) + float(open_qty),
            }

    return result


def apply_trades(
    conn, cursor, portfolio_id: int, trade_date: str, trades: list,
    desired: dict, current: dict, initial_capital: float, close_map: dict,
    snapshot_avg: dict | None = None,
) -> dict:
    """Persist trades and the post-trade portfolio state (positions + cash) atomically.

    - Inserts every trade into portfolio_trade.
    - Snapshots open positions (desired) for trade_date, and a qty=0 snapshot for
      previously held instruments that are now closed.
    - Upserts portfolio_cash as initial_capital + sum(amount) over the whole trade log.

    Returns a summary dict (cash_balance, positions_value, nav, trades_executed).
    """
    cursor.execute(
        "SELECT COALESCE(SUM(amount), 0) AS total FROM portfolio_trade WHERE portfolio_id = %s",
        (portfolio_id,),
    )
    cash_from_log = float(cursor.fetchone()["total"])

    for inst_id, side, qty, price, amount in trades:
        cursor.execute(
            """
            INSERT INTO portfolio_trade (portfolio_id, instrument_id, trade_date, side, qty, price, amount)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (portfolio_id, inst_id, trade_date, side, qty, price, amount),
        )

    upsert_pos = """
        INSERT INTO portfolio_position (portfolio_id, instrument_id, pos_date, qty, avg_price, market_value)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            qty = VALUES(qty),
            avg_price = VALUES(avg_price),
            market_value = VALUES(market_value)
    """

    positions_value = 0.0
    snapshot_avg = snapshot_avg or {}
    for inst_id, qty_signed in desired.items():
        avg_price = float(snapshot_avg.get(inst_id) or close_map.get(inst_id, 0.0) or 0.0)
        market_value = round(qty_signed * close_map.get(inst_id, avg_price), 2)
        positions_value += market_value
        cursor.execute(upsert_pos, (portfolio_id, inst_id, trade_date, qty_signed, round(avg_price, 6), market_value))

    # Closed positions get a qty=0 snapshot so the latest state reflects the closure.
    for inst_id in current.keys():
        if inst_id not in desired:
            cursor.execute(upsert_pos, (portfolio_id, inst_id, trade_date, 0, 0, 0))

    cash_total = initial_capital + cash_from_log + sum(a for (_, _, _, _, a) in trades)
    cursor.execute(
        """
        INSERT INTO portfolio_cash (portfolio_id, cash_date, balance)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE balance = VALUES(balance)
        """,
        (portfolio_id, trade_date, round(cash_total, 2)),
    )

    nav = round(cash_total + positions_value, 2)
    return {
        "cash_balance": round(cash_total, 2),
        "positions_value": round(positions_value, 2),
        "nav": nav,
        "trades_executed": len(trades),
        "positions_open": len(desired),
    }


def align_portfolio_to_recommendations(
    portfolio_code: str = "main", stale_days: int = DEFAULT_STALE_DAYS, force: bool = False
) -> dict:
    """Align the actual portfolio to the latest model recommendations.

    Weekly job (scheduled outside the nightly cycle): reads the latest rec_date from
    portfolio_recommendation, guards against stale recommendations, builds the target
    composition (respecting rebalance_threshold_eur) and applies it as simulated trades.
    """
    port_rows = execute_query(
        "SELECT id, initial_capital, rebalance_threshold_eur FROM portfolio WHERE code = %s",
        (portfolio_code,),
    )
    if not port_rows:
        logger.warning(f"Portfolio '{portfolio_code}' not found.")
        return {"skipped": "portfolio_not_found", "portfolio_code": portfolio_code}

    port = port_rows[0]
    portfolio_id = int(port["id"])
    initial_capital = float(port["initial_capital"])
    threshold = float(port.get("rebalance_threshold_eur") or 50.0)

    rec_rows = execute_query(
        """
        SELECT pr.instrument_id, pr.side, pr.target_qty, pr.rec_date
        FROM portfolio_recommendation pr
        WHERE pr.portfolio_id = %s
        AND pr.rec_date = (SELECT MAX(rec_date) FROM portfolio_recommendation WHERE portfolio_id = %s)
        """,
        (portfolio_id, portfolio_id),
    )
    if not rec_rows:
        logger.warning("No recommendations available: nothing to align.")
        return {"skipped": "no_recommendations", "portfolio_code": portfolio_code}

    rec_date = rec_rows[0]["rec_date"]
    rec_date_iso = rec_date.isoformat() if hasattr(rec_date, "isoformat") else str(rec_date)
    age_days = (date.today() - rec_date).days if hasattr(rec_date, "__sub__") else 0

    if age_days > stale_days and not force:
        logger.info(
            f"Recommendations stale (rec_date={rec_date_iso}, age={age_days}d > {stale_days}d): skipping alignment."
        )
        return {
            "skipped": "stale",
            "portfolio_code": portfolio_code,
            "rec_date": rec_date_iso,
            "stale_days": age_days,
        }

    wl_rows = execute_query(
        "SELECT instrument_id FROM portfolio_instrument WHERE portfolio_id = %s", (portfolio_id,)
    )
    watchlist = {int(r["instrument_id"]) for r in wl_rows}

    target = {}
    for r in rec_rows:
        inst_id = int(r["instrument_id"])
        if inst_id not in watchlist:
            logger.info(f"Instrument {inst_id} not in watchlist: skipped.")
            continue
        qty = float(r["target_qty"]) if r["target_qty"] is not None else 0.0
        if qty <= 0:
            continue
        if r["side"] == "short":
            qty = round_short_qty(qty)
            if qty == 0:
                logger.info(f"Instrument {inst_id}: short qty rounds to 0 -> position closed, skipped.")
                continue
        target[inst_id] = {"side": r["side"], "qty": qty, "avg_price": None}

    if not target:
        logger.warning("No valid target positions after filtering.")
        return {
            "skipped": "no_valid_targets",
            "portfolio_code": portfolio_code,
            "rec_date": rec_date_iso,
        }

    cur_rows = execute_query(
        """
        SELECT pp.instrument_id, pp.qty, pp.avg_price
        FROM portfolio_position pp
        JOIN (
            SELECT instrument_id, MAX(pos_date) AS max_date
            FROM portfolio_position WHERE portfolio_id = %s
            GROUP BY instrument_id
        ) cur ON cur.instrument_id = pp.instrument_id AND cur.max_date = pp.pos_date
        WHERE pp.portfolio_id = %s
        """,
        (portfolio_id, portfolio_id),
    )
    current = {
        int(r["instrument_id"]): {
            "qty": float(r["qty"]),
            "avg_price": float(r["avg_price"]) if r["avg_price"] else None,
        }
        for r in cur_rows
    }

    close_rows = execute_query(
        """
        SELECT pd.instrument_id, pd.close
        FROM price_daily pd
        JOIN (
            SELECT instrument_id, MAX(trade_date) AS max_date
            FROM price_daily WHERE close IS NOT NULL
            GROUP BY instrument_id
        ) mx ON mx.instrument_id = pd.instrument_id AND mx.max_date = pd.trade_date
        """
    )
    close_map = {int(r["instrument_id"]): float(r["close"]) for r in close_rows}

    trades, desired, snapshot_avg = build_trades(current, target, close_map, threshold_eur=threshold)
    if not trades:
        logger.info("No trades needed: portfolio is already aligned (within tolerance).")
        return {
            "skipped": "already_aligned",
            "portfolio_code": portfolio_code,
            "rec_date": rec_date_iso,
            "threshold_eur": round(threshold, 2),
            "target_count": len(target),
            "trades_executed": 0,
        }

    trade_date = date.today().isoformat()
    logger.info(f"Aligning portfolio '{portfolio_code}' to recommendations of {rec_date_iso}: {len(trades)} trades.")

    with get_db_connection(autocommit=False) as conn:
        with conn.cursor() as cursor:
            summary = apply_trades(
                conn, cursor, portfolio_id, trade_date, trades, desired, current,
                initial_capital, close_map, snapshot_avg=snapshot_avg,
            )

    logger.info(
        f"Alignment complete: {summary['trades_executed']} trades, "
        f"NAV €{summary['nav']:.2f} = cash €{summary['cash_balance']:.2f} + positions €{summary['positions_value']:.2f}"
    )
    return {
        "skipped": None,
        "portfolio_code": portfolio_code,
        "rec_date": rec_date_iso,
        "trade_date": trade_date,
        "threshold_eur": round(threshold, 2),
        "target_count": len(target),
        **summary,
    }
