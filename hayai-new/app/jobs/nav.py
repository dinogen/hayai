from datetime import date
from app.db import execute_query, get_db_connection
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.nav")

def run_nav_job(portfolio_code: str = "main") -> dict:
    logger.info("Computing daily mark-to-market (NAV)...")

    port_rows = execute_query("""
        SELECT id, initial_capital, risk_percentage FROM portfolio WHERE code = %s
    """, (portfolio_code,))

    if not port_rows:
        logger.warning(f"Portfolio '{portfolio_code}' not found.")
        return {"nav": 0}

    port = port_rows[0]
    portfolio_id = port['id']
    initial_capital = float(port['initial_capital'])
    risk_pct = float(port['risk_percentage'])

    # Simulation cash buffer: initial_capital - invested equity (e.g. €5,000 - €4,500 = €500)
    invested_capital = initial_capital * risk_pct
    cash_balance = round(initial_capital - invested_capital, 2)

    # Latest recommendation date
    rec_date_rows = execute_query("""
        SELECT MAX(rec_date) AS rec_date FROM portfolio_recommendation WHERE portfolio_id = %s
    """, (portfolio_id,))
    rec_date = rec_date_rows[0]['rec_date'] if rec_date_rows and rec_date_rows[0]['rec_date'] else None

    pos_date = date.today().isoformat()

    if not rec_date:
        logger.warning("No recommendations yet. NAV = cash only.")
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO portfolio_cash (portfolio_id, cash_date, balance)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE balance = VALUES(balance)
                """, (portfolio_id, pos_date, cash_balance))
        return {"nav": cash_balance, "cash_balance": cash_balance, "positions_value": 0.0,
                "positions_count": 0, "rec_date": None}

    # Recommendations with cost basis (close at rec_date) and latest close (mark-to-market)
    recs = execute_query("""
        SELECT pr.instrument_id, pr.target_qty, pd_rec.close AS cost_price,
               pd_latest.close AS latest_close
        FROM portfolio_recommendation pr
        JOIN price_daily pd_rec
          ON pd_rec.instrument_id = pr.instrument_id AND pd_rec.trade_date = pr.rec_date
        LEFT JOIN (
            SELECT instrument_id, MAX(trade_date) AS max_date
            FROM price_daily WHERE close IS NOT NULL
            GROUP BY instrument_id
        ) mx ON mx.instrument_id = pr.instrument_id
        LEFT JOIN price_daily pd_latest
          ON pd_latest.instrument_id = mx.instrument_id AND pd_latest.trade_date = mx.max_date
        WHERE pr.portfolio_id = %s
        AND pr.rec_date = (SELECT MAX(rec_date) FROM portfolio_recommendation WHERE portfolio_id = %s)
    """, (portfolio_id, portfolio_id))

    upsert_pos = """
        INSERT INTO portfolio_position (portfolio_id, instrument_id, pos_date, qty, avg_price, market_value)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            qty = VALUES(qty),
            avg_price = VALUES(avg_price),
            market_value = VALUES(market_value)
    """

    positions_value = 0.0
    count = 0

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for row in recs:
                qty = row['target_qty']
                if not qty or float(qty) <= 0:
                    continue

                cost_price = float(row['cost_price']) if row['cost_price'] else None
                latest_close = float(row['latest_close']) if row['latest_close'] else cost_price
                if not latest_close:
                    continue

                qty = float(qty)
                avg_price = round(cost_price if cost_price else latest_close, 6)
                market_value = round(qty * latest_close, 2)

                cursor.execute(upsert_pos, (portfolio_id, row['instrument_id'], pos_date, qty, avg_price, market_value))
                positions_value += market_value
                count += 1

            cursor.execute("""
                INSERT INTO portfolio_cash (portfolio_id, cash_date, balance)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE balance = VALUES(balance)
            """, (portfolio_id, pos_date, cash_balance))

    nav = round(cash_balance + positions_value, 2)
    logger.info(f"NAV {nav:.2f} = cash {cash_balance:.2f} + positions {positions_value:.2f} ({count} positions)")
    return {
        "nav": nav,
        "cash_balance": cash_balance,
        "positions_value": round(positions_value, 2),
        "positions_count": count,
        "rec_date": str(rec_date) if rec_date else None,
    }
