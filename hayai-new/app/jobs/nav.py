from datetime import date
from app.db import execute_query, get_db_connection
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.nav")

def run_nav_job(portfolio_code: str = "main") -> dict:
    logger.info("Computing daily mark-to-market (NAV) for actual portfolio...")

    port_rows = execute_query("""
        SELECT id, initial_capital FROM portfolio WHERE code = %s
    """, (portfolio_code,))

    if not port_rows:
        logger.warning(f"Portfolio '{portfolio_code}' not found.")
        return {"nav": 0}

    port = port_rows[0]
    portfolio_id = port['id']
    initial_capital = float(port['initial_capital'])

    pos_date = date.today().isoformat()

    # Current actual positions (latest snapshot, open positions only: qty != 0).
    # Short positions are stored with negative qty; market_value = qty * close.
    positions = execute_query("""
        SELECT pp.instrument_id, pp.qty, pp.avg_price, pp.market_value,
               pd.close AS latest_close
        FROM portfolio_position pp
        JOIN (
            SELECT instrument_id, MAX(pos_date) AS max_date
            FROM portfolio_position
            WHERE portfolio_id = %s
            GROUP BY instrument_id
        ) cur ON cur.instrument_id = pp.instrument_id AND cur.max_date = pp.pos_date
        LEFT JOIN (
            SELECT instrument_id, MAX(trade_date) AS max_date
            FROM price_daily WHERE close IS NOT NULL
            GROUP BY instrument_id
        ) mx ON mx.instrument_id = pp.instrument_id
        LEFT JOIN price_daily pd
          ON pd.instrument_id = mx.instrument_id AND pd.trade_date = mx.max_date
        WHERE pp.portfolio_id = %s AND pp.qty != 0
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
            for row in positions:
                qty = float(row['qty'])
                avg_price = float(row['avg_price']) if row['avg_price'] else None
                latest_close = float(row['latest_close']) if row['latest_close'] else None
                # Valuation price: latest close, fallback to avg price, then last market value.
                val_price = latest_close or avg_price
                if not val_price:
                    continue

                market_value = round(qty * val_price, 2)
                cursor.execute(upsert_pos, (portfolio_id, row['instrument_id'], pos_date, qty, avg_price, market_value))
                positions_value += market_value
                count += 1

            # Carry forward the last known cash balance (cash is updated only by manual trades).
            cash_rows = execute_query("""
                SELECT balance FROM portfolio_cash
                WHERE portfolio_id = %s ORDER BY cash_date DESC LIMIT 1
            """, (portfolio_id,))
            cash_balance = float(cash_rows[0]['balance']) if cash_rows else initial_capital

            cursor.execute("""
                INSERT INTO portfolio_cash (portfolio_id, cash_date, balance)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE balance = VALUES(balance)
            """, (portfolio_id, pos_date, round(cash_balance, 2)))

    nav = round(cash_balance + positions_value, 2)
    logger.info(f"NAV {nav:.2f} = cash {cash_balance:.2f} + positions {positions_value:.2f} ({count} positions)")
    return {
        "nav": nav,
        "cash_balance": round(cash_balance, 2),
        "positions_value": round(positions_value, 2),
        "positions_count": count,
        "pos_date": pos_date,
    }
