import pandas as pd
from datetime import datetime
from app.db import execute_query, get_db_connection
from app.jobs.cache import load_cached, save_cached
from app.logging_setup import setup_logger
from app.yf_client import YahooFinanceClient

logger = setup_logger("app.jobs.data")

def run_data_job(portfolio_code: str = "main") -> dict:
    logger.info("Fetching active instruments...")
    query = """
        SELECT i.id, i.symbol 
        FROM instrument i
        JOIN portfolio_instrument pi ON i.id = pi.instrument_id
        JOIN portfolio p ON pi.portfolio_id = p.id
        WHERE p.code = %s AND i.active = 1
    """
    instruments = execute_query(query, (portfolio_code,))
    if not instruments:
        logger.warning(f"No active instruments found for portfolio '{portfolio_code}'.")
        return {"instruments_processed": 0}

    client = YahooFinanceClient()

    success_count = 0
    upsert_query = """
        INSERT INTO price_daily (instrument_id, trade_date, open, high, low, close, adjusted_close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open = VALUES(open),
            high = VALUES(high),
            low = VALUES(low),
            close = VALUES(close),
            adjusted_close = VALUES(adjusted_close),
            volume = VALUES(volume)
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for ins in instruments:
                inst_id = ins['id']
                symbol = ins['symbol']

                df = load_cached(f"{symbol}_daily")
                if df is None:
                    try:
                        df = client.download_history(symbol, period="1y", auto_adjust=True)
                        if df.empty:
                            logger.warning(f"No history returned for {symbol}.")
                            continue
                        df = df.reset_index()
                        save_cached(f"{symbol}_daily", df)
                        logger.info(f"Downloaded {len(df)} daily records for {symbol}.")
                    except Exception as ex:
                        logger.error(f"Error downloading symbol {symbol}: {ex}")
                        continue

                date_col = 'Date' if 'Date' in df.columns else df.columns[0]

                def _to_float_or_none(v):
                    if v is None or pd.isna(v):
                        return None
                    return float(v)

                rows_to_insert = []
                for _, row in df.iterrows():
                    trade_date = pd.to_datetime(row[date_col]).strftime('%Y-%m-%d')
                    open_p = _to_float_or_none(row.get('Open'))
                    high_p = _to_float_or_none(row.get('High'))
                    low_p = _to_float_or_none(row.get('Low'))
                    close_p = _to_float_or_none(row.get('Close'))
                    adj_close = _to_float_or_none(row.get('Close'))
                    vol = int(row.get('Volume', 0)) if not pd.isna(row.get('Volume', 0)) else 0

                    if close_p is None:
                        continue

                    rows_to_insert.append((
                        inst_id, trade_date, open_p, high_p, low_p, close_p, adj_close, vol
                    ))

                if rows_to_insert:
                    cursor.executemany(upsert_query, rows_to_insert)
                    success_count += 1
                    logger.info(f"Upserted {len(rows_to_insert)} daily records for {symbol}.")

    return {"instruments_processed": success_count, "total_requested": len(instruments)}
