import requests
import yfinance as yf
from datetime import datetime
from app.db import execute_query, get_db_connection
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.news")

def run_news_job(portfolio_code: str = "main") -> dict:
    logger.info("Fetching active instruments for news ingestion...")
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
        return {"news_inserted": 0}

    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

    upsert_query = """
        INSERT INTO news (source_id, instrument_id, title, publisher, link, published_at, summary)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            title = VALUES(title),
            publisher = VALUES(publisher),
            link = VALUES(link),
            published_at = VALUES(published_at),
            summary = VALUES(summary)
    """

    total_inserted = 0
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for ins in instruments:
                inst_id = ins['id']
                symbol = ins['symbol']
                try:
                    ticker = yf.Ticker(symbol, session=session)
                    news_list = ticker.news
                    if not news_list:
                        continue

                    rows = []
                    for item in news_list:
                        source_id = str(item.get('id') or item.get('uuid') or item.get('link'))
                        if not source_id:
                            continue
                        
                        title = item.get('title', '')
                        publisher = item.get('publisher', '')
                        link = item.get('link', '')
                        
                        pub_time = item.get('providerPublishTime')
                        published_at = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M:%S') if pub_time else None
                        
                        summary = item.get('summary', '')

                        rows.append((source_id, inst_id, title, publisher, link, published_at, summary))

                    if rows:
                        cursor.executemany(upsert_query, rows)
                        total_inserted += cursor.rowcount
                        logger.info(f"Ingested {len(rows)} news items for {symbol}.")
                except Exception as e:
                    logger.error(f"Error fetching news for {symbol}: {e}")

    return {"news_inserted": total_inserted}
