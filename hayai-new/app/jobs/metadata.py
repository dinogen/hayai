from datetime import date, timedelta
from app.area import fallback_area_for_symbol, map_area
from app.db import execute_query, get_db_connection
from app.logging_setup import setup_logger
from app.yf_client import YahooFinanceClient

logger = setup_logger("app.jobs.metadata")

METADATA_TTL_DAYS = 30

def run_metadata_job(portfolio_code: str = "main", force: bool = False) -> dict:
    logger.info("Fetching instrument metadata (sector/country/area) from Yahoo Finance...")
    query = """
        SELECT i.id, i.symbol, i.instrument_type,
               i.metadata_date
        FROM instrument i
        JOIN portfolio_instrument pi ON i.id = pi.instrument_id
        JOIN portfolio p ON pi.portfolio_id = p.id
        WHERE p.code = %s AND i.active = 1
    """
    instruments = execute_query(query, (portfolio_code,))
    if not instruments:
        logger.warning(f"No active instruments found for portfolio '{portfolio_code}'.")
        return {"total": 0, "fetched": 0, "updated": 0, "failed": 0, "skipped_fresh": 0}

    client = YahooFinanceClient()

    today = date.today()
    cutoff = today - timedelta(days=METADATA_TTL_DAYS)

    update_query = """
        UPDATE instrument
        SET sector = %s, country = %s, area = %s, metadata_date = %s, updated_at = NOW()
        WHERE id = %s
    """

    fetched = 0
    updated = 0
    failed = 0
    skipped = 0

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for ins in instruments:
                inst_id = ins['id']
                symbol = ins['symbol']

                metadata_date = ins['metadata_date']
                fresh = metadata_date is not None and not force and metadata_date > cutoff
                if fresh:
                    skipped += 1
                    continue

                try:
                    info = client.fetch_info(symbol)
                    if not info:
                        logger.warning(f"Empty info returned for {symbol}.")
                        failed += 1
                        continue
                except Exception as ex:
                    logger.error(f"Error fetching metadata for {symbol}: {ex}")
                    failed += 1
                    continue

                is_etf_or_yield = ins['instrument_type'] in ('etf', 'bond_yield')
                sector = info.get('sector')
                if not sector and is_etf_or_yield:
                    sector = info.get('category')
                country = info.get('country')
                if country:
                    area = map_area(country)
                else:
                    area = fallback_area_for_symbol(symbol)

                cursor.execute(update_query, (sector, country, area, today, inst_id))
                fetched += 1
                if cursor.rowcount > 0:
                    updated += 1
                logger.info(
                    f"Metadata for {symbol}: sector={sector!r}, country={country!r}, area={area!r}"
                )

    logger.info(
        f"Metadata job done: {fetched} fetched, {updated} changed, {failed} failed, {skipped} skipped (fresh)"
    )
    return {
        "total": len(instruments),
        "fetched": fetched,
        "updated": updated,
        "failed": failed,
        "skipped_fresh": skipped,
    }
