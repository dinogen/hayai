import time
import requests
import yfinance as yf
from app.db import get_db_connection
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.universe")

UNIVERSE_SYMBOLS = [
    # Mega Cap Tech & Growth
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC",
    "QCOM", "AVGO", "CSCO", "ORCL", "IBM", "ADBE", "CRM", "TXN", "AMAT", "MU",
    # Financials & Banking
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BLK", "SCHW", "SPGI",
    # Healthcare & Biotech
    "UNH", "JNJ", "PFE", "MRK", "ABBV", "LLY", "TMO", "DHR", "BMY", "AMGN",
    # Consumer, Industrial & Energy
    "WMT", "PG", "KO", "PEP", "MCD", "DIS", "BA", "CAT", "HON", "UPS",
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "NEE", "DUK", "SO", "GE",
    # Mid-Cap Stocks (higher-risk additions)
    "ESTC", "GTLB", "DBX", "SMCI", "ON", "CROX", "WING", "FIVE", "MRNA", "NVAX",
    "GNRC", "MTZ", "MTDR", "SOFI", "AFRM",
    # ETFs (Broad, Sector, Bonds, Commodities)
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP",
    "XLK", "U", "XLRE", "TLT", "IEF", "GLD", "SLV", "USO", "VNQ", "ARKK",
    # EM High-Yield Sovereign Bond ETFs
    "EMB", "VWOB", "PCY", "EMLC",
    # International & Indices / Rates
    "EEM", "EFA", "EWJ", "FXI", "ASHR", "VGK", "EWZ", "INDA", "VWO", "IAU"
]


def _fetch_instrument_meta(symbol, session):
    """Fetch (name, instrument_type, currency) from yfinance, with graceful fallback."""
    try:
        info = yf.Ticker(symbol, session=session).info
    except Exception as e:
        logger.warning(f"Could not fetch metadata for {symbol}: {e}")
        info = {}

    name = info.get("shortName") or info.get("longName") or symbol

    quote_type = (info.get("quoteType") or "").upper()
    if quote_type == "ETF":
        inst_type = "etf"
    elif quote_type == "INDEX" or symbol.startswith("^"):
        inst_type = "bond_yield"
    else:
        inst_type = "stock"

    currency = info.get("currency") or "USD"
    return name, inst_type, currency


def run_universe_job(portfolio_code: str = "main") -> dict:
    """Seed/refresh the investment universe in `instrument` (active=1) WITHOUT
    linking the symbols to the portfolio watchlist.

    `portfolio_instrument` is left untouched: the watchlist stays under the
    explicit control of the user via the webapp. Instruments already present
    are skipped (idempotent)."""
    logger.info(f"Seeding {len(UNIVERSE_SYMBOLS)} universe symbols into `instrument`...")
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

    inserted = 0
    already_present = 0
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for idx, symbol in enumerate(UNIVERSE_SYMBOLS):
                cursor.execute("SELECT id FROM instrument WHERE symbol = %s", (symbol,))
                if cursor.fetchone():
                    already_present += 1
                    continue
                if idx > 0:
                    time.sleep(0.5)
                name, inst_type, currency = _fetch_instrument_meta(symbol, session)
                cursor.execute(
                    "INSERT INTO instrument (symbol, name, instrument_type, currency, active) VALUES (%s, %s, %s, %s, 1)",
                    (symbol, name, inst_type, currency)
                )
                inserted += 1
            conn.commit()

    logger.info(f"Universe seeding completed: {inserted} inserted, {already_present} already present.")
    return {"universe_symbols": len(UNIVERSE_SYMBOLS), "inserted": inserted, "already_present": already_present}
