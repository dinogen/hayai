"""
Downloads and caches OHLCV data for all symbols in universe.yaml.

- Stocks + macro symbols are stored as individual parquet files in data/cache/.
- A symbol is re-downloaded only if its cache file is missing or stale.
- Ticker names are sanitized for filesystem use (^, =, / replaced with _).

Usage:
    python -m data.downloader                  # download missing/stale only
    python -m data.downloader --force          # re-download everything
    python -m data.downloader --symbol NVDA    # single symbol
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

from settings import CACHE_MAX_AGE_DAYS, DATA_DIR, DATA_START, UNIVERSE_FILE


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sanitize(ticker: str) -> str:
    """Convert ticker to a safe filename stem."""
    return ticker.replace("^", "_").replace("=", "_").replace("/", "_")


def cache_path(ticker: str) -> Path:
    return DATA_DIR / f"{_sanitize(ticker)}.parquet"


def is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    age = (date.today() - date.fromtimestamp(path.stat().st_mtime)).days
    return age >= CACHE_MAX_AGE_DAYS


# ── Universe loader ───────────────────────────────────────────────────────────

def load_universe() -> dict:
    with open(UNIVERSE_FILE) as f:
        return yaml.safe_load(f)


def all_tickers(universe: dict) -> list[str]:
    stocks = [str(s["ticker"]) for s in universe["stocks"]]
    macro  = (
        universe["macro"]["indices"]
        + universe["macro"]["commodities"]
        + universe["macro"]["rates"]
        + universe["macro"]["fx"]
    )
    return stocks + macro


def stock_tickers(universe: dict) -> list[str]:
    # str() cast: YAML 1.1 parses bare ON/OFF/YES/NO as booleans
    return [str(s["ticker"]) for s in universe["stocks"]]


def macro_tickers(universe: dict) -> list[str]:
    m = universe["macro"]
    return m["indices"] + m["commodities"] + m["rates"] + m["fx"]


# ── Download ─────────────────────────────────────────────────────────────────

def _download(ticker: str, start: str = DATA_START) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # yfinance ≥ 0.2 sometimes returns MultiIndex columns even for one ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df.index   = pd.to_datetime(df.index)
    df.index.name = "date"

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep].sort_index()


def get(ticker: str, force: bool = False) -> pd.DataFrame:
    """Return cached DataFrame, downloading if necessary."""
    path = cache_path(ticker)
    if not force and not is_stale(path):
        return pd.read_parquet(path)

    df = _download(ticker)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


# ── Bulk download ─────────────────────────────────────────────────────────────

def download_all(
    tickers: list[str] | None = None,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    universe = load_universe()
    tickers  = tickers or all_tickers(universe)

    results: dict[str, pd.DataFrame] = {}
    failed:  list[str] = []

    for ticker in tickers:
        try:
            df = get(ticker, force=force)
            results[ticker] = df
            if verbose:
                print(
                    f"  ok  {ticker:<14} {len(df):5d} rows  "
                    f"{df.index[0].date()} → {df.index[-1].date()}"
                )
        except Exception as exc:
            failed.append(ticker)
            if verbose:
                print(f"  ERR {ticker:<14} {exc}")

    if failed and verbose:
        print(f"\nFailed ({len(failed)}): {failed}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OHLCV cache for universe")
    parser.add_argument("--force",  action="store_true", help="Re-download even if cache is fresh")
    parser.add_argument("--symbol", type=str, default=None, help="Download a single symbol")
    parser.add_argument("--macro",  action="store_true", help="Download macro symbols only")
    parser.add_argument("--stocks", action="store_true", help="Download stock symbols only")
    args = parser.parse_args()

    universe = load_universe()

    if args.symbol:
        tickers = [args.symbol]
    elif args.macro:
        tickers = macro_tickers(universe)
    elif args.stocks:
        tickers = stock_tickers(universe)
    else:
        tickers = None  # all

    print(f"Downloading {'all' if tickers is None else len(tickers)} symbols...")
    download_all(tickers=tickers, force=args.force)
    print("Done.")
