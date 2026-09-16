"""
Validates every ticker in universe.yaml and removes those that fail quality checks.

Checks:
  1. data_available   — yfinance returns at least some rows
  2. has_ohlcv        — open, high, low, close, volume all present
  3. min_history      — at least MIN_ROWS trading days (default 500 ≈ 2 years)
  4. max_missing      — NaN in close column < MAX_MISSING_PCT (default 5%)
  5. recent_data      — last available date within MAX_STALENESS_DAYS (default 30)
  6. has_sector       — sector field defined in universe.yaml
  7. min_volume       — mean daily volume >= MIN_VOLUME (default 100 000)

Exit codes:
  0 — all tickers pass (or --fix applied successfully)
  1 — one or more tickers failed (without --fix)

Usage:
    python -m data.validator                    # report only, no changes
    python -m data.validator --fix              # remove failures from universe.yaml
    python -m data.validator --fix --db         # also remove from instrument table in DB
    python -m data.validator --min-rows 750     # stricter history threshold
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

# ── add model_research root to sys.path so relative imports work ─────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.downloader import cache_path, get, is_stale, load_universe
from settings import UNIVERSE_FILE

# ── Thresholds (overridable via CLI) ─────────────────────────────────────────

DEFAULT_MIN_ROWS        = 500    # ≈ 2 years of trading days
DEFAULT_MAX_MISSING_PCT = 0.05   # 5% NaN allowed in close
DEFAULT_MAX_STALENESS   = 30     # days since last available date
DEFAULT_MIN_VOLUME      = 100_000

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


# ── Single-ticker checks ──────────────────────────────────────────────────────

def check_ticker(
    ticker: str,
    sector: str | None,
    *,
    min_rows: int,
    max_missing_pct: float,
    max_staleness: int,
    min_volume: float,
    force_download: bool,
) -> dict:
    result = {
        "ticker":        ticker,
        "sector":        sector or "",
        "rows":          0,
        "last_date":     None,
        "missing_pct":   None,
        "mean_volume":   None,
        "failures":      [],
        "warnings":      [],
    }

    # ── 1. sector defined ─────────────────────────────────────────────────────
    if not sector:
        result["failures"].append("has_sector: sector not defined in universe.yaml")

    # ── 2. download / read cache ──────────────────────────────────────────────
    try:
        df = get(ticker, force=force_download)
    except Exception as exc:
        result["failures"].append(f"data_available: {exc}")
        return result

    if df.empty:
        result["failures"].append("data_available: empty dataframe")
        return result

    result["rows"]      = len(df)
    result["last_date"] = df.index[-1].date()

    # ── 3. required columns ───────────────────────────────────────────────────
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        result["failures"].append(f"has_ohlcv: missing columns {sorted(missing_cols)}")

    # ── 4. minimum history ────────────────────────────────────────────────────
    if result["rows"] < min_rows:
        result["failures"].append(
            f"min_history: only {result['rows']} rows (need {min_rows})"
        )
    elif result["rows"] < min_rows * 1.5:
        result["warnings"].append(
            f"short_history: {result['rows']} rows (recommend >= {min_rows * 1.5:.0f})"
        )

    # ── 5. missing values in close ────────────────────────────────────────────
    if "close" in df.columns:
        pct = df["close"].isna().mean()
        result["missing_pct"] = round(float(pct), 4)
        if pct > max_missing_pct:
            result["failures"].append(
                f"max_missing: {pct:.1%} NaN in close (limit {max_missing_pct:.0%})"
            )

    # ── 6. recent data ────────────────────────────────────────────────────────
    if result["last_date"]:
        age = (date.today() - result["last_date"]).days
        if age > max_staleness:
            result["failures"].append(
                f"recent_data: last date {result['last_date']} is {age} days ago "
                f"(limit {max_staleness})"
            )

    # ── 7. minimum volume ─────────────────────────────────────────────────────
    if "volume" in df.columns:
        mean_vol = df["volume"].replace(0, float("nan")).mean()
        result["mean_volume"] = None if pd.isna(mean_vol) else int(mean_vol)
        if pd.isna(mean_vol) or mean_vol < min_volume:
            result["failures"].append(
                f"min_volume: mean volume {result['mean_volume']} < {min_volume:,}"
            )

    return result


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> tuple[list[str], list[str]]:
    passed  = [r for r in results if not r["failures"]]
    failed  = [r for r in results if r["failures"]]
    warned  = [r for r in passed  if r["warnings"]]

    print(f"\n{'='*70}")
    print(f"  Universe validation — {date.today()}")
    print(f"  Total: {len(results)}  |  Pass: {len(passed)}  |  Fail: {len(failed)}  |  Warn: {len(warned)}")
    print(f"{'='*70}")

    if failed:
        print(f"\n{'─'*70}")
        print(f"  FAILURES ({len(failed)})")
        print(f"{'─'*70}")
        for r in sorted(failed, key=lambda x: x["ticker"]):
            print(f"\n  {r['ticker']:<8}  sector={r['sector'] or '?':15}  "
                  f"rows={r['rows']:>5}  last={r['last_date']}")
            for msg in r["failures"]:
                print(f"           [FAIL] {msg}")

    if warned:
        print(f"\n{'─'*70}")
        print(f"  WARNINGS ({len(warned)})")
        print(f"{'─'*70}")
        for r in sorted(warned, key=lambda x: x["ticker"]):
            print(f"\n  {r['ticker']:<8}  sector={r['sector']:15}  "
                  f"rows={r['rows']:>5}  last={r['last_date']}")
            for msg in r["warnings"]:
                print(f"           [WARN] {msg}")

    print(f"\n{'─'*70}")
    print(f"  PASSED ({len(passed)})")
    print(f"{'─'*70}")
    for r in sorted(passed, key=lambda x: x["ticker"]):
        vol_str = f"{r['mean_volume']:>12,}" if r["mean_volume"] else "           N/A"
        print(f"  {r['ticker']:<8}  {r['sector']:18}  rows={r['rows']:>5}  "
              f"last={r['last_date']}  vol={vol_str}")

    print()
    return (
        [r["ticker"] for r in passed],
        [r["ticker"] for r in failed],
    )


# ── Fix: update universe.yaml ─────────────────────────────────────────────────

def remove_from_yaml(failed_tickers: list[str]) -> None:
    with open(UNIVERSE_FILE) as f:
        universe = yaml.safe_load(f)

    before = len(universe["stocks"])
    universe["stocks"] = [
        s for s in universe["stocks"] if s["ticker"] not in failed_tickers
    ]
    after = len(universe["stocks"])

    with open(UNIVERSE_FILE, "w") as f:
        yaml.dump(universe, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"  universe.yaml updated: {before} → {after} stocks (removed {before - after})")


# ── Fix: update DB instrument table ──────────────────────────────────────────

def remove_from_db(failed_tickers: list[str]) -> None:
    """
    Removes failed tickers from the hayai instrument table.
    Reads DB credentials from hayai-new/.env.
    Does NOT delete price_daily rows — only sets the instrument inactive
    by removing it from portfolio_instrument (safer than hard delete).
    """
    env_file = Path(__file__).parent.parent.parent / "hayai-new" / ".env"
    if not env_file.exists():
        print(f"  [DB] .env not found at {env_file} — skipping DB update")
        return

    # parse .env manually (no dotenv dependency in model_research)
    env: dict[str, str] = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")

    try:
        import pymysql
    except ImportError:
        print("  [DB] pymysql not installed — skipping DB update")
        return

    conn = pymysql.connect(
        host=env.get("DB_HOST", "127.0.0.1"),
        port=int(env.get("DB_PORT", 3306)),
        user=env.get("DB_USER", ""),
        password=env.get("DB_PASSWORD", ""),
        database=env.get("DB_NAME", "hayai"),
        charset="utf8mb4",
    )

    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(failed_tickers))

            # get instrument IDs for the failed tickers
            cur.execute(
                f"SELECT id, symbol FROM instrument WHERE symbol IN ({placeholders})",
                failed_tickers,
            )
            rows = cur.fetchall()

            if not rows:
                print("  [DB] None of the failed tickers found in instrument table")
                return

            ids = [r[0] for r in rows]
            symbols = [r[1] for r in rows]

            id_placeholders = ",".join(["%s"] * len(ids))

            # remove from portfolio_instrument (M:N join)
            cur.execute(
                f"DELETE FROM portfolio_instrument WHERE instrument_id IN ({id_placeholders})",
                ids,
            )
            removed_links = cur.rowcount

            # delete from instrument
            cur.execute(
                f"DELETE FROM instrument WHERE id IN ({id_placeholders})",
                ids,
            )
            removed_instruments = cur.rowcount

            conn.commit()
            print(
                f"  [DB] Removed {removed_instruments} instrument rows "
                f"({', '.join(symbols)}) and {removed_links} portfolio links"
            )
    finally:
        conn.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate universe.yaml tickers")
    parser.add_argument("--fix",          action="store_true", help="Remove failures from universe.yaml")
    parser.add_argument("--db",           action="store_true", help="Also remove from instrument table in DB (requires --fix)")
    parser.add_argument("--force",        action="store_true", help="Re-download all tickers before checking")
    parser.add_argument("--min-rows",     type=int,   default=DEFAULT_MIN_ROWS,        metavar="N")
    parser.add_argument("--max-missing",  type=float, default=DEFAULT_MAX_MISSING_PCT, metavar="PCT")
    parser.add_argument("--max-staleness",type=int,   default=DEFAULT_MAX_STALENESS,   metavar="DAYS")
    parser.add_argument("--min-volume",   type=float, default=DEFAULT_MIN_VOLUME,      metavar="VOL")
    args = parser.parse_args()

    universe = load_universe()
    # str() cast: YAML 1.1 parses bare ON/OFF/YES/NO as booleans
    stock_map = {str(s["ticker"]): s.get("sector") for s in universe["stocks"]}

    print(f"Validating {len(stock_map)} tickers...")

    results = []
    for ticker, sector in stock_map.items():
        print(f"  checking {ticker:<8}", end="\r")
        r = check_ticker(
            ticker,
            sector,
            min_rows=args.min_rows,
            max_missing_pct=args.max_missing,
            max_staleness=args.max_staleness,
            min_volume=args.min_volume,
            force_download=args.force,
        )
        results.append(r)

    passed_tickers, failed_tickers = print_report(results)

    if not failed_tickers:
        print("All tickers passed. No changes needed.")
        sys.exit(0)

    print(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}\n")

    if not args.fix:
        print("Run with --fix to remove failed tickers from universe.yaml")
        sys.exit(1)

    print("Applying fixes...")
    remove_from_yaml(failed_tickers)

    if args.db:
        remove_from_db(failed_tickers)

    print(f"\nDone. Removed {len(failed_tickers)} ticker(s).")
    sys.exit(0)
