"""
EXP-014 — Portfolio simulation.

Prende le predizioni del modello migliore e simula un portafoglio
long-short (Q5 long, Q1 short) o long-only (Q5).

Metriche riportate:
    gross_annual_ret   rendimento lordo annualizzato
    net_annual_ret     rendimento netto (dopo costi)
    sharpe             Sharpe ratio (annualizzato, rf=0)
    max_drawdown       massimo drawdown del NAV
    turnover_annual    rotazione annuale (round-trip %)
    hit_rate           % giorni con rendimento positivo

I rendimenti effettivi sono caricati dai parquet in data/cache/.
I costi di transazione (cost_bps) sono one-way in basis points.

Usage:
    python -m evaluation.portfolio --dir experiments/results/dnn_linear_lr1e4_drop02
    python -m evaluation.portfolio --dir experiments/results/dnn_linear_lr1e4_drop02 --cost 5 10 20
    python -m evaluation.portfolio --dir experiments/results/dnn_linear_lr1e4_drop02 --long-only
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.downloader import get
from settings import TARGET_HORIZONS


# ── Raw return loader ─────────────────────────────────────────────────────────

def get_raw_returns(meta: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    For each (date, ticker) in meta, compute the raw h-day forward log return.
    Loads OHLCV from cache for each ticker.

    Returns DataFrame indexed as meta, columns y5_raw, y10_raw, y15_raw.
    """
    raw = pd.DataFrame(
        np.nan,
        index=meta.index,
        columns=[f"y{h}_raw" for h in horizons],
        dtype=np.float32,
    )

    for ticker, grp in meta.groupby("ticker"):
        try:
            df = get(ticker)
        except Exception:
            continue
        if df.empty or "close" not in df.columns:
            continue

        log_c = np.log(df["close"].astype(float).ffill().clip(lower=1e-8))
        dates  = pd.to_datetime(grp["date"].values)

        for h in horizons:
            fwd = log_c.shift(-h) - log_c
            vals = fwd.reindex(dates, method=None).values
            raw.loc[grp.index, f"y{h}_raw"] = vals.astype(np.float32)

    return raw


# ── Portfolio simulation ──────────────────────────────────────────────────────

def simulate(
    pred:       np.ndarray,     # (n,) predicted signal
    raw_ret:    np.ndarray,     # (n,) raw h-day forward return
    dates:      np.ndarray,     # (n,) date per sample
    tickers:    np.ndarray,     # (n,) ticker per sample
    *,
    n_quintiles: int   = 5,
    cost_bps:    float = 10.0,  # one-way cost in basis points
    long_only:   bool  = False,
    min_stocks:  int   = 10,
) -> pd.DataFrame:
    """
    Simulate daily portfolio returns.

    Each day: rank stocks by pred, go long Q5 (top 20%), short Q1 (bottom 20%).
    Raw h-day return is treated as the period return (overlapping approximation).
    Transaction costs applied based on daily turnover.

    Returns DataFrame with columns:
        gross_ret, cost, net_ret, q1_ret, q5_ret, n_stocks
    indexed by date.
    """
    df = pd.DataFrame({
        "date":   pd.to_datetime(dates),
        "ticker": tickers,
        "pred":   pred,
        "ret":    raw_ret,
    }).dropna(subset=["ret"])

    rows: list[dict] = []
    prev_q5: set[str] = set()
    prev_q1: set[str] = set()

    for date, grp in df.groupby("date"):
        n = len(grp)
        if n < min_stocks:
            continue

        q_size = max(1, n // n_quintiles)
        ranked = grp.sort_values("pred")

        q1_grp = ranked.iloc[:q_size]
        q5_grp = ranked.iloc[-q_size:]

        q1_tickers = set(q1_grp["ticker"])
        q5_tickers = set(q5_grp["ticker"])

        q1_ret = float(q1_grp["ret"].mean())
        q5_ret = float(q5_grp["ret"].mean())

        gross = q5_ret if long_only else (q5_ret - q1_ret)

        # turnover = fraction of positions that changed vs yesterday
        to_q5 = len(q5_tickers.symmetric_difference(prev_q5)) / max(len(q5_tickers), 1)
        if long_only:
            turnover = to_q5
        else:
            to_q1 = len(q1_tickers.symmetric_difference(prev_q1)) / max(len(q1_tickers), 1)
            turnover = (to_q5 + to_q1) / 2.0

        cost = turnover * cost_bps / 10_000

        prev_q5 = q5_tickers
        prev_q1 = q1_tickers

        rows.append({
            "date":      date,
            "gross_ret": gross,
            "cost":      cost,
            "net_ret":   gross - cost,
            "q1_ret":    q1_ret,
            "q5_ret":    q5_ret,
            "n_stocks":  n,
        })

    return pd.DataFrame(rows).set_index("date")


# ── Statistics ─────────────────────────────────────────────────────────────────

def portfolio_stats(
    sim: pd.DataFrame,
    horizon: int = 5,
    trading_days: int = 252,
    cost_bps: float = 10.0,
) -> dict:
    """
    Compute annualized portfolio statistics from a simulation DataFrame.

    The h-day returns are converted to per-day equivalents for annualization
    (divide by horizon).
    """
    per_day_factor = 1.0 / horizon

    gross = sim["gross_ret"] * per_day_factor
    net   = sim["net_ret"]   * per_day_factor
    cost  = sim["cost"]      * per_day_factor

    ann = trading_days

    # NAV series (start at 1)
    nav_gross = (1 + gross).cumprod()
    nav_net   = (1 + net).cumprod()

    def max_dd(nav: pd.Series) -> float:
        roll_max = nav.cummax()
        dd = (nav - roll_max) / roll_max
        return float(dd.min())

    def sharpe(ret: pd.Series) -> float:
        if ret.std() == 0:
            return float("nan")
        return float(ret.mean() / ret.std() * np.sqrt(ann))

    n_days = len(sim)

    return {
        "n_days":            n_days,
        "gross_annual_ret":  float(gross.mean() * ann),
        "net_annual_ret":    float(net.mean() * ann),
        "avg_cost_annual":   float(cost.mean() * ann),
        "sharpe_gross":      sharpe(gross),
        "sharpe_net":        sharpe(net),
        "max_dd_gross":      max_dd(nav_gross),
        "max_dd_net":        max_dd(nav_net),
        "hit_rate":          float((net > 0).mean()),
        "turnover_annual":   float(sim["cost"].mean() / (cost_bps / 10_000) * ann),
    }


def print_portfolio_report(stats: dict, label: str, cost_bps: float) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}  (cost={cost_bps:.0f}bps one-way)")
    print(f"{'='*60}")
    print(f"  Days evaluated  : {stats['n_days']:,}")
    print(f"  Gross annual ret: {stats['gross_annual_ret']:+.1%}")
    print(f"  Net annual ret  : {stats['net_annual_ret']:+.1%}")
    print(f"  Annual cost drag: {stats['avg_cost_annual']:.2%}")
    print(f"  Sharpe (gross)  : {stats['sharpe_gross']:+.2f}")
    print(f"  Sharpe (net)    : {stats['sharpe_net']:+.2f}")
    print(f"  Max drawdown    : {stats['max_dd_net']:.1%}")
    print(f"  Hit rate (net)  : {stats['hit_rate']:.1%}")
    print(f"  Annual turnover : {stats['turnover_annual']:.0%}")


# ── Full analysis ─────────────────────────────────────────────────────────────

def run_portfolio_analysis(
    results_dir: Path,
    horizons:   list[int] = None,
    cost_levels: list[float] = (5.0, 10.0, 20.0),
    long_only:  bool = False,
) -> None:
    if horizons is None:
        horizons = TARGET_HORIZONS

    pred_path = results_dir / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} not found.\n"
            f"Re-run the experiment to generate predictions.parquet."
        )

    print(f"\nLoading predictions from {pred_path.parent.name}...")
    pred_df = pd.read_parquet(pred_path)
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    print(f"  {len(pred_df):,} samples  |  "
          f"{pred_df['date'].min().date()} → {pred_df['date'].max().date()}")

    print("\nLoading raw forward returns from cache...")
    raw = get_raw_returns(pred_df, horizons)
    print(f"  Missing: {raw.isna().sum().sum()} values")

    for h in horizons:
        pred_col = f"pred_y{h}"
        raw_col  = f"y{h}_raw"
        if pred_col not in pred_df.columns:
            continue

        y_pred   = pred_df[pred_col].values
        y_raw    = raw[raw_col].values
        dates    = pred_df["date"].values
        tickers  = pred_df["ticker"].values

        for cost_bps in cost_levels:
            sim   = simulate(y_pred, y_raw, dates, tickers,
                             cost_bps=cost_bps, long_only=long_only)
            stats = portfolio_stats(sim, horizon=h, cost_bps=cost_bps)
            label = f"Y{h} {'Long-only Q5' if long_only else 'L/S Q5-Q1'}"
            print_portfolio_report(stats, label, cost_bps)

        # save simulation at middle cost level
        mid_cost = cost_levels[len(cost_levels) // 2]
        sim_mid  = simulate(y_pred, y_raw, dates, tickers,
                            cost_bps=mid_cost, long_only=long_only)
        sim_mid.to_csv(results_dir / f"portfolio_y{h}.csv")

    print(f"\nPortfolio CSVs saved to {results_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Portfolio simulation (EXP-014)")
    parser.add_argument("--dir",       type=str, required=True,
                        help="Path to experiment results dir (must contain predictions.parquet)")
    parser.add_argument("--cost",      type=float, nargs="+", default=[5.0, 10.0, 20.0],
                        help="Transaction costs in bps (one-way)")
    parser.add_argument("--horizons",  type=int, nargs="+", default=TARGET_HORIZONS)
    parser.add_argument("--long-only", action="store_true", dest="long_only")
    args = parser.parse_args()

    run_portfolio_analysis(
        results_dir  = Path(args.dir),
        horizons     = args.horizons,
        cost_levels  = args.cost,
        long_only    = args.long_only,
    )
