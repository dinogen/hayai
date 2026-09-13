"""
Evaluation metrics for Hayai V2 model research.

Primary metric : mean daily Spearman correlation
                 (ranking quality across the universe each day)

Secondary       : quintile analysis, hit rate, MAE/RMSE/R², significance tests

All ranking functions operate on ONE horizon at a time.
To evaluate all three, call them with y_pred[:,0], y_pred[:,1], y_pred[:,2].

Usage:
    from evaluation.metrics import evaluate, print_report

    results = evaluate(y_pred[:,0], y_true[:,0], meta, label="EXP-002 Y5")
    print_report(results)
"""
from __future__ import annotations

import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Daily Spearman ────────────────────────────────────────────────────────────

def daily_spearman(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    dates:  pd.Series,
    min_stocks: int = 10,
) -> pd.Series:
    """
    Compute Spearman(prediction, actual) for each trading day.

    Parameters
    ----------
    y_pred      : (n,)  model predictions
    y_true      : (n,)  actual forward returns
    dates       : (n,)  date of each sample
    min_stocks  : skip days with fewer than this many stocks

    Returns
    -------
    pd.Series indexed by date, values are Spearman rho.
    """
    dates = pd.to_datetime(dates)
    records = []
    for day, grp in pd.DataFrame({"pred": y_pred, "true": y_true, "date": dates}).groupby("date"):
        if len(grp) < min_stocks:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, _ = stats.spearmanr(grp["pred"], grp["true"])
        if np.isnan(rho):
            continue
        records.append({"date": day, "spearman": rho})
    if not records:
        return pd.Series(dtype=float)
    return pd.DataFrame(records).set_index("date")["spearman"]


def daily_kendall(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    dates:  pd.Series,
    min_stocks: int = 10,
) -> pd.Series:
    """Kendall tau for each trading day."""
    dates = pd.to_datetime(dates)
    records = []
    for day, grp in pd.DataFrame({"pred": y_pred, "true": y_true, "date": dates}).groupby("date"):
        if len(grp) < min_stocks:
            continue
        tau, _ = stats.kendalltau(grp["pred"], grp["true"])
        records.append({"date": day, "kendall": tau})
    if not records:
        return pd.Series(dtype=float)
    return pd.DataFrame(records).set_index("date")["kendall"]


def spearman_stats(series: pd.Series) -> dict:
    """
    Summary statistics for the daily Spearman series.
    Includes t-test (H0: mean = 0) and bootstrap 95% CI.
    """
    _empty = {
        "mean": float("nan"), "median": float("nan"), "std": float("nan"),
        "q25": float("nan"), "q75": float("nan"), "t_stat": float("nan"),
        "p_value": float("nan"), "ci_lo_95": float("nan"), "ci_hi_95": float("nan"),
        "pct_positive": float("nan"), "n_days": 0,
    }
    s = series.dropna()
    if len(s) < 5:
        return _empty

    mean   = float(s.mean())
    median = float(s.median())
    std    = float(s.std())
    q25    = float(s.quantile(0.25))
    q75    = float(s.quantile(0.75))

    t_stat, p_value = stats.ttest_1samp(s, 0.0)

    # bootstrap 95% CI on the mean
    rng        = np.random.default_rng(42)
    boot_means = [rng.choice(s.values, size=len(s), replace=True).mean() for _ in range(2000)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    pct_positive = float((s > 0).mean())

    return {
        "mean":         mean,
        "median":       median,
        "std":          std,
        "q25":          q25,
        "q75":          q75,
        "t_stat":       float(t_stat),
        "p_value":      float(p_value),
        "ci_lo_95":     float(ci_lo),
        "ci_hi_95":     float(ci_hi),
        "pct_positive": pct_positive,
        "n_days":       len(s),
    }


# ── Quintile analysis ─────────────────────────────────────────────────────────

def quintile_analysis(
    y_pred:     np.ndarray,
    y_true:     np.ndarray,
    dates:      pd.Series,
    n_quintiles: int = 5,
    min_stocks:  int = 10,
) -> pd.DataFrame:
    """
    Each day, rank stocks by y_pred into n_quintiles groups.
    Return mean actual return per quintile, averaged across days.

    Returns DataFrame with columns [mean, median, std, n_days]
    indexed by quintile label (Q1=bottom ... Q5=top).
    """
    dates = pd.to_datetime(dates)
    df    = pd.DataFrame({"pred": y_pred, "true": y_true, "date": dates})

    daily_quintile_means: list[dict] = []

    for _, grp in df.groupby("date"):
        if len(grp) < min_stocks:
            continue
        grp = grp.copy()
        try:
            grp["quintile"] = pd.qcut(
                grp["pred"], q=n_quintiles,
                labels=[f"Q{i+1}" for i in range(n_quintiles)],
                duplicates="drop",
            )
        except ValueError:
            continue  # skip day when all predictions are identical
        for q, q_grp in grp.groupby("quintile", observed=True):
            daily_quintile_means.append({"quintile": q, "ret": q_grp["true"].mean()})

    if not daily_quintile_means:
        return pd.DataFrame()

    result = (
        pd.DataFrame(daily_quintile_means)
        .groupby("quintile")["ret"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"count": "n_days"})
    )
    return result


def quintile_spread(q_df: pd.DataFrame) -> float:
    """Q5 mean return minus Q1 mean return."""
    if q_df.empty or "Q5" not in q_df.index or "Q1" not in q_df.index:
        return float("nan")
    return float(q_df.loc["Q5", "mean"] - q_df.loc["Q1", "mean"])


def monotonic_score(q_df: pd.DataFrame) -> float:
    """
    Fraction of adjacent quintile pairs where Q(i+1) > Q(i).
    1.0 = perfectly monotone Q1 < Q2 < Q3 < Q4 < Q5.
    """
    if q_df.empty:
        return float("nan")
    means = q_df["mean"].values
    pairs = len(means) - 1
    if pairs == 0:
        return float("nan")
    return float(sum(means[i + 1] > means[i] for i in range(pairs)) / pairs)


# ── Prediction quality ────────────────────────────────────────────────────────

def prediction_quality(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """MAE, RMSE, R², directional hit rate."""
    mask   = ~(np.isnan(y_pred) | np.isnan(y_true))
    p, t   = y_pred[mask], y_true[mask]
    if len(p) == 0:
        return {}

    mae  = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))

    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    hit = float(np.mean(np.sign(p) == np.sign(t)))

    return {"mae": mae, "rmse": rmse, "r2": r2, "hit_rate": hit, "n": int(mask.sum())}


# ── Full evaluation ───────────────────────────────────────────────────────────

def evaluate(
    y_pred:    np.ndarray,
    y_true:    np.ndarray,
    meta:      pd.DataFrame,
    label:     str = "",
    n_quintiles: int = 5,
) -> dict:
    """
    Run the full evaluation suite for one horizon.

    Parameters
    ----------
    y_pred  : (n,)  predictions
    y_true  : (n,)  actuals
    meta    : DataFrame with columns [date, ticker, sector]
    label   : human-readable name for reports (e.g. 'EXP-002 Y5')

    Returns
    -------
    dict with keys: label, spearman, kendall, quintiles, prediction, sector
    """
    dates = meta["date"]

    sp_series = daily_spearman(y_pred, y_true, dates)
    kd_series = daily_kendall(y_pred,  y_true, dates)
    q_df      = quintile_analysis(y_pred, y_true, dates, n_quintiles)
    pq        = prediction_quality(y_pred, y_true)

    # per-sector Spearman
    sector_sp: dict[str, dict] = {}
    for sector, grp in meta.groupby("sector"):
        idx = grp.index
        s   = daily_spearman(y_pred[idx], y_true[idx], dates.iloc[idx])
        sector_sp[sector] = spearman_stats(s)

    return {
        "label":      label,
        "spearman":   {**spearman_stats(sp_series), "series": sp_series},
        "kendall":    {**spearman_stats(kd_series),  "series": kd_series},
        "quintiles":  {
            "table":       q_df,
            "spread":      quintile_spread(q_df),
            "monotonic":   monotonic_score(q_df),
        },
        "prediction": pq,
        "sector":     sector_sp,
    }


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(results: dict) -> None:
    label = results.get("label", "")
    sp    = results["spearman"]
    q     = results["quintiles"]
    pq    = results["prediction"]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    nan = float("nan")
    print(f"\n  Daily Spearman  (n={sp.get('n_days', 0)} days)")
    print(f"    mean       : {sp.get('mean',      nan):+.4f}")
    print(f"    median     : {sp.get('median',    nan):+.4f}")
    print(f"    std        : {sp.get('std',       nan):.4f}")
    print(f"    [Q25, Q75] : [{sp.get('q25', nan):+.4f}, {sp.get('q75', nan):+.4f}]")
    print(f"    95% CI     : [{sp.get('ci_lo_95', nan):+.4f}, {sp.get('ci_hi_95', nan):+.4f}]")
    print(f"    t-stat     : {sp.get('t_stat',   nan):+.2f}   p={sp.get('p_value', nan):.4f}")
    pct = sp.get('pct_positive', nan)
    print(f"    % positive : {pct:.1%}" if not np.isnan(pct) else "    % positive : nan")

    print(f"\n  Quintile analysis  (spread Q5-Q1 = {q['spread']:+.4f})")
    print(f"    monotone score : {q['monotonic']:.2f}  (1.0 = perfect Q1<Q2<Q3<Q4<Q5)")
    if not q["table"].empty:
        for qname, row in q["table"].iterrows():
            bar = "█" * max(0, int((row["mean"] + 0.5) * 20))
            print(f"    {qname} : {row['mean']:+.4f}  {bar}")

    print(f"\n  Prediction quality")
    print(f"    MAE      : {pq.get('mae',  float('nan')):.4f}")
    print(f"    RMSE     : {pq.get('rmse', float('nan')):.4f}")
    print(f"    R²       : {pq.get('r2',   float('nan')):.4f}")
    print(f"    Hit rate : {pq.get('hit_rate', float('nan')):.2%}")

    if results.get("sector"):
        print(f"\n  Spearman by sector")
        for sector, s in sorted(results["sector"].items()):
            if s:
                print(f"    {sector:<18} mean={s['mean']:+.4f}  p={s['p_value']:.3f}")

    print()


# ── Experiment comparison table (R10) ─────────────────────────────────────────

def compare_experiments(results_list: list[dict]) -> pd.DataFrame:
    """
    Build the R10 comparison table from a list of evaluate() outputs.

    Columns: label, spearman_mean, spearman_p, q5_q1_spread,
             monotonic, hit_rate, mae, rmse, r2
    """
    rows = []
    for r in results_list:
        sp = r.get("spearman", {})
        q  = r.get("quintiles", {})
        pq = r.get("prediction", {})
        rows.append({
            "label":          r.get("label", ""),
            "spearman_mean":  sp.get("mean",         float("nan")),
            "spearman_p":     sp.get("p_value",       float("nan")),
            "ci_lo":          sp.get("ci_lo_95",      float("nan")),
            "ci_hi":          sp.get("ci_hi_95",      float("nan")),
            "pct_positive":   sp.get("pct_positive",  float("nan")),
            "q5_q1_spread":   q.get("spread",         float("nan")),
            "monotonic":      q.get("monotonic",      float("nan")),
            "hit_rate":       pq.get("hit_rate",      float("nan")),
            "mae":            pq.get("mae",           float("nan")),
            "rmse":           pq.get("rmse",          float("nan")),
            "r2":             pq.get("r2",            float("nan")),
        })
    df = pd.DataFrame(rows).set_index("label")
    return df.sort_values("spearman_mean", ascending=False)


# ── Save / load results ───────────────────────────────────────────────────────

def save_results(results: dict, path: Path) -> None:
    """Save scalar metrics to CSV (excludes series and DataFrames)."""
    sp = results.get("spearman", {})
    q  = results.get("quintiles", {})
    pq = results.get("prediction", {})

    row = {
        "label":         results.get("label", ""),
        "spearman_mean": sp.get("mean"),
        "spearman_p":    sp.get("p_value"),
        "ci_lo_95":      sp.get("ci_lo_95"),
        "ci_hi_95":      sp.get("ci_hi_95"),
        "pct_positive":  sp.get("pct_positive"),
        "n_days":        sp.get("n_days"),
        "q5_q1_spread":  q.get("spread"),
        "monotonic":     q.get("monotonic"),
        "hit_rate":      pq.get("hit_rate"),
        "mae":           pq.get("mae"),
        "rmse":          pq.get("rmse"),
        "r2":            pq.get("r2"),
    }
    pd.DataFrame([row]).to_csv(path, index=False)

    # also save the daily Spearman series
    sp_series = sp.get("series")
    if sp_series is not None and len(sp_series):
        sp_series.to_csv(path.parent / (path.stem + "_spearman_series.csv"))

    # quintile table
    q_table = q.get("table")
    if q_table is not None and not q_table.empty:
        q_table.to_csv(path.parent / (path.stem + "_quintiles.csv"))
