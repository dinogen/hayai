"""
EXP-001 — Non-ML baseline strategies.

Strategies implemented:
    random        : random ranking (lower bound — should be ≈ 0)
    equal_weight  : all stocks scored identically (degenerate case)
    mom_5d        : rank by 5-day past log return
    mom_20d       : rank by 20-day past log return
    mom_vol_adj   : rank by 20-day return / realized vol (Sharpe-like momentum)

All strategies extract signals from X[:, -1, :] — the last day of the window —
so they operate on information available at prediction time (no leakage).

Feature indices in X (must match features/stock_features.py FEATURE_NAMES):
    0  log_ret_1d
    1  log_ret_5d
    2  log_ret_20d
    3  vol_5d
    4  vol_20d
    5  vol_ratio
    6  open_close
    7  high_close
    8  low_close
    9  high_low
    10 volume_ratio
    11 volume_trend

Usage:
    python -m models.baseline
    python run_experiment.py --exp EXP-001
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    compare_experiments,
    evaluate,
    print_report,
    save_results,
)
from features.stock_features import load_dataset
from settings import DEFAULT_WINDOW, TARGET_HORIZONS
from training.walk_forward import make_folds

# ── Feature index map (must match FEATURE_NAMES in stock_features.py) ────────

_F = {
    "log_ret_1d":    0,
    "log_ret_5d":    1,
    "log_ret_20d":   2,
    "vol_5d":        3,
    "vol_20d":       4,
    "vol_ratio":     5,
    "open_close":    6,
    "high_close":    7,
    "low_close":     8,
    "high_low":      9,
    "volume_ratio":  10,
    "volume_trend":  11,
}

STRATEGIES = ["random", "equal_weight", "mom_5d", "mom_20d", "mom_vol_adj"]


# ── Prediction functions ──────────────────────────────────────────────────────

def predict(X: np.ndarray, strategy: str, seed: int = 42) -> np.ndarray:
    """
    Compute a ranking score for each sample using the chosen strategy.

    Parameters
    ----------
    X        : (n, window, n_features)
    strategy : one of STRATEGIES
    seed     : used only by 'random'

    Returns
    -------
    y_pred : (n,)  — higher = more bullish prediction
    """
    last = X[:, -1, :]   # (n, n_features) — last day of the window

    if strategy == "random":
        rng = np.random.default_rng(seed)
        return rng.standard_normal(len(X))

    if strategy == "equal_weight":
        return np.zeros(len(X))

    if strategy == "mom_5d":
        return last[:, _F["log_ret_5d"]]

    if strategy == "mom_20d":
        return last[:, _F["log_ret_20d"]]

    if strategy == "mom_vol_adj":
        mom = last[:, _F["log_ret_20d"]]
        vol = last[:, _F["vol_20d"]]
        return mom / (vol + 1e-8)

    raise ValueError(f"Unknown strategy: {strategy!r}. Choose from {STRATEGIES}")


# ── Walk-forward evaluation ───────────────────────────────────────────────────

def run_baseline(exp: dict, out_dir: Path) -> None:
    """
    Entry point called by run_experiment.py for EXP-001.

    Evaluates all baseline strategies across all walk-forward folds,
    saves per-strategy results and the R10 comparison table.
    """
    window   = exp.get("window") or DEFAULT_WINDOW
    horizons = exp.get("targets", TARGET_HORIZONS)
    seed     = exp.get("seed", 42)
    variants = exp.get("variants", STRATEGIES)

    print(f"Loading dataset (window={window})...")
    X, y, meta = load_dataset(window=window)

    folds, _ = make_folds(meta, verbose=True)
    if not folds:
        print("No folds generated.")
        return

    # collect per-strategy results across all folds
    # key: strategy → list of (y_pred, y_true, meta) per fold
    strategy_preds: dict[str, list[tuple]] = {s: [] for s in variants}

    for fold in folds:
        X_test    = X[fold.test_idx]
        meta_test = meta.iloc[fold.test_idx].reset_index(drop=True)

        for strategy in variants:
            y_pred = predict(X_test, strategy, seed=seed)
            strategy_preds[strategy].append((y_pred, fold.test_idx, meta_test))

    # evaluate each strategy, for each horizon
    all_results = []

    for strategy in variants:
        print(f"\n{'─'*50}")
        print(f"  Strategy: {strategy}")
        print(f"{'─'*50}")

        # concatenate across folds
        preds_list = strategy_preds[strategy]
        y_pred_all = np.concatenate([p[0]                     for p in preds_list])
        idx_all    = np.concatenate([p[1]                     for p in preds_list])
        meta_all   = pd.concat(    [p[2] for p in preds_list], ignore_index=True)

        for h_idx, h in enumerate(horizons):
            y_true_all = y[idx_all, h_idx]
            label      = f"{exp['id']} {strategy} Y{h}"

            results = evaluate(y_pred_all, y_true_all, meta_all, label=label)
            print_report(results)

            csv_path = out_dir / f"{strategy}_y{h}.csv"
            save_results(results, csv_path)
            all_results.append(results)

    # R10 comparison table
    print(f"\n{'='*70}")
    print("  R10 — Experiment comparison table")
    print(f"{'='*70}\n")
    comparison = compare_experiments(all_results)
    print(comparison[["spearman_mean", "spearman_p", "q5_q1_spread", "monotonic", "hit_rate"]].to_string())

    comparison.to_csv(out_dir / "comparison_r10.csv")
    print(f"\nResults saved to {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EXP-001 baseline strategies")
    parser.add_argument("--window",   type=int,  default=DEFAULT_WINDOW)
    parser.add_argument("--strategy", type=str,  default=None,
                        help=f"Single strategy to run. Options: {STRATEGIES}")
    parser.add_argument("--horizon",  type=int,  default=5,
                        help="Target horizon to evaluate (5, 10, or 15)")
    parser.add_argument("--seed",     type=int,  default=42)
    args = parser.parse_args()

    X, y, meta = load_dataset(window=args.window)
    folds, _   = make_folds(meta, verbose=False)

    if not folds:
        print("No folds generated.")
        sys.exit(1)

    strategies = [args.strategy] if args.strategy else STRATEGIES
    h_idx      = TARGET_HORIZONS.index(args.horizon) if args.horizon in TARGET_HORIZONS else 0

    preds_list: dict[str, list] = {s: [] for s in strategies}
    for fold in folds:
        X_test    = X[fold.test_idx]
        meta_test = meta.iloc[fold.test_idx].reset_index(drop=True)
        for s in strategies:
            preds_list[s].append((predict(X_test, s, seed=args.seed), fold.test_idx, meta_test))

    all_results = []
    for s in strategies:
        y_pred_all = np.concatenate([p[0] for p in preds_list[s]])
        idx_all    = np.concatenate([p[1] for p in preds_list[s]])
        meta_all   = pd.concat([p[2]      for p in preds_list[s]], ignore_index=True)
        y_true_all = y[idx_all, h_idx]

        label   = f"baseline {s} Y{args.horizon}"
        results = evaluate(y_pred_all, y_true_all, meta_all, label=label)
        print_report(results)
        all_results.append(results)

    if len(all_results) > 1:
        print("\n── Comparison ──────────────────────────────────────────────────\n")
        df = compare_experiments(all_results)
        print(df[["spearman_mean", "spearman_p", "q5_q1_spread", "hit_rate"]].to_string())
