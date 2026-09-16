"""
Walk-forward cross-validation splitter with purging/embargo.

Why purging matters:
    Targets Y5/Y10/Y15 are overlapping windows. A sample at date t with
    target Y15 uses close prices up to t+15. If training ends at T and
    validation starts at T+1, the last training samples are correlated
    with the first validation samples. The embargo removes training
    samples whose forward window bleeds into the validation period.

Fold structure:
    ┌─────────────────┬──────┬───────────┬──────┬──────────┬──────┬─────────┐
    │   TRAIN         │ EMB  │ VALIDATION │ EMB  │  TEST    │ ...  │ HOLDOUT │
    └─────────────────┴──────┴───────────┴──────┴──────────┴──────┴─────────┘

Rolling mode:  training window is fixed size, slides forward
Expanding mode: training window grows (includes all past data)

Usage:
    from training.walk_forward import make_folds, describe_folds

    folds, holdout_idx = make_folds(meta)
    describe_folds(folds, holdout_idx, meta)

    for fold in folds:
        X_train, y_train = X[fold.train_idx], y[fold.train_idx]
        X_val,   y_val   = X[fold.val_idx],   y[fold.val_idx]
        X_test,  y_test  = X[fold.test_idx],  y[fold.test_idx]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay, DateOffset

sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import (
    WF_EMBARGO_DAYS,
    WF_TEST_MONTHS,
    WF_TRAIN_YEARS,
    WF_VAL_MONTHS,
)

HOLDOUT_MONTHS = 12   # last N months reserved for blind final test (EXP-016)
MIN_SPLIT_SAMPLES = 50  # warn if any split has fewer samples than this


# ── Fold dataclass ────────────────────────────────────────────────────────────

@dataclass
class Fold:
    fold_id:     int
    train_idx:   np.ndarray
    val_idx:     np.ndarray
    test_idx:    np.ndarray
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    val_start:   pd.Timestamp
    val_end:     pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_id:02d}  "
            f"train {self.train_start.date()} → {self.train_end.date()}  "
            f"val {self.val_start.date()} → {self.val_end.date()}  "
            f"test {self.test_start.date()} → {self.test_end.date()}  "
            f"[{len(self.train_idx):,} / {len(self.val_idx):,} / {len(self.test_idx):,}]"
        )

    @property
    def sizes(self) -> tuple[int, int, int]:
        return len(self.train_idx), len(self.val_idx), len(self.test_idx)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _idx_between(
    dates: pd.Series,
    start: pd.Timestamp,
    end:   pd.Timestamp,
) -> np.ndarray:
    """Indices of samples where start <= date < end."""
    return np.where((dates >= start) & (dates < end))[0]


# ── Main splitter ─────────────────────────────────────────────────────────────

def make_folds(
    meta: pd.DataFrame,
    train_years:    int   = WF_TRAIN_YEARS,
    val_months:     int   = WF_VAL_MONTHS,
    test_months:    int   = WF_TEST_MONTHS,
    embargo_days:   int   = WF_EMBARGO_DAYS,
    mode: Literal["rolling", "expanding"] = "rolling",
    holdout_months: int   = HOLDOUT_MONTHS,
    verbose:        bool  = True,
) -> tuple[list[Fold], np.ndarray]:
    """
    Generate walk-forward folds from a meta DataFrame.

    Parameters
    ----------
    meta            : DataFrame with 'date' column (one row per sample)
    train_years     : length of each training window in years
    val_months      : length of each validation window in months
    test_months     : length of each test window in months (also the step size)
    embargo_days    : trading-day gap inserted between train/val and val/test
    mode            : 'rolling' = fixed-size train; 'expanding' = growing train
    holdout_months  : months reserved at the end for the blind final test
    verbose         : print fold summary

    Returns
    -------
    folds       : list of Fold objects (train/val/test indices)
    holdout_idx : indices of samples in the final holdout period
    """
    dates = pd.to_datetime(meta["date"])
    date_min = dates.min()
    date_max = dates.max()

    # ── reserve final holdout ────────────────────────────────────────────────
    if holdout_months > 0:
        holdout_start = date_max - DateOffset(months=holdout_months)
        holdout_idx   = _idx_between(dates, holdout_start, date_max + DateOffset(days=1))
        available_end = holdout_start
    else:
        holdout_idx   = np.array([], dtype=int)
        available_end = date_max + DateOffset(days=1)

    # ── generate folds ───────────────────────────────────────────────────────
    folds: list[Fold] = []
    origin_start = date_min   # anchor for expanding mode

    # first training window starts at date_min
    fold_train_start = date_min

    fold_id = 0
    while True:
        train_start = fold_train_start if mode == "rolling" else origin_start
        train_end   = fold_train_start + DateOffset(years=train_years)

        emb1_end    = train_end   + BDay(embargo_days)
        val_start   = emb1_end
        val_end     = val_start   + DateOffset(months=val_months)

        emb2_end    = val_end     + BDay(embargo_days)
        test_start  = emb2_end
        test_end    = test_start  + DateOffset(months=test_months)

        # stop when test period exceeds available data (before holdout)
        if test_end > available_end:
            break

        train_idx = _idx_between(dates, train_start, train_end)
        val_idx   = _idx_between(dates, val_start,   val_end)
        test_idx  = _idx_between(dates, test_start,  test_end)

        # skip degenerate folds
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            fold_train_start += DateOffset(months=test_months)
            continue

        folds.append(Fold(
            fold_id     = fold_id,
            train_idx   = train_idx,
            val_idx     = val_idx,
            test_idx    = test_idx,
            train_start = pd.Timestamp(train_start),
            train_end   = pd.Timestamp(train_end),
            val_start   = pd.Timestamp(val_start),
            val_end     = pd.Timestamp(val_end),
            test_start  = pd.Timestamp(test_start),
            test_end    = pd.Timestamp(test_end),
        ))

        fold_id += 1
        # slide forward by one test window
        fold_train_start += DateOffset(months=test_months)

    if verbose:
        describe_folds(folds, holdout_idx, meta)

    return folds, holdout_idx


# ── Summary ───────────────────────────────────────────────────────────────────

def describe_folds(
    folds:       list[Fold],
    holdout_idx: np.ndarray,
    meta:        pd.DataFrame,
) -> None:
    dates = pd.to_datetime(meta["date"])

    print(f"\n{'='*72}")
    print(f"  Walk-forward folds  ({len(folds)} folds)")
    print(f"{'='*72}")
    print(f"  {'Fold':>5}  {'Train':^24}  {'Val':^15}  {'Test':^15}  "
          f"{'Tr':>7}  {'Va':>6}  {'Te':>6}")
    print(f"  {'-'*5}  {'-'*24}  {'-'*15}  {'-'*15}  "
          f"  {'-'*6}  {'-'*6}  {'-'*6}")

    for f in folds:
        n_tr, n_va, n_te = f.sizes
        warn = " !" if min(n_va, n_te) < MIN_SPLIT_SAMPLES else ""
        print(
            f"  {f.fold_id:>5}  "
            f"{str(f.train_start.date()):>12}→{str(f.train_end.date()):<12}  "
            f"{str(f.val_start.date()):>7}→{str(f.val_end.date()):<7}  "
            f"{str(f.test_start.date()):>7}→{str(f.test_end.date()):<7}  "
            f"{n_tr:>7,}  {n_va:>6,}  {n_te:>6,}{warn}"
        )

    if len(holdout_idx):
        h_start = dates.iloc[holdout_idx].min().date()
        h_end   = dates.iloc[holdout_idx].max().date()
        print(f"\n  HOLDOUT  {h_start} → {h_end}  ({len(holdout_idx):,} samples)")
        print(f"  ** Never use holdout for model selection — EXP-016 only **")

    print(f"{'='*72}\n")


def ascii_timeline(folds: list[Fold], holdout_idx: np.ndarray, meta: pd.DataFrame, width: int = 72) -> str:
    """Return a compact ASCII visualization of fold coverage."""
    dates   = pd.to_datetime(meta["date"])
    d_min   = dates.min()
    d_max   = dates.max()
    span    = (d_max - d_min).days or 1

    def pos(d: pd.Timestamp) -> int:
        return int((d - d_min).days / span * (width - 2))

    lines = []
    lines.append(f"  {d_min.date()} {'':>{width-22}} {d_max.date()}")
    lines.append("  |" + " " * (width - 4) + "|")

    for f in folds:
        row = [" "] * width
        for d_s, d_e, ch in [
            (f.train_start, f.train_end,  "T"),
            (f.val_start,   f.val_end,    "V"),
            (f.test_start,  f.test_end,   "E"),
        ]:
            for x in range(pos(d_s), min(pos(d_e), width - 1)):
                row[x] = ch
        lines.append(f"  {''.join(row)}  fold {f.fold_id:02d}")

    if len(holdout_idx):
        row   = [" "] * width
        h_s   = dates.iloc[holdout_idx].min()
        h_e   = dates.iloc[holdout_idx].max()
        for x in range(pos(h_s), min(pos(h_e), width - 1)):
            row[x] = "H"
        lines.append(f"  {''.join(row)}  HOLDOUT")

    lines.append("  |" + " " * (width - 4) + "|")
    lines.append("  T=train  V=val  E=test  H=holdout")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from features.stock_features import load_dataset
    from settings import DEFAULT_WINDOW

    parser = argparse.ArgumentParser(description="Show walk-forward fold structure")
    parser.add_argument("--window",   type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--mode",     type=str, default="rolling", choices=["rolling", "expanding"])
    parser.add_argument("--holdout",  type=int, default=HOLDOUT_MONTHS, metavar="MONTHS")
    parser.add_argument("--train",    type=int, default=WF_TRAIN_YEARS,  metavar="YEARS")
    parser.add_argument("--val",      type=int, default=WF_VAL_MONTHS,   metavar="MONTHS")
    parser.add_argument("--test",     type=int, default=WF_TEST_MONTHS,  metavar="MONTHS")
    parser.add_argument("--embargo",  type=int, default=WF_EMBARGO_DAYS, metavar="DAYS")
    parser.add_argument("--timeline", action="store_true", help="Print ASCII timeline")
    args = parser.parse_args()

    _, _, meta = load_dataset(window=args.window)

    folds, holdout_idx = make_folds(
        meta,
        train_years    = args.train,
        val_months     = args.val,
        test_months    = args.test,
        embargo_days   = args.embargo,
        mode           = args.mode,
        holdout_months = args.holdout,
        verbose        = True,
    )

    if args.timeline:
        print(ascii_timeline(folds, holdout_idx, meta))
