"""
Compute macro (market-level) features for the instruments in universe.yaml.

5 features per instrument:
    log_ret_1d    log(close_t / close_{t-1})
    log_ret_5d    log(close_t / close_{t-5})
    log_ret_20d   log(close_t / close_{t-20})
    vol_20d       20-day realized volatility (annualized)
    vol_ratio     vol_5d / vol_20d  — short/long vol regime

The output is a date-indexed DataFrame:
    columns = ["^GSPC_log_ret_1d", "^GSPC_log_ret_5d", ...]
    index   = DatetimeIndex of trading days

This DataFrame is then merged into each stock's feature array when
building a dataset with macro features (see stock_features.build_dataset).

Feature sets:
    indices       6 equity indices (EXP-005)
    idxcommod    +4 commodities   (EXP-006)
    fullmacro    +rates + FX      (EXP-007)

Usage:
    python -m features.macro_features                         # build indices
    python -m features.macro_features --set idxcommod
    python -m features.macro_features --set fullmacro
    python -m features.macro_features --set fullmacro --load  # describe saved
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.downloader import get, load_universe
from settings import FEATURES_DIR

# ── Feature names (per instrument) ────────────────────────────────────────────

MACRO_FEAT_NAMES = ["log_ret_1d", "log_ret_5d", "log_ret_20d", "vol_20d", "vol_ratio"]

# ── Feature-set definitions ───────────────────────────────────────────────────

def _get_tickers_by_set(feature_set: str) -> list[str]:
    """Return list of macro tickers for a given feature set name."""
    uni   = load_universe()
    macro = uni.get("macro", {})
    indices    = [str(t) for t in macro.get("indices",     [])]
    commodities = [str(t) for t in macro.get("commodities", [])]
    rates      = [str(t) for t in macro.get("rates",       [])]
    fx         = [str(t) for t in macro.get("fx",          [])]

    if feature_set == "indices":
        return indices
    if feature_set == "idxcommod":
        return indices + commodities
    if feature_set == "fullmacro":
        return indices + commodities + rates + fx
    raise ValueError(f"Unknown feature set: {feature_set!r}. "
                     f"Use 'indices', 'idxcommod', or 'fullmacro'.")


# Registry: features field in registry.yaml → (macro tickers set, dataset tag)
FEATURES_REGISTRY: dict[str, tuple[str | None, str]] = {
    "stock_only":                     (None,         ""),
    "stock_only_rel":                 (None,         "rel"),
    "stock_plus_indices":             ("indices",    "indices"),
    "stock_plus_indices_commodities": ("idxcommod",  "idxcommod"),
    "full_macro":                     ("fullmacro",  "fullmacro"),
}


def get_macro_config(features_field: str) -> tuple[list[str] | None, str]:
    """
    Returns (macro_tickers, dataset_tag) for a given registry features field.
    Used by run_dnn / run_cnn / run_gru to decide which dataset to load.
    """
    if features_field not in FEATURES_REGISTRY:
        raise ValueError(
            f"Unknown features field: {features_field!r}. "
            f"Valid: {list(FEATURES_REGISTRY)}"
        )
    set_name, tag = FEATURES_REGISTRY[features_field]
    if set_name is None:
        return None, ""
    tickers = _get_tickers_by_set(set_name)
    return tickers, tag


# ── Per-instrument feature computation ────────────────────────────────────────

def _compute_instrument_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """5 features for one macro instrument. Columns prefixed with ticker."""
    close  = df["close"].astype(float)
    log_c  = np.log(close.clip(lower=1e-8))
    lr1d   = log_c.diff(1)

    feat = pd.DataFrame(index=df.index)
    feat["log_ret_1d"]  = lr1d
    feat["log_ret_5d"]  = log_c.diff(5)
    feat["log_ret_20d"] = log_c.diff(20)
    feat["vol_20d"]     = lr1d.rolling(20, min_periods=20).std() * np.sqrt(252)
    vol_5d              = lr1d.rolling(5,  min_periods=5).std()  * np.sqrt(252)
    feat["vol_ratio"]   = vol_5d / (feat["vol_20d"] + 1e-8)

    feat.columns = [f"{ticker}_{c}" for c in MACRO_FEAT_NAMES]
    return feat


# ── Matrix builder ─────────────────────────────────────────────────────────────

def build_macro_matrix(tickers: list[str], verbose: bool = True) -> pd.DataFrame:
    """
    Build a date-indexed macro feature matrix for the given tickers.

    Returns
    -------
    DataFrame  index=DatetimeIndex, columns=[ticker_feature, ...]
               Forward-filled across index to align different trading calendars.
               Any remaining NaN (beginning of series) filled with 0.
    """
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        try:
            df = get(ticker)
        except Exception as exc:
            print(f"  Warning: could not load {ticker}: {exc}")
            continue
        if df.empty or "close" not in df.columns:
            print(f"  Warning: empty data for {ticker}")
            continue
        feat = _compute_instrument_features(df.ffill(), ticker)
        frames.append(feat)

    if not frames:
        raise RuntimeError("No macro instruments loaded.")

    macro_df = pd.concat(frames, axis=1).sort_index()
    macro_df = macro_df.ffill().fillna(0.0)

    if verbose:
        print(f"  Macro matrix: {macro_df.shape[0]:,} days × {macro_df.shape[1]} features")
        print(f"  Date range  : {macro_df.index.min().date()} → {macro_df.index.max().date()}")
        print(f"  Instruments : {tickers}")

    return macro_df


# ── Save / load ────────────────────────────────────────────────────────────────

def save_macro_matrix(macro_df: pd.DataFrame, tag: str) -> Path:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURES_DIR / f"macro_{tag}.parquet"
    macro_df.to_parquet(path)
    print(f"Saved {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    return path


def load_macro_matrix(tag: str) -> pd.DataFrame:
    path = FEATURES_DIR / f"macro_{tag}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python -m features.macro_features --set {tag}' first."
        )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build macro feature matrix")
    parser.add_argument(
        "--set", type=str, default="indices",
        choices=["indices", "idxcommod", "fullmacro"],
        help="Which macro feature set to build",
    )
    parser.add_argument("--load", action="store_true", help="Describe saved matrix")
    args = parser.parse_args()

    if args.load:
        df = load_macro_matrix(args.set)
        print(f"Loaded macro_{args.set}: {df.shape}")
        print(df.tail(3).to_string())
    else:
        tickers = _get_tickers_by_set(args.set)
        print(f"\nBuilding macro matrix for set '{args.set}':")
        print(f"  Tickers: {tickers}")
        macro_df = build_macro_matrix(tickers, verbose=True)
        save_macro_matrix(macro_df, args.set)
