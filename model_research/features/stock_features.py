"""
Compute per-stock time-series features and forward-return targets.

Feature set — 12 features per trading day:
    log_ret_1d    log(close_t / close_{t-1})
    log_ret_5d    log(close_t / close_{t-5})
    log_ret_20d   log(close_t / close_{t-20})
    vol_5d        5-day realized volatility (annualized)
    vol_20d       20-day realized volatility (annualized)
    vol_ratio     vol_5d / vol_20d  — short/long vol regime
    open_close    (open - close) / close
    high_close    (high - close) / close
    low_close     (low  - close) / close
    high_low      (high - low)   / close
    volume_ratio  volume / volume_ma20
    volume_trend  volume_ma5 / volume_ma20

Targets — 3 normalized forward returns:
    y5    log(close_{t+5}  / close_t) / vol_20d,  clipped to [-3, +3]
    y10   log(close_{t+10} / close_t) / vol_20d,  clipped to [-3, +3]
    y15   log(close_{t+15} / close_t) / vol_20d,  clipped to [-3, +3]

Output shapes:
    X    (n_samples, window, 12)    — float32
    y    (n_samples, 3)             — float32
    meta DataFrame [date, ticker, sector]

Usage:
    python -m features.stock_features              # build and save
    python -m features.stock_features --window 10
    python -m features.stock_features --load       # describe saved dataset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.downloader import get, load_universe
from settings import DEFAULT_WINDOW, FEATURES_DIR, TARGET_CLIP, TARGET_HORIZONS

# ── Feature names (order must match _compute_features output) ────────────────

FEATURE_NAMES = [
    "log_ret_1d",
    "log_ret_5d",
    "log_ret_20d",
    "vol_5d",
    "vol_20d",
    "vol_ratio",
    "open_close",
    "high_close",
    "low_close",
    "high_low",
    "volume_ratio",
    "volume_trend",
]

TARGET_NAMES = ["y5", "y10", "y15"]


# ── Per-stock feature computation ─────────────────────────────────────────────

def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 12 features on the full OHLCV history.
    All values are dimensionless. NaN at series edges where rolling
    windows have insufficient history.
    """
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    vol   = df["volume"].astype(float).replace(0, np.nan)

    feat   = pd.DataFrame(index=df.index)
    log_c  = np.log(close.clip(lower=1e-8))
    eps    = close.abs().clip(lower=1e-8) * 1e-6 + 1e-8

    # returns
    feat["log_ret_1d"]  = log_c.diff(1)
    feat["log_ret_5d"]  = log_c.diff(5)
    feat["log_ret_20d"] = log_c.diff(20)

    # volatility
    feat["vol_5d"]    = feat["log_ret_1d"].rolling(5,  min_periods=5).std()  * np.sqrt(252)
    feat["vol_20d"]   = feat["log_ret_1d"].rolling(20, min_periods=20).std() * np.sqrt(252)
    feat["vol_ratio"] = feat["vol_5d"] / (feat["vol_20d"] + 1e-8)

    # intraday structure
    feat["open_close"] = (open_ - close) / (close + eps)
    feat["high_close"] = (high  - close) / (close + eps)
    feat["low_close"]  = (low   - close) / (close + eps)
    feat["high_low"]   = (high  - low)   / (close + eps)

    # volume
    vol_ma20 = vol.rolling(20, min_periods=10).mean()
    vol_ma5  = vol.rolling(5,  min_periods=3).mean()
    feat["volume_ratio"] = vol / (vol_ma20 + 1.0)
    feat["volume_trend"] = vol_ma5 / (vol_ma20 + 1.0)

    return feat[FEATURE_NAMES]


def _compute_targets(
    close: pd.Series,
    vol_20d: pd.Series,
    horizons: list[int],
    clip: float,
    market_close: "pd.Series | None" = None,
) -> pd.DataFrame:
    """
    Forward log returns normalized by realized volatility, clipped to [-clip, +clip].
    If market_close is provided, computes excess return over the market (alpha).
    NaN at the last max(horizons) rows where future prices are unavailable.
    """
    log_c = np.log(close.clip(lower=1e-8))
    tgt   = pd.DataFrame(index=close.index)

    if market_close is not None:
        log_m = np.log(market_close.reindex(close.index, method="ffill").clip(lower=1e-8))
    else:
        log_m = None

    for h in horizons:
        raw = log_c.shift(-h) - log_c
        if log_m is not None:
            raw = raw - (log_m.shift(-h) - log_m)
        normalized = raw / (vol_20d + 1e-8)
        tgt[f"y{h}"] = normalized.clip(-clip, clip)
    return tgt


# ── Window builder ────────────────────────────────────────────────────────────

def _extract_windows(
    feature_arr: np.ndarray,   # (T, n_features)
    target_arr:  np.ndarray,   # (T, n_targets)
    window: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Slide a window of length `window` across the time axis.
    Returns X (n, window, f), y (n, t), and list of end indices.

    Skips windows that contain any NaN in features or targets.
    After ffill + dropna on OHLCV the only NaNs should be at the
    series edges, so internal gaps are not expected.
    """
    X_list, y_list, idx_list = [], [], []
    for i in range(window - 1, len(feature_arr)):
        w = feature_arr[i - window + 1 : i + 1]
        t = target_arr[i]
        if np.any(np.isnan(w)) or np.any(np.isnan(t)):
            continue
        X_list.append(w)
        y_list.append(t)
        idx_list.append(i)
    if not X_list:
        return np.empty((0, window, feature_arr.shape[1])), np.empty((0, target_arr.shape[1])), []
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        idx_list,
    )


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(
    window:          int                  = DEFAULT_WINDOW,
    horizons:        list[int]            = TARGET_HORIZONS,
    clip:            float                = TARGET_CLIP,
    verbose:         bool                 = True,
    macro_df:        "pd.DataFrame | None" = None,
    market_ticker:   "str | None"          = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build the full (X, y, meta) dataset for all stocks in universe.yaml.

    valid_from is respected: samples before a ticker's IPO date are excluded.

    Parameters
    ----------
    macro_df : optional date-indexed DataFrame of macro features.
               When provided, macro features are appended to stock features
               along the feature axis: X shape becomes (n, window, 12+n_macro).
    market_ticker : optional ticker (e.g. "^GSPC") to compute market-relative
                    targets (excess return over market benchmark).

    Returns
    -------
    X    : float32  (n_samples, window, n_features)
    y    : float32  (n_samples, len(horizons))
    meta : DataFrame  columns=[date, ticker, sector]
    """
    universe   = load_universe()
    stock_info = {
        str(s["ticker"]): {
            "sector":     s.get("sector", ""),
            "valid_from": pd.Timestamp(s.get("valid_from", "2000-01-01")),
        }
        for s in universe["stocks"]
    }

    market_close: pd.Series | None = None
    if market_ticker is not None:
        mkt_df = get(market_ticker)
        market_close = mkt_df["close"].astype(float).ffill()
        if verbose:
            print(f"  Market benchmark: {market_ticker}  ({len(market_close):,} days)")

    all_X:    list[np.ndarray]   = []
    all_y:    list[np.ndarray]   = []
    all_meta: list[pd.DataFrame] = []
    skipped = 0

    tickers  = list(stock_info.keys())
    iterator = tqdm(tickers, desc="Building features", ncols=80) if verbose else tickers

    for ticker in iterator:
        info = stock_info[ticker]
        try:
            df = get(ticker)
        except Exception as exc:
            if verbose:
                tqdm.write(f"  skip {ticker}: {exc}")
            skipped += 1
            continue

        if df.empty or "close" not in df.columns:
            skipped += 1
            continue

        # forward-fill small gaps (trading halts, public holidays)
        df = df.ffill()

        feat = _compute_features(df)
        tgt  = _compute_targets(df["close"], feat["vol_20d"], horizons, clip,
                                market_close=market_close)

        # align on shared index, drop any remaining NaN rows
        combined = feat.join(tgt).dropna()

        # apply valid_from — exclude pre-IPO samples
        combined = combined[combined.index >= info["valid_from"]]

        if len(combined) < window + max(horizons):
            if verbose:
                tqdm.write(f"  skip {ticker}: only {len(combined)} usable rows")
            skipped += 1
            continue

        feat_arr = combined[FEATURE_NAMES].values.astype(np.float32)
        tgt_arr  = combined[[f"y{h}" for h in horizons]].values.astype(np.float32)

        # append macro features aligned by date
        if macro_df is not None:
            macro_aligned = (
                macro_df
                .reindex(combined.index, method="ffill")
                .fillna(0.0)
                .values
                .astype(np.float32)
            )
            feat_arr = np.concatenate([feat_arr, macro_aligned], axis=1)

        X_t, y_t, idx_list = _extract_windows(feat_arr, tgt_arr, window)

        if X_t.shape[0] == 0:
            skipped += 1
            continue

        dates = [combined.index[i] for i in idx_list]

        all_X.append(X_t)
        all_y.append(y_t)
        all_meta.append(pd.DataFrame({
            "date":   dates,
            "ticker": ticker,
            "sector": info["sector"],
        }))

    if not all_X:
        raise RuntimeError(
            "No samples built. Run 'python -m data.downloader' first."
        )

    X    = np.concatenate(all_X,    axis=0)
    y    = np.concatenate(all_y,    axis=0)
    meta = pd.concat(all_meta, ignore_index=True)
    meta["date"] = pd.to_datetime(meta["date"])

    if verbose:
        n = X.shape[0]
        print(f"\nDataset built:")
        print(f"  Samples  : {n:,}")
        print(f"  X shape  : {X.shape}  (samples × window × features)")
        print(f"  y shape  : {y.shape}  (samples × targets)")
        print(f"  Skipped  : {skipped} tickers")
        print(f"  Dates    : {meta['date'].min().date()} → {meta['date'].max().date()}")
        print(f"  Sectors  :\n{meta['sector'].value_counts().to_string()}")
        print(f"\n  Target stats:")
        y_df = pd.DataFrame(y, columns=[f"y{h}" for h in horizons])
        print(y_df.describe().round(4).to_string())

    return X, y, meta


# ── Save / load ───────────────────────────────────────────────────────────────

def save_dataset(
    X:      np.ndarray,
    y:      np.ndarray,
    meta:   pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    tag:    str = "",
) -> Path:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    stem     = f"dataset_w{window}" + (f"_{tag}" if tag else "")
    npz_path = FEATURES_DIR / f"{stem}.npz"
    meta_path = FEATURES_DIR / f"{stem}_meta.parquet"

    np.savez_compressed(str(npz_path), X=X, y=y)
    meta.to_parquet(meta_path)

    size_mb = npz_path.stat().st_size / 1024 / 1024
    print(f"Saved {npz_path.name}  ({size_mb:.1f} MB)")
    print(f"Saved {meta_path.name}")
    return npz_path


def load_dataset(
    window: int = DEFAULT_WINDOW,
    tag:    str = "",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    stem      = f"dataset_w{window}" + (f"_{tag}" if tag else "")
    npz_path  = FEATURES_DIR / f"{stem}.npz"
    meta_path = FEATURES_DIR / f"{stem}_meta.parquet"

    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run 'python -m features.stock_features' first."
        )

    data = np.load(str(npz_path))
    meta = pd.read_parquet(meta_path)
    meta["date"] = pd.to_datetime(meta["date"])
    return data["X"], data["y"], meta


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build stock feature dataset")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--tag",    type=str, default="")
    parser.add_argument("--macro",    type=str, default=None,
                        choices=["indices", "idxcommod", "fullmacro"],
                        help="Include macro features")
    parser.add_argument("--relative", action="store_true",
                        help="Use market-relative targets (excess return vs ^GSPC)")
    parser.add_argument("--load",     action="store_true")
    args = parser.parse_args()

    parts = []
    if args.macro:     parts.append(args.macro)
    if args.relative:  parts.append("rel")
    tag = args.tag or "_".join(parts)

    if args.load:
        X, y, meta = load_dataset(window=args.window, tag=tag)
        print(f"Loaded: X={X.shape}  y={y.shape}  meta={meta.shape}")
        print(f"Dates : {meta['date'].min().date()} → {meta['date'].max().date()}")
        print(meta["sector"].value_counts().to_string())
    else:
        macro_df = None
        if args.macro:
            from features.macro_features import build_macro_matrix, _get_tickers_by_set
            tickers  = _get_tickers_by_set(args.macro)
            macro_df = build_macro_matrix(tickers, verbose=True)
        market_ticker = "^GSPC" if args.relative else None
        X, y, meta = build_dataset(window=args.window, macro_df=macro_df,
                                   market_ticker=market_ticker)
        save_dataset(X, y, meta, window=args.window, tag=tag)
