import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.db import execute_query
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.dataset_builder")

TRD = 5
TARGET_CLIP_MIN = -3.0
TARGET_CLIP_MAX = 3.0
WINSOR_QUANTILE = 0.005

BASE_FEATURE_COLS = [
    'log_return', 'mom_5', 'mom_10', 'mom_20',
    'vol_10', 'vol_20', 'vol_ratio', 'zscore_20',
    'trend_50', 'vol_regime', 'mom_vol_adj', 'volume_shock'
]

CROSS_SECTIONAL_FEATURE_COLS = [
    'ret_1',
    'x_rank_mom5', 'x_rank_mom20', 'x_rank_trend50',
    'rel_mom5_spy', 'rel_mom20_spy', 'excess_ret_5', 'beta_20',
    'mkt_ret_5', 'mkt_ret_20', 'breadth_20', 'dispersion_20',
]

FEATURE_COLS = BASE_FEATURE_COLS + CROSS_SECTIONAL_FEATURE_COLS


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute base technical features and target for a single instrument sorted by date."""
    df = df.copy()
    df['ret_1'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return'] = np.log(df['close'] / df['close'].shift(TRD))
    df['mom_5'] = df['close'].pct_change(5)
    df['mom_10'] = df['close'].pct_change(10)
    df['mom_20'] = df['close'].pct_change(20)

    df['vol_10'] = df['log_return'].rolling(10).std()
    df['vol_20'] = df['log_return'].rolling(20).std()
    df['vol_ratio'] = df['vol_10'] / df['vol_20']

    ma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['zscore_20'] = (df['close'] - ma_20) / std_20

    ma_50 = df['close'].rolling(50).mean()
    df['trend_50'] = (df['close'] - ma_50) / ma_50

    log_ret_1 = np.log(df['close'] / df['close'].shift(1))
    vol_10_reg = log_ret_1.rolling(10).std()
    vol_60_reg = log_ret_1.rolling(60).std()
    df['vol_regime'] = vol_10_reg / vol_60_reg

    df['mom_vol_adj'] = df['mom_20'] / df['vol_20']

    vol_mean_20 = df['volume'].rolling(20).mean()
    df['volume_shock'] = df['volume'] / vol_mean_20

    rolling_max = df['close'].rolling(252, min_periods=20).max()
    is_high = df['close'] == rolling_max
    positions = np.arange(len(df))
    last_high_pos = pd.Series(np.where(is_high, positions, np.nan)).ffill().fillna(0)
    df['days_since_high'] = np.log1p(positions - last_high_pos.to_numpy())

    fwd_close = df['close'].shift(-TRD)
    df['target'] = np.log(fwd_close / df['close']) / df['vol_20']
    df['target'] = df['target'].clip(TARGET_CLIP_MIN, TARGET_CLIP_MAX)

    return df


def _add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add market regime and cross-sectional (relative) features computed on the full panel."""
    spy = panel[panel['symbol'] == 'SPY'].set_index('trade_date')

    panel['_spy_ret'] = panel['trade_date'].map(spy['ret_1'])
    panel['mkt_mom5'] = panel['trade_date'].map(spy['mom_5'])
    panel['mkt_ret_20'] = panel['trade_date'].map(spy['mom_20'])
    panel['mkt_ret_5'] = panel['trade_date'].map(spy['log_return'])

    panel['rel_mom5_spy'] = panel['mom_5'] - panel['mkt_mom5']
    panel['rel_mom20_spy'] = panel['mom_20'] - panel['mkt_ret_20']

    by_date = panel.groupby('trade_date')
    panel['x_rank_mom5'] = by_date['mom_5'].rank(pct=True)
    panel['x_rank_mom20'] = by_date['mom_20'].rank(pct=True)
    panel['x_rank_trend50'] = by_date['trend_50'].rank(pct=True)
    panel['excess_ret_5'] = panel['log_return'] - by_date['log_return'].transform('mean')
    panel['breadth_20'] = by_date['mom_20'].transform(lambda s: (s > 0).mean())
    panel['dispersion_20'] = by_date['mom_20'].transform('std')

    beta = pd.Series(index=panel.index, dtype=float)
    for symbol, g in panel.groupby('symbol'):
        r = g['ret_1']
        s = g['_spy_ret']
        cov = r.rolling(20, min_periods=10).cov(s)
        var = s.rolling(20, min_periods=10).var()
        beta.loc[g.index] = (cov / var.where(var > 1e-12)).values
    panel['beta_20'] = beta

    dow = panel['trade_date'].dt.dayofweek
    panel['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    panel['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    return panel.drop(columns=['_spy_ret', 'mkt_mom5'], errors='ignore')


def compute_panel_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute base + cross-sectional + regime features for a panel of instruments.

    raw_df needs columns: instrument_id, symbol, trade_date, open, high, low, close, volume.
    Returns a panel (one row per instrument/date) sorted by symbol, trade_date, containing
    all FEATURE_COLS plus symbol, instrument_id and trade_date (and target when computable).
    """
    df = raw_df.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    feature_dfs = []
    for symbol, group in df.groupby('symbol'):
        group = group.sort_values('trade_date').copy()
        group = compute_features(group)
        feature_dfs.append(group)

    panel = pd.concat(feature_dfs, ignore_index=True)
    panel = _add_cross_sectional_features(panel)
    return panel


def build_raw_df() -> pd.DataFrame:
    """Load price rows from the database as a DataFrame."""
    rows = execute_query("""
        SELECT p.instrument_id, i.symbol, p.trade_date, p.open, p.high, p.low, p.close, p.volume
        FROM price_daily p
        JOIN instrument i ON p.instrument_id = i.id
        ORDER BY p.instrument_id, p.trade_date
    """)
    return pd.DataFrame(rows)


def split_by_date(clean_df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15) -> tuple:
    """Chronological split of the panel rows by trade_date.

    Returns (train_mask, val_mask, test_mask, cutoffs) where cutoffs holds the last
    train date and the last validation date, usable later with split_by_cutoffs().
    """
    dates = np.sort(clean_df['trade_date'].unique())
    n = len(dates)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    n_train = n - n_val - n_test

    train_mask = clean_df['trade_date'].isin(dates[:n_train]).values
    val_mask = clean_df['trade_date'].isin(dates[n_train:n_train + n_val]).values
    test_mask = clean_df['trade_date'].isin(dates[n_train + n_val:]).values

    cutoffs = {
        'train_end': pd.to_datetime(dates[n_train - 1]).strftime('%Y-%m-%d'),
        'val_end': pd.to_datetime(dates[n_train + n_val - 1]).strftime('%Y-%m-%d'),
    }
    return train_mask, val_mask, test_mask, cutoffs


def split_by_cutoffs(clean_df: pd.DataFrame, train_end: str, val_end: str) -> tuple:
    """Reconstruct train/val/test masks from stored chronological cutoffs."""
    d = clean_df['trade_date']
    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)
    train_mask = (d <= train_end).values
    val_mask = ((d > train_end) & (d <= val_end)).values
    test_mask = (d > val_end).values
    return train_mask, val_mask, test_mask


def read_model_config(artifact_path) -> dict:
    """Read the model config.json (feature_columns, split, cutoffs...)."""
    p = Path(artifact_path) / "config.json"
    if p.exists():
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    return {}


def build_training_dataset() -> tuple:
    """Build feature+target dataset from the DB.

    Returns (clean_df, feature_cols, mins, maxs, label_min, label_max) where
    clean_df contains symbol, trade_date, winsorized features and target; mins/maxs are the
    per-feature min/max series and label_min/label_max the target extremes.
    """
    raw_df = build_raw_df()
    if raw_df.empty:
        logger.error("No price data found in database for training.")
        return None

    logger.info(f"Loaded {len(raw_df)} price rows. Computing features...")
    panel = compute_panel_features(raw_df)

    clean_df = panel.dropna(subset=FEATURE_COLS + ['target']).copy()
    dropped = len(panel) - len(clean_df)
    logger.info(f"Rows after dropna: {len(clean_df)} (dropped {dropped})")

    if clean_df.empty:
        logger.error("Clean dataset is empty after dropping NaNs.")
        return None

    X = clean_df[FEATURE_COLS]
    lo = X.quantile(WINSOR_QUANTILE)
    hi = X.quantile(1 - WINSOR_QUANTILE)
    clean_df[FEATURE_COLS] = X.clip(lo, hi, axis=1)
    X = clean_df[FEATURE_COLS]

    mins = X.min()
    maxs = X.max()
    label_min = float(clean_df['target'].min())
    label_max = float(clean_df['target'].max())

    return clean_df, FEATURE_COLS, mins, maxs, label_min, label_max
