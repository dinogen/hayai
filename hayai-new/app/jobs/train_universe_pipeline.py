import os
import time
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

from app.db import get_db_connection, execute_query
from app.jobs.cache import load_cached, save_cached
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.train_universe_pipeline")

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
    # ETFs (Broad, Sector, Bonds, Commodities)
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", 
    "XLK", "U", "XLRE", "TLT", "IEF", "GLD", "SLV", "USO", "VNQ", "ARKK",
    # International & Indices / Rates
    "EEM", "EFA", "EWJ", "FXI", "ASHR", "VGK", "EWZ", "INDA", "VWO", "IAU"
]

def seed_universe():
    logger.info(f"Seeding {len(UNIVERSE_SYMBOLS)} symbols into database...")
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Ensure portfolio 'main' exists
            cursor.execute("SELECT id FROM portfolio WHERE code = 'main'")
            row = cursor.fetchone()
            if not row:
                cursor.execute("INSERT INTO portfolio (code, name, cash_balance) VALUES ('main', 'Main Portfolio', 5000.00)")
                portfolio_id = cursor.lastrowid
            else:
                portfolio_id = row['id']

            for symbol in UNIVERSE_SYMBOLS:
                # Insert instrument if not exists
                cursor.execute("SELECT id FROM instrument WHERE symbol = %s", (symbol,))
                ins_row = cursor.fetchone()
                if not ins_row:
                    inst_type = 'bond_yield' if symbol.startswith('^') or symbol in ['TLT', 'IEF'] else 'stock'
                    cursor.execute(
                        "INSERT INTO instrument (symbol, name, instrument_type, active) VALUES (%s, %s, %s, 1)",
                        (symbol, symbol, inst_type)
                    )
                    inst_id = cursor.lastrowid
                else:
                    inst_id = ins_row['id']

                # Link to portfolio if not linked
                cursor.execute(
                    "SELECT 1 FROM portfolio_instrument WHERE portfolio_id = %s AND instrument_id = %s",
                    (portfolio_id, inst_id)
                )
                if not cursor.fetchone():
                    cursor.execute(
                        "INSERT INTO portfolio_instrument (portfolio_id, instrument_id) VALUES (%s, %s)",
                        (portfolio_id, inst_id)
                    )
            conn.commit()
    logger.info("Universe seeding completed.")

def download_historical_data(period="5y"):
    logger.info(f"Downloading historical data (period={period}) for all active instruments...")
    instruments = execute_query("SELECT id, symbol FROM instrument WHERE active = 1")
    if not instruments:
        logger.warning("No active instruments found.")
        return

    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    upsert_query = """
        INSERT INTO price_daily (instrument_id, trade_date, open, high, low, close, adjusted_close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open = VALUES(open),
            high = VALUES(high),
            low = VALUES(low),
            close = VALUES(close),
            adjusted_close = VALUES(adjusted_close),
            volume = VALUES(volume)
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for idx, ins in enumerate(instruments):
                inst_id = ins['id']
                symbol = ins['symbol']

                df = load_cached(f"{symbol}_{period}")
                if df is None:
                    try:
                        if idx > 0:
                            time.sleep(1.0)
                        ticker = yf.Ticker(symbol, session=session)
                        df = ticker.history(period=period, auto_adjust=True)
                        if df.empty:
                            logger.warning(f"No history for {symbol}")
                            continue
                        df = df.reset_index()
                        save_cached(f"{symbol}_{period}", df)
                        logger.info(f"Downloaded and cached {len(df)} bars for {symbol}")
                    except Exception as e:
                        logger.error(f"Error downloading {symbol}: {e}")
                        continue

                date_col = 'Date' if 'Date' in df.columns else df.columns[0]

                rows = []
                for _, row in df.iterrows():
                    trade_date = pd.to_datetime(row[date_col]).strftime('%Y-%m-%d')
                    o = float(row.get('Open', 0)) or None
                    h = float(row.get('High', 0)) or None
                    l = float(row.get('Low', 0)) or None
                    c = float(row.get('Close', 0)) or None
                    v = int(row.get('Volume', 0)) if not pd.isna(row.get('Volume', 0)) else 0
                    if c is None:
                        continue
                    rows.append((inst_id, trade_date, o, h, l, c, c, v))

                if rows:
                    cursor.executemany(upsert_query, rows)
    logger.info("Historical data download completed.")

def build_dataset_and_train():
    logger.info("Building dataset from database for training...")
    rows = execute_query("""
        SELECT p.instrument_id, i.symbol, p.trade_date, p.open, p.high, p.low, p.close, p.volume
        FROM price_daily p
        JOIN instrument i ON p.instrument_id = i.id
        ORDER BY p.instrument_id, p.trade_date
    """)
    raw_df = pd.DataFrame(rows)

    if raw_df.empty:
        logger.error("No price data found in database for training.")
        return

    logger.info(f"Loaded {len(raw_df)} price rows. Computing features...")

    for col in ['open', 'high', 'low', 'close', 'volume']:
        raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
    raw_df['trade_date'] = pd.to_datetime(raw_df['trade_date'])

    feature_cols = [
        'log_return', 'mom_5', 'mom_10', 'mom_20',
        'vol_10', 'vol_20', 'vol_ratio', 'zscore_20',
        'trend_50', 'vol_regime', 'mom_vol_adj', 'volume_shock'
    ]

    feature_dfs = []
    for symbol, group in raw_df.groupby('symbol'):
        df = group.sort_values('trade_date').copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(5))
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

        # Target: forward return over 5 days, scaled by volatility 20
        fwd_close = df['close'].shift(-5)
        df['target'] = np.log(fwd_close / df['close']) / df['vol_20']
        df['target'] = df['target'].clip(-3.0, 3.0)

        feature_dfs.append(df)

    full_df = pd.concat(feature_dfs, ignore_index=True)

    clean_df = full_df.dropna(subset=feature_cols + ['target']).copy()
    
    if clean_df.empty:
        logger.error("Clean dataset is empty after dropping NaNs.")
        return

    X = clean_df[feature_cols]
    y = clean_df['target']

    # Min-max normalization
    mins = X.min()
    maxs = X.max()
    X_norm = (X - mins) / (maxs - mins + 1e-8)

    # Normalize target between 0 and 1 for sigmoid output
    label_min = y.min()
    label_max = y.max()
    y_norm = (y - label_min) / (label_max - label_min + 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)

    logger.info(f"Training MLP model on {len(X_train)} samples...")

    model = Sequential([
        Input(shape=(len(feature_cols),)),
        Dense(100, activation='relu'),
        Dense(80, activation='relu'),
        Dense(20, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1)

    # Save outputs
    model_dir = os.path.abspath("model/stock_model/v1")
    os.makedirs(model_dir, exist_ok=True)

    keras_path = os.path.join(model_dir, "model.keras")
    model.save(keras_path)

    # Export to ONNX via SavedModel + tf2onnx CLI (compatible with Keras 3)
    saved_model_dir = os.path.join(model_dir, "saved_model")
    model.export(saved_model_dir)
    onnx_path = os.path.join(model_dir, "model.onnx")
    import subprocess, sys
    res = subprocess.run(
        [sys.executable, "-m", "tf2onnx.convert",
         "--saved-model", saved_model_dir,
         "--output", onnx_path],
        capture_output=True, text=True
    )
    if res.returncode != 0:
        logger.error(f"tf2onnx conversion failed: {res.stderr}")
        raise RuntimeError("tf2onnx conversion failed")
    logger.info("ONNX export completed.")

    # Save normalization & config (format col,value expected by predict.py)
    norm_df = pd.DataFrame({
        'col': mins.index.tolist() + ['target'],
        'value': mins.tolist() + [float(label_min)]
    })
    norm_df.to_csv(os.path.join(model_dir, "mins.csv"), index=False)
    norm_df = pd.DataFrame({
        'col': maxs.index.tolist() + ['target'],
        'value': maxs.tolist() + [float(label_max)]
    })
    norm_df.to_csv(os.path.join(model_dir, "maxs.csv"), index=False)

    clip_min = -3.0
    clip_max = 3.0

    config = {
        "feature_columns": feature_cols,
        "label_min": float(label_min),
        "label_max": float(label_max),
        "clip_min": clip_min,
        "clip_max": clip_max
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Register model in model_registry and link it to the portfolio
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO model_registry
                    (name, version, artifact_path, feature_columns,
                     label_min, label_max, clip_min, clip_max, metrics, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'active')
                ON DUPLICATE KEY UPDATE
                    artifact_path = VALUES(artifact_path),
                    feature_columns = VALUES(feature_columns),
                    label_min = VALUES(label_min),
                    label_max = VALUES(label_max),
                    clip_min = VALUES(clip_min),
                    clip_max = VALUES(clip_max),
                    status = 'active'
            """, (
                "stock_model", "v1", model_dir, json.dumps(feature_cols),
                float(label_min), float(label_max), clip_min, clip_max,
                json.dumps({"samples": int(len(clean_df))})
            ))
            model_id = cursor.lastrowid
            cursor.execute("SELECT id FROM model_registry WHERE name='stock_model' AND version='v1'")
            row = cursor.fetchone()
            if row:
                model_id = row['id']
            cursor.execute("UPDATE portfolio SET model_id = %s WHERE code = 'main'", (model_id,))
        conn.commit()

    logger.info(f"Training complete. Artifacts saved in {model_dir} (model_registry id={model_id})")

if __name__ == "__main__":
    seed_universe()
    download_historical_data("5y")
    build_dataset_and_train()
