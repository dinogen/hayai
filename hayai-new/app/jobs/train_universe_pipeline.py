import os
import time
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from app.db import get_db_connection, execute_query
from app.jobs.cache import load_cached, save_cached
from app.jobs.dataset_builder import build_training_dataset, split_by_date
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.train_universe_pipeline")

MODEL_NAME = "stock_model"
MODEL_VERSION = "v2"

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
    # Mid-Cap Stocks (higher-risk additions)
    "ESTC", "GTLB", "DBX", "SMCI", "ON", "CROX", "WING", "FIVE", "MRNA", "NVAX",
    "GNRC", "MTZ", "MTDR", "SOFI", "AFRM",
    # ETFs (Broad, Sector, Bonds, Commodities)
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", 
    "XLK", "U", "XLRE", "TLT", "IEF", "GLD", "SLV", "USO", "VNQ", "ARKK",
    # EM High-Yield Sovereign Bond ETFs
    "EMB", "VWOB", "PCY", "EMLC",
    # International & Indices / Rates
    "EEM", "EFA", "EWJ", "FXI", "ASHR", "VGK", "EWZ", "INDA", "VWO", "IAU"
]

def _fetch_instrument_meta(symbol, session):
    """Fetch (name, instrument_type, currency) from yfinance, with graceful fallback."""
    try:
        info = yf.Ticker(symbol, session=session).info
    except Exception as e:
        logger.warning(f"Could not fetch metadata for {symbol}: {e}")
        info = {}

    name = info.get("shortName") or info.get("longName") or symbol

    quote_type = (info.get("quoteType") or "").upper()
    if quote_type == "ETF":
        inst_type = "etf"
    elif quote_type == "INDEX" or symbol.startswith("^"):
        inst_type = "bond_yield"
    else:
        inst_type = "stock"

    currency = info.get("currency") or "USD"
    return name, inst_type, currency

def seed_universe():
    logger.info(f"Seeding {len(UNIVERSE_SYMBOLS)} symbols into database...")
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Ensure portfolio 'main' exists
            cursor.execute("SELECT id FROM portfolio WHERE code = 'main'")
            row = cursor.fetchone()
            if not row:
                cursor.execute("INSERT INTO portfolio (code, name, initial_capital) VALUES ('main', 'Main Portfolio', 5000.00)")
                portfolio_id = cursor.lastrowid
            else:
                portfolio_id = row['id']

            for idx, symbol in enumerate(UNIVERSE_SYMBOLS):
                # Insert instrument if not exists
                cursor.execute("SELECT id FROM instrument WHERE symbol = %s", (symbol,))
                ins_row = cursor.fetchone()
                if not ins_row:
                    if idx > 0:
                        time.sleep(0.5)
                    name, inst_type, currency = _fetch_instrument_meta(symbol, session)
                    cursor.execute(
                        "INSERT INTO instrument (symbol, name, instrument_type, currency, active) VALUES (%s, %s, %s, %s, 1)",
                        (symbol, name, inst_type, currency)
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

def build_dataset_and_train(split: str = 'random', version: str = None, make_active: bool = True):
    logger.info("Building dataset from database for training...")
    dataset = build_training_dataset()
    if dataset is None:
        return

    clean_df, feature_cols, mins, maxs, label_min, label_max = dataset
    version = version or MODEL_VERSION
    split = (split or 'random').lower()

    if split == 'time':
        train_mask, val_mask, test_mask, cutoffs = split_by_date(clean_df)
        X_train_raw = clean_df.loc[train_mask, feature_cols]
        y_train_raw = clean_df.loc[train_mask, 'target']
        mins = X_train_raw.min()
        maxs = X_train_raw.max()
        label_min = float(y_train_raw.min())
        label_max = float(y_train_raw.max())

        X_train = (X_train_raw - mins) / (maxs - mins + 1e-8)
        y_train = (y_train_raw - label_min) / (label_max - label_min + 1e-8)
        X_val = (clean_df.loc[val_mask, feature_cols] - mins) / (maxs - mins + 1e-8)
        y_val = (clean_df.loc[val_mask, 'target'] - label_min) / (label_max - label_min + 1e-8)
        X_test = (clean_df.loc[test_mask, feature_cols] - mins) / (maxs - mins + 1e-8)
        y_test = (clean_df.loc[test_mask, 'target'] - label_min) / (label_max - label_min + 1e-8)
        validation_data = (X_val, y_val)
        logger.info(f"Time split: train={len(X_train)} val={len(X_val)} test={len(X_test)} "
                    f"(train<={cutoffs['train_end']}, val<={cutoffs['val_end']})")
    else:
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            clean_df[feature_cols], clean_df['target'], test_size=0.2, random_state=42
        )
        mins = X_train_raw.min()
        maxs = X_train_raw.max()
        label_min = float(y_train_raw.min())
        label_max = float(y_train_raw.max())

        X_train = (X_train_raw - mins) / (maxs - mins + 1e-8)
        y_train = (y_train_raw - label_min) / (label_max - label_min + 1e-8)
        X_test = (X_test_raw - mins) / (maxs - mins + 1e-8)
        y_test = (y_test_raw - label_min) / (label_max - label_min + 1e-8)
        validation_data = (X_test, y_test)
        cutoffs = None

    logger.info(f"Training MLP model on {len(X_train)} samples...")

    model = Sequential([
        Input(shape=(len(feature_cols),)),
        Dense(100, activation='relu'),
        Dense(80, activation='relu'),
        Dense(20, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=64,
              validation_data=validation_data, verbose=1, callbacks=[early_stopping])

    # Save outputs
    model_dir = os.path.abspath(f"model/{MODEL_NAME}/{version}")
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
        "clip_max": clip_max,
        "split": split,
    }
    if cutoffs:
        config["train_end"] = cutoffs["train_end"]
        config["val_end"] = cutoffs["val_end"]
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Register model in model_registry and (optionally) link it to the portfolio
    reg_status = 'active' if make_active else 'draft'
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO model_registry
                    (name, version, artifact_path, feature_columns,
                     label_min, label_max, clip_min, clip_max, metrics, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    artifact_path = VALUES(artifact_path),
                    feature_columns = VALUES(feature_columns),
                    label_min = VALUES(label_min),
                    label_max = VALUES(label_max),
                    clip_min = VALUES(clip_min),
                    clip_max = VALUES(clip_max),
                    metrics = VALUES(metrics),
                    status = VALUES(status)
            """, (
                MODEL_NAME, version, model_dir, json.dumps(feature_cols),
                float(label_min), float(label_max), clip_min, clip_max,
                json.dumps({"samples": int(len(clean_df)), "split": split}), reg_status
            ))
            model_id = cursor.lastrowid
            cursor.execute("SELECT id FROM model_registry WHERE name=%s AND version=%s",
                           (MODEL_NAME, version))
            row = cursor.fetchone()
            if row:
                model_id = row['id']
            if make_active:
                cursor.execute("UPDATE portfolio SET model_id = %s WHERE code = 'main'", (model_id,))
        conn.commit()

    logger.info(f"Training complete. Artifacts saved in {model_dir} (model_registry id={model_id}, status={reg_status})")

if __name__ == "__main__":
    seed_universe()
    download_historical_data("5y")
    build_dataset_and_train()
