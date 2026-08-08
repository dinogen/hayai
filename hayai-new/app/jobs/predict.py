import os
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path
from app.db import execute_query, get_db_connection
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.predict")

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute type-agnostic technical features for an instrument dataframe."""
    df = df.sort_values('trade_date').copy()
    trd = 5 # target return days

    df['log_return'] = np.log(df['close'] / df['close'].shift(trd))
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

    df = df.dropna().reset_index(drop=True)
    return df

def run_predict_job(portfolio_code: str = "main") -> dict:
    logger.info("Fetching active model for prediction...")
    model_rows = execute_query("""
        SELECT m.id, m.artifact_path, m.feature_columns, m.label_min, m.label_max, m.clip_min, m.clip_max
        FROM model_registry m
        JOIN portfolio p ON p.model_id = m.id
        WHERE p.code = %s AND m.status = 'active'
    """, (portfolio_code,))

    if not model_rows:
        # Fallback to any active model
        model_rows = execute_query("SELECT id, artifact_path, feature_columns, label_min, label_max, clip_min, clip_max FROM model_registry WHERE status = 'active' LIMIT 1")
    
    if not model_rows:
        logger.warning("No active model found in model_registry. Skipping prediction.")
        return {"predictions_made": 0, "status": "no_active_model"}

    model_info = model_rows[0]
    model_id = model_info['id']
    artifact_path = Path(model_info['artifact_path'])
    feature_cols = json.loads(model_info['feature_columns'])
    label_min = float(model_info['label_min'])
    label_max = float(model_info['label_max'])
    clip_min = float(model_info['clip_min'])
    clip_max = float(model_info['clip_max'])

    onnx_file = artifact_path / "model.onnx"
    mins_file = artifact_path / "mins.csv"
    maxs_file = artifact_path / "maxs.csv"

    if not onnx_file.exists() or not mins_file.exists() or not maxs_file.exists():
        logger.error(f"Model artifact files missing in {artifact_path}")
        raise FileNotFoundError(f"Model artifact files missing in {artifact_path}")

    mins = pd.read_csv(mins_file, index_col='col')['value'].drop('target', errors='ignore')
    maxs = pd.read_csv(maxs_file, index_col='col')['value'].drop('target', errors='ignore')

    logger.info(f"Loading ONNX model from {onnx_file}...")
    ort_session = ort.InferenceSession(str(onnx_file))
    input_name = ort_session.get_inputs()[0].name

    logger.info("Fetching instruments and price data...")
    instruments = execute_query("""
        SELECT i.id, i.symbol
        FROM instrument i
        JOIN portfolio_instrument pi ON i.id = pi.instrument_id
        JOIN portfolio p ON pi.portfolio_id = p.id
        WHERE p.code = %s AND i.active = 1
    """, (portfolio_code,))

    upsert_query = """
        INSERT INTO model_prediction (model_id, instrument_id, as_of_date, prediction, vol_20)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            prediction = VALUES(prediction),
            vol_20 = VALUES(vol_20)
    """

    predictions_count = 0
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for ins in instruments:
                inst_id = ins['id']
                symbol = ins['symbol']

                prices = execute_query("""
                    SELECT trade_date, open, high, low, close, adjusted_close, volume
                    FROM price_daily
                    WHERE instrument_id = %s
                    ORDER BY trade_date ASC
                """, (inst_id,))

                if not prices or len(prices) < 60:
                    logger.warning(f"Insufficient price history for {symbol} (< 60 bars).")
                    continue

                df = pd.DataFrame(prices)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = compute_features(df)
                if df.empty:
                    logger.warning(f"Features dataframe empty for {symbol} after cleaning.")
                    continue

                # Take the latest row (latest date)
                latest_row = df.iloc[-1]
                as_of_date = pd.to_datetime(latest_row['trade_date']).strftime('%Y-%m-%d')
                vol_20 = float(latest_row.get('vol_20', 0.0))

                # Extract features matching training columns
                feat_dict = {}
                for col in feature_cols:
                    feat_dict[col] = latest_row.get(col, 0.0)

                x_df = pd.DataFrame([feat_dict], columns=feature_cols)
                # Normalize
                x_norm = (x_df - mins) / (maxs - mins)
                x_norm = x_norm.fillna(0.0).values.astype(np.float32)

                # Predict
                outputs = ort_session.run(None, {input_name: x_norm})
                raw_pred = float(outputs[0][0][0])

                # Denormalize & Clip
                pred_denorm = raw_pred * (label_max - label_min) + label_min
                final_pred = float(np.clip(pred_denorm, clip_min, clip_max))

                cursor.execute(upsert_query, (model_id, inst_id, as_of_date, final_pred, vol_20))
                predictions_count += 1
                logger.info(f"Prediction for {symbol} on {as_of_date}: {final_pred:.4f}")

    return {"predictions_made": predictions_count}
