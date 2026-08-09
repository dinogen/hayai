import json
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort

from app.db import execute_query, get_db_connection
from app.jobs.dataset_builder import compute_panel_features
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.predict")


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
    rows = execute_query("""
        SELECT p.instrument_id, i.symbol, p.trade_date, p.open, p.high, p.low, p.close, p.adjusted_close, p.volume
        FROM price_daily p
        JOIN instrument i ON p.instrument_id = i.id
        JOIN portfolio_instrument pi ON i.id = pi.instrument_id
        JOIN portfolio pf ON pi.portfolio_id = pf.id
        WHERE pf.code = %s AND i.active = 1
        ORDER BY p.instrument_id, p.trade_date
    """, (portfolio_code,))

    if not rows:
        logger.warning(f"No price data found for portfolio '{portfolio_code}'.")
        return {"predictions_made": 0}

    raw_df = pd.DataFrame(rows)
    bar_counts = raw_df.groupby('instrument_id').size()

    logger.info("Building cross-sectional feature panel...")
    panel = compute_panel_features(raw_df)
    if panel.empty:
        logger.error("Feature panel is empty.")
        return {"predictions_made": 0}

    latest = panel.sort_values('trade_date').groupby('instrument_id', as_index=False).tail(1).set_index('instrument_id')

    upsert_query = """
        INSERT INTO model_prediction (model_id, instrument_id, as_of_date, prediction, vol_20)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            prediction = VALUES(prediction),
            vol_20 = VALUES(vol_20)
    """

    x_df = latest[feature_cols].reindex(columns=feature_cols)
    x_norm = (x_df - mins) / (maxs - mins)
    x_norm = x_norm.fillna(0.0).values.astype(np.float32)

    logger.info(f"Running ONNX inference on {len(latest)} instruments...")
    outputs = ort_session.run(None, {input_name: x_norm})
    raw_pred = outputs[0].ravel()
    pred_denorm = raw_pred * (label_max - label_min) + label_min
    final_pred = np.clip(pred_denorm, clip_min, clip_max)

    predictions_count = 0
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for i, (inst_id, row) in enumerate(latest.iterrows()):
                symbol = row['symbol']
                if bar_counts.get(inst_id, 0) < 60:
                    logger.warning(f"Insufficient price history for {symbol} (< 60 bars).")
                    continue
                as_of_date = pd.to_datetime(row['trade_date']).strftime('%Y-%m-%d')
                vol_20 = float(row.get('vol_20', 0.0)) if pd.notna(row.get('vol_20', 0.0)) else 0.0
                cursor.execute(upsert_query, (model_id, inst_id, as_of_date, float(final_pred[i]), vol_20))
                predictions_count += 1
                logger.info(f"Prediction for {symbol} on {as_of_date}: {final_pred[i]:.4f}")

    return {"predictions_made": predictions_count}
