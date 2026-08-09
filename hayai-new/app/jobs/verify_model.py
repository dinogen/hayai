import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.config import settings
from app.db import execute_query
from app.jobs.dataset_builder import build_training_dataset, read_model_config, split_by_cutoffs
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.verify_model")

DEFAULT_MODEL_DIR = Path("model/stock_model/v1")


def _load_model(portfolio_code: str, model_version: str = None):
    if model_version:
        model_rows = execute_query("""
            SELECT id, name, version, artifact_path, feature_columns,
                   label_min, label_max, clip_min, clip_max
            FROM model_registry WHERE name = 'stock_model' AND version = %s
        """, (model_version,))
    else:
        model_rows = execute_query("""
            SELECT m.id, m.name, m.version, m.artifact_path, m.feature_columns,
                   m.label_min, m.label_max, m.clip_min, m.clip_max
            FROM model_registry m
            JOIN portfolio p ON p.model_id = m.id
            WHERE p.code = %s AND m.status = 'active'
        """, (portfolio_code,))

        if not model_rows:
            model_rows = execute_query("""
                SELECT id, name, version, artifact_path, feature_columns,
                       label_min, label_max, clip_min, clip_max
                FROM model_registry WHERE status = 'active' LIMIT 1
            """)

    return model_rows[0] if model_rows else None


def _load_onnx_session(artifact_path: Path):
    onnx_file = artifact_path / "model.onnx"
    if not onnx_file.exists():
        logger.error(f"ONNX model file missing in {artifact_path}")
        raise FileNotFoundError(f"ONNX model file missing in {artifact_path}")
    session = ort.InferenceSession(str(onnx_file))
    return session, session.get_inputs()[0].name


def _nan_report(name: str, arr: np.ndarray) -> list:
    total = int(np.size(arr))
    na = int(np.isnan(np.asarray(arr, dtype=float)).sum())
    inf = int(np.isinf(np.asarray(arr, dtype=float)).sum())
    return {
        "name": name,
        "total": total,
        "nan": na,
        "inf": inf,
    }


def run_verify_model_job(portfolio_code: str = "main", model_version: str = None) -> dict:
    logger.info(f"Running model verification for portfolio '{portfolio_code}' (version={model_version or 'active'})...")

    model_info = _load_model(portfolio_code, model_version)
    if not model_info:
        logger.error("No active model found in model_registry. Skipping verification.")
        return {"status": "no_active_model", "report_file": None}

    model_name = model_info.get("name", "stock_model")
    model_version = model_info.get("version", "v1")
    artifact_path = Path(model_info["artifact_path"])
    feature_cols = json.loads(model_info["feature_columns"])
    label_min_model = float(model_info["label_min"])
    label_max_model = float(model_info["label_max"])
    clip_min = float(model_info["clip_min"])
    clip_max = float(model_info["clip_max"])

    logger.info(f"Model: {model_name} v{model_version} at {artifact_path}")

    if not artifact_path.exists():
        logger.warning(f"Artifact path {artifact_path} not found, falling back to {DEFAULT_MODEL_DIR}")
        artifact_path = DEFAULT_MODEL_DIR

    model_config = read_model_config(artifact_path)
    split_mode = model_config.get("split", "random")

    ort_session, input_name = _load_onnx_session(artifact_path)

    logger.info("Building dataset from database...")
    dataset = build_training_dataset()
    if dataset is None:
        logger.error("Dataset build failed. Skipping verification.")
        return {"status": "dataset_empty", "report_file": None}

    clean_df, _, mins, maxs, label_min_rec, label_max_rec = dataset

    X = clean_df[feature_cols]
    y = clean_df["target"]

    if split_mode == "time":
        train_end = model_config.get("train_end")
        val_end = model_config.get("val_end")
        if not train_end or not val_end:
            logger.error("Model config missing train_end/val_end for time split. Aborting.")
            return {"status": "missing_cutoffs", "report_file": None}
        train_mask, val_mask, test_mask = split_by_cutoffs(clean_df, train_end, val_end)

        X_train_raw = clean_df.loc[train_mask, feature_cols]
        mins = X_train_raw.min()
        maxs = X_train_raw.max()
        label_min_rec = float(clean_df.loc[train_mask, "target"].min())
        label_max_rec = float(clean_df.loc[train_mask, "target"].max())

        X_train = (clean_df.loc[train_mask, feature_cols] - mins) / (maxs - mins + 1e-8)
        X_test = (clean_df.loc[test_mask, feature_cols] - mins) / (maxs - mins + 1e-8)
        y_train = (clean_df.loc[train_mask, "target"] - label_min_rec) / (label_max_rec - label_min_rec + 1e-8)
        y_test = (clean_df.loc[test_mask, "target"] - label_min_rec) / (label_max_rec - label_min_rec + 1e-8)
        split_ratio = len(X_train) / (len(X_train) + len(X_test))
    else:
        X_norm = (X - mins) / (maxs - mins + 1e-8)
        y_norm = (y - label_min_rec) / (label_max_rec - label_min_rec + 1e-8)

        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y_norm, test_size=0.2, random_state=42
        )
        split_ratio = len(X_train) / (len(X_train) + len(X_test))

    checks = [
        _nan_report("X_train", X_train.to_numpy()),
        _nan_report("X_test", X_test.to_numpy()),
        _nan_report("y_train", y_train.to_numpy()),
        _nan_report("y_test", y_test.to_numpy()),
    ]
    nan_total = sum(c["nan"] + c["inf"] for c in checks)

    # Predict on the whole test set with the deployed ONNX model
    x_test_np = X_test.to_numpy(dtype=np.float32)
    raw_pred = ort_session.run(None, {input_name: x_test_np})[0].ravel()
    pred_denorm = raw_pred * (label_max_model - label_min_model) + label_min_model
    pred_native = np.clip(pred_denorm, clip_min, clip_max)

    actual_native = y_test.to_numpy(dtype=float) * (label_max_rec - label_min_rec) + label_min_rec

    rmse = float(np.sqrt(mean_squared_error(actual_native, pred_native)))
    mae = float(mean_absolute_error(actual_native, pred_native))
    r2 = float(r2_score(actual_native, pred_native))

    baseline_mean = np.full_like(actual_native, float(np.mean(actual_native)))
    rmse_baseline = float(np.sqrt(mean_squared_error(actual_native, baseline_mean)))
    mae_baseline = float(mean_absolute_error(actual_native, baseline_mean))

    pred_dir = np.sign(pred_native)
    actual_dir = np.sign(actual_native)
    hit_rate = float(np.mean(pred_dir == actual_dir))

    n = len(actual_native)
    tp = int(np.sum((pred_dir > 0) & (actual_dir > 0)))
    fp = int(np.sum((pred_dir > 0) & (actual_dir <= 0)))
    tn = int(np.sum((pred_dir <= 0) & (actual_dir <= 0)))
    fn = int(np.sum((pred_dir <= 0) & (actual_dir > 0)))

    # Sample 100 random test rows for the spot check
    n_sample = min(100, len(X_test))
    sample_idx = X_test.sample(n=n_sample, random_state=42).index
    sample_meta = clean_df.loc[sample_idx, ['symbol', 'trade_date']]
    actual_series = pd.Series(actual_native, index=X_test.index)
    pred_series = pd.Series(pred_native, index=X_test.index)
    sample_rows = pd.DataFrame({
        'symbol': sample_meta['symbol'].values,
        'trade_date': pd.to_datetime(sample_meta['trade_date']).dt.strftime('%Y-%m-%d').values,
        'actual': np.round(actual_series.loc[sample_idx].values, 4),
        'prediction': np.round(pred_series.loc[sample_idx].values, 4),
    })
    sample_rows['match'] = np.sign(sample_rows['prediction']) == np.sign(sample_rows['actual'])

    # Drift check: recomputed label range vs deployed one
    label_drift = abs(label_min_rec - label_min_model) > 1e-6 or abs(label_max_rec - label_max_model) > 1e-6

    errors = []
    if nan_total > 0:
        errors.append(f"Trovati {nan_total} valori NaN/Inf nei dati di training/test.")

    report_lines = _build_report(
        model_name=model_name,
        model_version=model_version,
        artifact_path=str(artifact_path),
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        portfolio=portfolio_code,
        split_mode=split_mode,
        raw_rows=len(clean_df),
        train_size=len(X_train),
        test_size=len(X_test),
        split_ratio=split_ratio,
        checks=checks,
        nan_total=nan_total,
        rmse=rmse, mae=mae, r2=r2,
        hit_rate=hit_rate,
        tp=tp, fp=fp, tn=tn, fn=fn,
        rmse_baseline=rmse_baseline, mae_baseline=mae_baseline,
        sample_rows=sample_rows,
        label_min_rec=label_min_rec, label_max_rec=label_max_rec,
        label_min_model=label_min_model, label_max_model=label_max_model,
        label_drift=label_drift,
        errors=errors,
    )

    report_dir = settings.LOGS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"model_verification_{model_name}_{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_lines)

    for line in report_lines.splitlines():
        if line.strip():
            logger.info(line)

    if errors:
        logger.error("Verification FAILED: " + " | ".join(errors))
        raise RuntimeError("Verification failed: " + "; ".join(errors))

    logger.info(f"Verification completed. Report written to {report_file}")
    return {
        "status": "ok",
        "report_file": str(report_file),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "hit_rate": hit_rate,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }


def _build_report(model_name, model_version, artifact_path, timestamp, portfolio,
                  split_mode, raw_rows, train_size, test_size, split_ratio, checks, nan_total,
                  rmse, mae, r2, hit_rate, tp, fp, tn, fn,
                  rmse_baseline, mae_baseline, sample_rows,
                  label_min_rec, label_max_rec, label_min_model, label_max_model,
                  label_drift, errors) -> str:
    lines = []
    sep = "=" * 78
    lines.append(sep)
    lines.append("REPORT VERIFICA MODELLO — HAYAI v2")
    lines.append(sep)
    lines.append(f"Modello      : {model_name} v{model_version.lstrip('v')}")
    lines.append(f"Artifact     : {artifact_path}")
    lines.append(f"Portfolio    : {portfolio}")
    lines.append(f"Data/ora     : {timestamp}")
    lines.append("")

    lines.append(sep)
    lines.append("1) QUALITA' DATASET (assenza di null/NaN)")
    lines.append(sep)
    lines.append(f"Righe totali (dopo dropna): {raw_rows}")
    for c in checks:
        status = "OK" if (c['nan'] == 0 and c['inf'] == 0) else "KO"
        lines.append(f"  - {c['name']:<10}: {c['total']:>8} valori | NaN={c['nan']} | Inf={c['inf']} [{status}]")
    lines.append(f"Esito        : {'PULITO (nessun null/NaN/Inf)' if nan_total == 0 else f'TROVATI {nan_total} null/NaN/Inf'}")
    lines.append("")

    lines.append(sep)
    if split_mode == 'time':
        lines.append("2) SPLIT TRAIN/TEST (holdout cronologico 70/15/15)")
        lines.append(sep)
        lines.append(f"  - Training: {train_size:>8} righe ({train_size / (train_size + test_size) * 100:.1f}% delle righe non-test)")
        lines.append(f"  - Test    : {test_size:>8} righe (ultime date, mai usate dall'early stopping)")
        lines.append(f"  - Split attuale: {split_ratio * 100:.1f}% train / {(1 - split_ratio) * 100:.1f}% test (per data)")
    else:
        lines.append("2) SPLIT TRAIN/TEST (80/20)")
        lines.append(sep)
        lines.append(f"  - Training: {train_size:>8} righe ({train_size / (train_size + test_size) * 100:.1f}%)")
        lines.append(f"  - Test    : {test_size:>8} righe ({test_size / (train_size + test_size) * 100:.1f}%)")
        lines.append(f"  - Split attuale: {split_ratio * 100:.1f}% / {(1 - split_ratio) * 100:.1f}% (atteso 80% / 20%, random_state=42)")
    lines.append("")

    lines.append(sep)
    lines.append("3) METRICHE (unita' del target = ln-return/vol_20, clip [-3, 3])")
    lines.append(sep)
    lines.append("DEFINIZIONI:")
    lines.append("  - RMSE (scarto quadratico medio) = sqrt(media((pred - actual)^2)): errore tipico")
    lines.append("    in unita' del target; piu' piccolo = migliore.")
    lines.append("  - MAE (errore medio assoluto) = media(|pred - actual|): errore medio in unita' del target.")
    lines.append("  - R2 (coefficiente di determinazione) = 1 - SS_res/SS_tot: quota di varianza spiegata;")
    lines.append("    per i rendimenti finanziari valori bassi o ~0 sono normali, non un errore.")
    lines.append("  - Hit-rate direzionale = % di osservazioni con segno(pred) == segno(actual):")
    lines.append("    quante volte il modello indovina la direzione (su/giu').")
    lines.append("")
    lines.append(f"  RMSE              : {rmse:.6f}")
    lines.append(f"  MAE               : {mae:.6f}")
    lines.append(f"  R2                : {r2:.6f}")
    lines.append(f"  Hit-rate          : {hit_rate * 100:.2f}%")
    lines.append("")
    lines.append("  Matrice direzione (pred/actual):")
    lines.append(f"    su/su (TP)   : {tp}")
    lines.append(f"    su/giu' (FP) : {fp}")
    lines.append(f"    giu'/giu'(TN): {tn}")
    lines.append(f"    giu'/su (FN) : {fn}")
    lines.append("")
    lines.append("  Baseline ingenua (predici sempre la media del test):")
    lines.append(f"    RMSE baseline : {rmse_baseline:.6f}")
    lines.append(f"    MAE baseline  : {mae_baseline:.6f}")
    lines.append(f"    -> il modello {'supera' if rmse < rmse_baseline else 'NON supera'} la baseline (RMSE)")
    lines.append("")

    lines.append(sep)
    lines.append("4) SPOT CHECK — 100 RIGHE CASUALI DEL TEST SET (predizione vs actual)")
    lines.append(sep)
    if sample_rows.empty:
        lines.append("  (test set vuoto)")
    else:
        header = f"  {'symbol':<8} {'trade_date':<12} {'actual':>10} {'prediction':>10} {'match':>6}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for _, r in sample_rows.iterrows():
            lines.append(f"  {str(r['symbol']):<8} {str(r['trade_date']):<12} {r['actual']:>10.4f} {r['prediction']:>10.4f} {'OK' if r['match'] else 'NO':>6}")
    lines.append("")

    lines.append(sep)
    lines.append("5) DRIFT MIN/MAX (label target)")
    lines.append(sep)
    lines.append(f"  Ricomputati  : min={label_min_rec:.4f} max={label_max_rec:.4f}")
    lines.append(f"  Deployati    : min={label_min_model:.4f} max={label_max_model:.4f}")
    if label_drift:
        lines.append("  ATTENZIONE: i min/max ricomputati differiscono da quelli del modello deployato.")
        lines.append("  Possibile drift dei dati: valutare un retraining.")
    else:
        lines.append("  Nessuna differenza significativa.")
    lines.append("")

    if errors:
        lines.append(sep)
        lines.append("6) ERRORI")
        lines.append(sep)
        for e in errors:
            lines.append(f"  [ERROR] {e}")
        lines.append("")

    lines.append(sep)
    lines.append("FINE REPORT")
    lines.append(sep)
    return "\n".join(lines)


if __name__ == "__main__":
    portfolio = sys.argv[1] if len(sys.argv) > 1 else "main"
    version = sys.argv[2] if len(sys.argv) > 2 else None
    run_verify_model_job(portfolio_code=portfolio, model_version=version)
