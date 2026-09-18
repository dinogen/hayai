"""Train a draft chronological model with train-only preprocessing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

from app.db import get_db_connection
from app.jobs.dataset_builder import (
    FEATURE_COLS,
    _apply_train_only_preprocessing,
    build_raw_df,
    compute_panel_features,
    split_by_date,
)
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.train_clean_chronological")

MODEL_NAME = "stock_model"
DEFAULT_VERSION = "v5_clean_time"
CLIP_MIN = -3.0
CLIP_MAX = 3.0


def _clean_train_only_dataset() -> tuple[pd.DataFrame, list[str], pd.Series, pd.Series, float, float, dict[str, str]]:
    raw = build_raw_df()
    if raw.empty:
        raise RuntimeError("No price data available")
    panel = compute_panel_features(raw)
    clean = panel.dropna(subset=FEATURE_COLS + ["target"]).copy()
    if clean.empty:
        raise RuntimeError("Dataset empty after feature/target validation")

    train_mask, validation_mask, test_mask, cutoffs = split_by_date(clean)
    clean = _apply_train_only_preprocessing(clean, FEATURE_COLS, train_mask)

    train = clean.loc[train_mask]
    mins = train[FEATURE_COLS].min()
    maxs = train[FEATURE_COLS].max()
    label_min = float(train["target"].min())
    label_max = float(train["target"].max())
    return clean, FEATURE_COLS, mins, maxs, label_min, label_max, cutoffs


def _build_model(feature_count: int) -> Sequential:
    model = Sequential([
        Input(shape=(feature_count,)),
        Dense(100, activation="relu"),
        Dense(80, activation="relu"),
        Dense(20, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model


def _export_onnx(model: Sequential, model_dir: Path) -> None:
    saved_model_dir = model_dir / "saved_model"
    model.export(saved_model_dir)
    result = subprocess.run(
        [sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(saved_model_dir), "--output", str(model_dir / "model.onnx")],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError("tf2onnx conversion failed")


def _register_draft(version: str, model_dir: Path, feature_cols: list[str], label_min: float, label_max: float, samples: int, cutoffs: dict[str, str]) -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO model_registry
                   (name, version, artifact_path, feature_columns, label_min, label_max,
                    clip_min, clip_max, metrics, status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'draft')
                   ON DUPLICATE KEY UPDATE artifact_path=VALUES(artifact_path),
                   feature_columns=VALUES(feature_columns), label_min=VALUES(label_min),
                   label_max=VALUES(label_max), clip_min=VALUES(clip_min),
                   clip_max=VALUES(clip_max), metrics=VALUES(metrics), status='draft'""",
                (MODEL_NAME, version, str(model_dir), json.dumps(feature_cols), label_min, label_max,
                 CLIP_MIN, CLIP_MAX, json.dumps({"samples": samples, "split": "time", "preprocessing": "train_only", **cutoffs})),
            )
        conn.commit()


def run_clean_chronological_training_job(portfolio_code: str = "main", model_version: str | None = DEFAULT_VERSION) -> dict:
    """Train and register a non-active chronological draft model."""
    del portfolio_code  # Kept for CLI job compatibility; training uses the full universe.
    model_version = model_version or DEFAULT_VERSION
    tf.random.set_seed(42)
    np.random.seed(42)
    clean, feature_cols, mins, maxs, label_min, label_max, cutoffs = _clean_train_only_dataset()
    train_mask = clean["trade_date"] <= pd.to_datetime(cutoffs["train_end"])
    validation_mask = (clean["trade_date"] > pd.to_datetime(cutoffs["train_end"])) & (clean["trade_date"] <= pd.to_datetime(cutoffs["val_end"]))
    test_mask = clean["trade_date"] > pd.to_datetime(cutoffs["val_end"])

    def normalize(features: pd.DataFrame) -> np.ndarray:
        clipped = features.clip(mins, maxs, axis=1)
        return ((clipped - mins) / (maxs - mins + 1e-8)).to_numpy(dtype=np.float32)

    X_train = normalize(clean.loc[train_mask, feature_cols])
    X_validation = normalize(clean.loc[validation_mask, feature_cols])
    y_train = ((clean.loc[train_mask, "target"] - label_min) / (label_max - label_min + 1e-8)).to_numpy(dtype=np.float32)
    y_validation = ((clean.loc[validation_mask, "target"] - label_min) / (label_max - label_min + 1e-8)).to_numpy(dtype=np.float32)

    logger.info("Training clean chronological draft: train=%d validation=%d test=%d", train_mask.sum(), validation_mask.sum(), test_mask.sum())
    model = _build_model(len(feature_cols))
    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_validation, y_validation),
        verbose=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    model_dir = Path(__file__).resolve().parents[2] / "model" / MODEL_NAME / model_version
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "model.keras")
    _export_onnx(model, model_dir)
    pd.DataFrame({"col": list(feature_cols) + ["target"], "value": list(mins) + [label_min]}).to_csv(model_dir / "mins.csv", index=False)
    pd.DataFrame({"col": list(feature_cols) + ["target"], "value": list(maxs) + [label_max]}).to_csv(model_dir / "maxs.csv", index=False)
    (model_dir / "config.json").write_text(json.dumps({"feature_columns": feature_cols, "label_min": label_min, "label_max": label_max, "clip_min": CLIP_MIN, "clip_max": CLIP_MAX, "split": "time", "train_end": cutoffs["train_end"], "val_end": cutoffs["val_end"], "preprocessing": "train_only_winsorization_and_scaling"}, indent=2), encoding="utf-8")
    _register_draft(model_version, model_dir, feature_cols, label_min, label_max, int(len(clean)), cutoffs)
    return {"status": "ok", "version": model_version, "artifact_path": str(model_dir), "registry_status": "draft", "train_rows": int(train_mask.sum()), "validation_rows": int(validation_mask.sum()), "test_rows": int(test_mask.sum()), "train_end": cutoffs["train_end"], "val_end": cutoffs["val_end"]}
