"""
EXP-002 / EXP-003 — Multi-task DNN per la previsione dei rendimenti.

Architettura:
    Flatten(20 × 12 = 240)
    → Dense(128, relu) [+ Dropout opzionale]
    → Dense(64,  relu)
    → Dense(32,  relu)
    → Dense(3,   activation)     ← Y5, Y10, Y15

Output activation:
    linear  (EXP-002) : output ∈ (-∞, +∞)
    arctan  (EXP-003) : 3 * (2/π) * atan(x) ∈ (-3, +3) asintoticamente

Normalizzazione:
    StandardScaler per feature (12 medie e std) calcolato SOLO sul training set
    di ciascun fold. Applicato come (X - mean) / std su tutte le finestre.

Walk-forward:
    Per ogni fold: scaler fittato su train, modello addestrato da zero,
    previsioni raccolte sul test. Poi concatenate su tutti i fold.

Usage:
    python run_experiment.py --exp EXP-002
    python run_experiment.py --exp EXP-003
    python -m models.dnn --activation linear
    python -m models.dnn --activation arctan --dropout 0.1
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import compare_experiments, evaluate, print_report, save_results
from features.stock_features import FEATURE_NAMES, load_dataset
from settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_WINDOW,
    TARGET_HORIZONS,
)
from training.walk_forward import make_folds

# ── Tensorflow lazy import (evita il log di TF all'import del modulo) ─────────

def _tf():
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    return tf


# ── Output activation ─────────────────────────────────────────────────────────

def _scaled_arctan(x):
    tf = _tf()
    return 3.0 * (2.0 / np.pi) * tf.math.atan(x)


def _get_activation(name: str):
    if name == "linear":
        return "linear"
    if name == "arctan":
        return _scaled_arctan
    raise ValueError(f"Unknown output activation: {name!r}. Use 'linear' or 'arctan'.")


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(
    input_dim:         int,
    layers:            list[int] = (128, 64, 32),
    n_outputs:         int       = 3,
    output_activation: str       = "linear",
    dropout:           float     = 0.0,
    lr:                float     = DEFAULT_LR,
    loss:              str       = "mse",
    seed:              int       = DEFAULT_SEED,
):
    tf = _tf()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inp = tf.keras.Input(shape=(input_dim,), name="input")
    x   = inp
    for i, units in enumerate(layers):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout, seed=seed + i, name=f"drop_{i}")(x)

    out = tf.keras.layers.Dense(
        n_outputs,
        activation=_get_activation(output_activation),
        name="output",
    )(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
    )
    return model


# ── Normalizzazione ───────────────────────────────────────────────────────────

def fit_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcola mean e std per ciascuna delle n_features colonne,
    su tutti i campioni e tutti i timestep del training set.

    X_train : (n_samples, window, n_features)
    Returns : mean (n_features,), std (n_features,)
    """
    n, w, f = X_train.shape
    flat    = X_train.reshape(-1, f)           # (n*window, n_features)
    mean    = flat.mean(axis=0)
    std     = flat.std(axis=0) + 1e-8
    return mean, std


def transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Applica StandardScaler per feature. Broadcasting: (n, w, f) - (f,)."""
    return (X - mean) / std


# ── Training di un singolo fold ───────────────────────────────────────────────

def train_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    layers:            list[int],
    output_activation: str,
    dropout:           float,
    lr:                float,
    loss:              str,
    epochs:            int,
    batch_size:        int,
    patience:          int,
    seed:              int,
    verbose:           int = 0,
):
    tf = _tf()

    n_features = X_tr.shape[1]   # già flattened

    model = build_model(
        input_dim         = n_features,
        layers            = layers,
        n_outputs         = y_tr.shape[1],
        output_activation = output_activation,
        dropout           = dropout,
        lr                = lr,
        loss              = loss,
        seed              = seed,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor   = "val_loss",
            patience  = patience,
            restore_best_weights = True,
            verbose   = 0,
        )
    ]

    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data = (X_va, y_va),
        epochs          = epochs,
        batch_size      = batch_size,
        callbacks       = callbacks,
        verbose         = verbose,
    )
    elapsed = time.time() - t0

    best_epoch    = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = float(np.min(history.history["val_loss"]))

    return model, {
        "best_epoch":    best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_s":     round(elapsed, 1),
        "total_epochs":  len(history.history["val_loss"]),
    }


# ── Walk-forward principale ───────────────────────────────────────────────────

def run_dnn(exp: dict, out_dir: Path) -> None:
    """Entry point per run_experiment.py (EXP-002, EXP-003)."""
    window            = exp.get("window", DEFAULT_WINDOW)
    horizons          = exp.get("targets", TARGET_HORIZONS)
    output_activation = exp.get("output_activation", "linear")
    loss              = exp.get("loss", "mse")
    layers            = exp.get("architecture", {}).get("layers", [128, 64, 32])
    dropout           = exp.get("dropout", 0.0)
    lr                = exp.get("lr", DEFAULT_LR)
    epochs            = exp.get("epochs", DEFAULT_EPOCHS)
    batch_size        = exp.get("batch_size", DEFAULT_BATCH_SIZE)
    patience          = exp.get("patience", DEFAULT_PATIENCE)
    seed              = exp.get("seed", DEFAULT_SEED)

    features_field = exp.get("features", "stock_only")
    from features.macro_features import get_macro_config
    _, dataset_tag = get_macro_config(features_field)

    print(f"\nLoading dataset (window={window}, tag={dataset_tag or 'stock_only'})...")
    X, y, meta = load_dataset(window=window, tag=dataset_tag)
    print(f"  X={X.shape}  y={y.shape}")

    folds, _ = make_folds(meta, verbose=True)
    if not folds:
        print("No folds generated.")
        return

    n_features_flat = window * len(FEATURE_NAMES)
    print(f"\nArchitecture: {n_features_flat} → {layers} → {len(horizons)} outputs")
    print(f"Activation: {output_activation}  |  Loss: {loss}  |  Seed: {seed}\n")

    # accumulate predictions across folds
    pred_collector: list[tuple[np.ndarray, np.ndarray, pd.DataFrame]] = []
    fold_stats: list[dict] = []

    for fold in folds:
        X_tr = X[fold.train_idx];  y_tr = y[fold.train_idx]
        X_va = X[fold.val_idx];    y_va = y[fold.val_idx]
        X_te = X[fold.test_idx];   y_te = y[fold.test_idx]
        meta_te = meta.iloc[fold.test_idx].reset_index(drop=True)

        # normalise per feature (fit on train only)
        mean, std     = fit_scaler(X_tr)
        X_tr_n        = transform(X_tr, mean, std).reshape(len(X_tr), -1)
        X_va_n        = transform(X_va, mean, std).reshape(len(X_va), -1)
        X_te_n        = transform(X_te, mean, std).reshape(len(X_te), -1)

        print(f"  Fold {fold.fold_id:02d}  "
              f"train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}  ",
              end="", flush=True)

        model, stats = train_fold(
            X_tr_n, y_tr, X_va_n, y_va,
            layers            = layers,
            output_activation = output_activation,
            dropout           = dropout,
            lr                = lr,
            loss              = loss,
            epochs            = epochs,
            batch_size        = batch_size,
            patience          = patience,
            seed              = seed + fold.fold_id,
        )

        print(f"epoch={stats['best_epoch']}/{stats['total_epochs']}  "
              f"val_loss={stats['best_val_loss']:.4f}  "
              f"({stats['elapsed_s']}s)")

        # save model weights per fold
        fold_dir = out_dir / f"fold_{fold.fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(fold_dir / "model.keras"))
        np.save(str(fold_dir / "scaler_mean.npy"), mean)
        np.save(str(fold_dir / "scaler_std.npy"),  std)

        y_pred = model.predict(X_te_n, verbose=0)   # (n_test, n_horizons)
        pred_collector.append((y_pred, fold.test_idx, meta_te))
        fold_stats.append({"fold": fold.fold_id, **stats})

    # concatenate across folds
    y_pred_all = np.concatenate([p[0]  for p in pred_collector], axis=0)
    idx_all    = np.concatenate([p[1]  for p in pred_collector], axis=0)
    meta_all   = pd.concat(    [p[2]   for p in pred_collector], ignore_index=True)
    y_true_all = y[idx_all]

    # fold training summary
    print(f"\n{'─'*55}")
    print(f"  Fold training summary")
    print(f"{'─'*55}")
    fs_df = pd.DataFrame(fold_stats)
    print(fs_df.to_string(index=False))
    fs_df.to_csv(out_dir / "fold_stats.csv", index=False)

    # evaluate each horizon
    all_results = []
    for h_idx, h in enumerate(horizons):
        label   = f"{exp['id']} Y{h} ({output_activation})"
        results = evaluate(y_pred_all[:, h_idx], y_true_all[:, h_idx], meta_all, label=label)
        print_report(results)
        save_results(results, out_dir / f"metrics_y{h}.csv")
        all_results.append(results)

    # R10 comparison across horizons
    print(f"\n── Comparison across horizons ──────────────────────────────────\n")
    df = compare_experiments(all_results)
    print(df[["spearman_mean", "spearman_p", "q5_q1_spread", "monotonic", "hit_rate"]].to_string())
    df.to_csv(out_dir / "comparison_r10.csv")

    # save raw predictions for portfolio simulation
    pred_df = meta_all.copy()
    for h_idx, h in enumerate(horizons):
        pred_df[f"pred_y{h}"] = y_pred_all[:, h_idx]
        pred_df[f"true_y{h}"] = y_true_all[:, h_idx]
    pred_df.to_parquet(out_dir / "predictions.parquet")

    print(f"\nResults saved to {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DNN model (EXP-002/003)")
    parser.add_argument("--activation", type=str,   default="linear",  choices=["linear", "arctan"])
    parser.add_argument("--loss",       type=str,   default="mse",     choices=["mse", "huber"])
    parser.add_argument("--layers",     type=int,   nargs="+",         default=[128, 64, 32])
    parser.add_argument("--dropout",    type=float, default=0.0)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience",   type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument("--window",     type=int,   default=DEFAULT_WINDOW)
    parser.add_argument("--seed",       type=int,   default=DEFAULT_SEED)
    parser.add_argument("--tag",        type=str,   default="", help="Suffix for output directory")
    args = parser.parse_args()

    run_id = f"dnn_{args.activation}" + (f"_{args.tag}" if args.tag else "")

    exp = {
        "id":                 run_id,
        "window":             args.window,
        "targets":            TARGET_HORIZONS,
        "output_activation":  args.activation,
        "loss":               args.loss,
        "architecture":       {"layers": args.layers},
        "dropout":            args.dropout,
        "lr":                 args.lr,
        "epochs":             args.epochs,
        "batch_size":         args.batch_size,
        "patience":           args.patience,
        "seed":               args.seed,
    }

    out = Path("experiments") / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    run_dnn(exp, out)
