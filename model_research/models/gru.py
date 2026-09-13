"""
EXP-012 — GRU per la previsione dei rendimenti.

Architettura:
    Input(window, n_features)
    → GRU(64, return_sequences=False)
    → Dense(32, relu)
    → Dense(3, activation)     ← Y5, Y10, Y15

Opzionale con 2 layer GRU:
    → GRU(64, return_sequences=True)
    → GRU(32, return_sequences=False)
    → Dense(32, relu)
    → Dense(3, activation)

Differenza rispetto al DNN:
    - Input 3D (window, n_features) — nessun flatten
    - GRU processa la sequenza temporale preservando l'ordine
    - return_sequences=True solo se ci sono layer GRU successivi

Normalizzazione: identica al DNN (per feature, fit solo su train).

Usage:
    python run_experiment.py --exp EXP-012
    python -m models.gru
    python -m models.gru --units 64 32 --dropout 0.1 --recurrent-dropout 0.1
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
from models.dnn import _get_activation, _tf, fit_scaler, transform
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


# ── Model builder ─────────────────────────────────────────────────────────────

def build_gru(
    window:             int,
    n_features:         int,
    units:              list[int] = (64,),
    dense_units:        int       = 32,
    n_outputs:          int       = 3,
    output_activation:  str       = "linear",
    dropout:            float     = 0.0,
    recurrent_dropout:  float     = 0.0,
    lr:                 float     = DEFAULT_LR,
    loss:               str       = "mse",
    seed:               int       = DEFAULT_SEED,
):
    tf = _tf()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inp = tf.keras.Input(shape=(window, n_features), name="input")
    x   = inp

    for i, u in enumerate(units):
        return_seq = i < len(units) - 1   # True per tutti tranne l'ultimo
        x = tf.keras.layers.GRU(
            u,
            return_sequences  = return_seq,
            dropout           = dropout,
            recurrent_dropout = recurrent_dropout,
            name              = f"gru_{i}",
        )(x)

    x   = tf.keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    out = tf.keras.layers.Dense(
        n_outputs,
        activation = _get_activation(output_activation),
        name       = "output",
    )(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
        loss      = loss,
    )
    return model


# ── Training di un singolo fold ───────────────────────────────────────────────

def _train_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    *,
    window: int, n_features: int,
    units: list[int], dense_units: int,
    output_activation: str, dropout: float, recurrent_dropout: float,
    lr: float, loss: str, epochs: int, batch_size: int, patience: int, seed: int,
):
    tf = _tf()
    model = build_gru(
        window=window, n_features=n_features,
        units=units, dense_units=dense_units,
        n_outputs=y_tr.shape[1], output_activation=output_activation,
        dropout=dropout, recurrent_dropout=recurrent_dropout,
        lr=lr, loss=loss, seed=seed,
    )
    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data = (X_va, y_va),
        epochs          = epochs,
        batch_size      = batch_size,
        callbacks       = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=0,
        )],
        verbose = 0,
    )
    best_epoch    = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = float(np.min(history.history["val_loss"]))
    return model, {
        "best_epoch":    best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_s":     round(time.time() - t0, 1),
        "total_epochs":  len(history.history["val_loss"]),
    }


# ── Walk-forward principale ───────────────────────────────────────────────────

def run_gru(exp: dict, out_dir: Path) -> None:
    window             = exp.get("window", DEFAULT_WINDOW)
    horizons           = exp.get("targets", TARGET_HORIZONS)
    output_activation  = exp.get("output_activation", "linear")
    loss               = exp.get("loss", "mse")
    units              = exp.get("architecture", {}).get("units", [64])
    dense_units        = exp.get("architecture", {}).get("dense_units", 32)
    dropout            = exp.get("dropout", 0.0)
    recurrent_dropout  = exp.get("recurrent_dropout", 0.0)
    lr                 = exp.get("lr", DEFAULT_LR)
    epochs             = exp.get("epochs", DEFAULT_EPOCHS)
    batch_size         = exp.get("batch_size", DEFAULT_BATCH_SIZE)
    patience           = exp.get("patience", DEFAULT_PATIENCE)
    seed               = exp.get("seed", DEFAULT_SEED)

    features_field = exp.get("features", "stock_only")
    from features.macro_features import get_macro_config
    _, dataset_tag = get_macro_config(features_field)

    print(f"\nLoading dataset (window={window}, tag={dataset_tag or 'stock_only'})...")
    X, y, meta = load_dataset(window=window, tag=dataset_tag)
    n_features = X.shape[2]
    print(f"  X={X.shape}  y={y.shape}")

    folds, _ = make_folds(meta, verbose=True)
    if not folds:
        print("No folds generated.")
        return

    print(f"\nArchitecture: GRU{units} → Dense({dense_units}) → {len(horizons)} outputs")
    print(f"Activation: {output_activation}  |  Loss: {loss}  |  Dropout: {dropout}  |  Seed: {seed}\n")

    pred_collector: list[tuple] = []
    fold_stats: list[dict]      = []

    for fold in folds:
        X_tr = X[fold.train_idx];  y_tr = y[fold.train_idx]
        X_va = X[fold.val_idx];    y_va = y[fold.val_idx]
        X_te = X[fold.test_idx]
        meta_te = meta.iloc[fold.test_idx].reset_index(drop=True)

        mean, std = fit_scaler(X_tr)
        X_tr_n    = transform(X_tr, mean, std)   # (n, window, n_features) — no flatten
        X_va_n    = transform(X_va, mean, std)
        X_te_n    = transform(X_te, mean, std)

        print(f"  Fold {fold.fold_id:02d}  train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}  ",
              end="", flush=True)

        model, stats = _train_fold(
            X_tr_n, y_tr, X_va_n, y_va,
            window=window, n_features=n_features,
            units=units, dense_units=dense_units,
            output_activation=output_activation,
            dropout=dropout, recurrent_dropout=recurrent_dropout,
            lr=lr, loss=loss, epochs=epochs, batch_size=batch_size,
            patience=patience, seed=seed + fold.fold_id,
        )

        print(f"epoch={stats['best_epoch']}/{stats['total_epochs']}  "
              f"val_loss={stats['best_val_loss']:.4f}  ({stats['elapsed_s']}s)")

        fold_dir = out_dir / f"fold_{fold.fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(fold_dir / "model.keras"))
        np.save(str(fold_dir / "scaler_mean.npy"), mean)
        np.save(str(fold_dir / "scaler_std.npy"),  std)

        y_pred = model.predict(X_te_n, verbose=0)
        pred_collector.append((y_pred, fold.test_idx, meta_te))
        fold_stats.append({"fold": fold.fold_id, **stats})

    y_pred_all = np.concatenate([p[0] for p in pred_collector], axis=0)
    idx_all    = np.concatenate([p[1] for p in pred_collector], axis=0)
    meta_all   = pd.concat([p[2]      for p in pred_collector], ignore_index=True)
    y_true_all = y[idx_all]

    fs_df = pd.DataFrame(fold_stats)
    print(f"\n{fs_df.to_string(index=False)}")
    fs_df.to_csv(out_dir / "fold_stats.csv", index=False)

    all_results = []
    for h_idx, h in enumerate(horizons):
        label   = f"{exp['id']} Y{h} GRU"
        results = evaluate(y_pred_all[:, h_idx], y_true_all[:, h_idx], meta_all, label=label)
        print_report(results)
        save_results(results, out_dir / f"metrics_y{h}.csv")
        all_results.append(results)

    df = compare_experiments(all_results)
    print(df[["spearman_mean", "spearman_p", "q5_q1_spread", "monotonic", "hit_rate"]].to_string())
    df.to_csv(out_dir / "comparison_r10.csv")
    print(f"\nResults saved to {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GRU model (EXP-012)")
    parser.add_argument("--activation",         type=str,   default="linear", choices=["linear", "arctan"])
    parser.add_argument("--loss",               type=str,   default="mse",    choices=["mse", "huber"])
    parser.add_argument("--units",              type=int,   nargs="+",        default=[64])
    parser.add_argument("--dense",              type=int,   default=32)
    parser.add_argument("--dropout",            type=float, default=0.0)
    parser.add_argument("--recurrent-dropout",  type=float, default=0.0,
                        dest="recurrent_dropout")
    parser.add_argument("--lr",                 type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs",             type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",         type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience",           type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument("--window",             type=int,   default=DEFAULT_WINDOW)
    parser.add_argument("--seed",               type=int,   default=DEFAULT_SEED)
    parser.add_argument("--tag",                type=str,   default="")
    args = parser.parse_args()

    run_id = "gru" + (f"_{args.tag}" if args.tag else "")
    exp = {
        "id":                 run_id,
        "window":             args.window,
        "targets":            TARGET_HORIZONS,
        "output_activation":  args.activation,
        "loss":               args.loss,
        "architecture":       {"units": args.units, "dense_units": args.dense},
        "dropout":            args.dropout,
        "recurrent_dropout":  args.recurrent_dropout,
        "lr":                 args.lr,
        "epochs":             args.epochs,
        "batch_size":         args.batch_size,
        "patience":           args.patience,
        "seed":               args.seed,
    }
    out = Path("experiments") / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    run_gru(exp, out)
