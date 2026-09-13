"""
Smoke test per evaluation/metrics.py.

Carica il dataset reale e valuta tre scenari:
  - random   : previsioni casuali (atteso Spearman ≈ 0)
  - oracle   : previsioni perfette = y_true (atteso Spearman = 1)
  - inverse  : previsioni invertite = -y_true (atteso Spearman = -1)

Utile per verificare che le metriche funzionino correttamente
prima di addestrare qualsiasi modello.

Lancio:
    python test_metrics.py
    python test_metrics.py --window 10
    python test_metrics.py --fold 0      # usa solo il primo fold di test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from evaluation.metrics import compare_experiments, evaluate, print_report
from features.stock_features import load_dataset
from settings import DEFAULT_WINDOW
from training.walk_forward import make_folds


def main(window: int, fold_id: int | None) -> None:
    print(f"Loading dataset (window={window})...")
    X, y, meta = load_dataset(window=window)
    print(f"  X={X.shape}  y={y.shape}  meta={meta.shape}\n")

    # usa il primo fold di test disponibile
    folds, holdout_idx = make_folds(meta, verbose=False)
    if not folds:
        print("Nessun fold generato — il dataset potrebbe essere troppo corto.")
        sys.exit(1)

    f = folds[fold_id if fold_id is not None else 0]
    print(f"Fold usato: {f}\n")

    idx  = f.test_idx
    meta_test = meta.iloc[idx].reset_index(drop=True)

    # Y5 (indice 0)
    y_true = y[idx, 0]

    rng = np.random.default_rng(42)

    scenarios = {
        "RANDOM (atteso Spearman ≈ 0)":   rng.standard_normal(len(y_true)),
        "ORACLE (atteso Spearman = 1)":    y_true.copy(),
        "INVERSE (atteso Spearman = -1)":  -y_true.copy(),
    }

    all_results = []
    for label, y_pred in scenarios.items():
        r = evaluate(y_pred, y_true, meta_test, label=label)
        print_report(r)
        all_results.append(r)

    print("\n── Tabella comparativa ──────────────────────────────────────────\n")
    df = compare_experiments(all_results)
    print(df[["spearman_mean", "spearman_p", "q5_q1_spread", "monotonic", "hit_rate"]].to_string())
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--fold",   type=int, default=None, help="Indice del fold da usare (default: 0)")
    args = parser.parse_args()
    main(window=args.window, fold_id=args.fold)
