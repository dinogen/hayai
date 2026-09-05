#!/usr/bin/env bash
set -euo pipefail

# HAYAI v2 - Ingestion, Training, Verify & Backtest (100 Assets, 5 Years)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Attiva l'ambiente virtuale (venv-tf: Python 3.13 + TensorFlow) e imposta il PYTHONPATH su hayai-new
source "$SCRIPT_DIR/venv-tf/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/hayai-new"

echo "========================================================"
echo " HAYAI v2: Download dati (5y), Training, Verify e Backtest"
echo "========================================================"

echo
echo "=== [1/3] Ingestion e Training v2 ==="
python -m app.jobs.train_universe_pipeline

echo
echo "=== [2/3] Verifica modello v2 ==="
python -m app.cli verify --version v2

echo
echo "=== [3/3] Backtest selezione v2 ==="
python -m app.cli backtest --version v2

echo
echo "Pipeline completata con successo."
