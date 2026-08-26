#!/usr/bin/env bash
set -euo pipefail

# HAYAI v2 - Ciclo Notturno Completo
# data - news - sentiment - predict - signal - recommend - summaries

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Attiva l'ambiente virtuale e imposta il PYTHONPATH su hayai-new
source "$SCRIPT_DIR/venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/hayai-new"

run_job() {
    local step="$1"
    local label="$2"
    echo "[$step/7] $label..."
    python -m app.cli "$step_cmd"
}

echo "========================================================"
echo " HAYAI v2: Ciclo Notturno Completo"
echo " data - news - sentiment - predict - signal - recommend"
echo "========================================================"

declare -a JOBS=(
    "data|Job data - aggiornamento prezzi giornalieri (yfinance)"
    "news|Job news - scarico notizie per gli strumenti monitorati"
    "sentiment|Job sentiment - analisi notizie con DeepSeek"
    "predict|Job predict - quant_score dal modello ONNX"
    "signal|Job signal - segnale ibrido (Quant + LLM sentiment)"
    "recommend|Job recommend - composizione long/short su 5.000 EUR"
    "summaries|Job summaries - riepilogo markdown notizie e sentiment"
)

i=1
for job in "${JOBS[@]}"; do
    cmd="${job%%|*}"
    label="${job##*|}"
    echo "[$i/7] $label..."
    python -m app.cli "$cmd"
    i=$((i + 1))
done

echo
echo "Ciclo completo terminato. Apri la webapp su http://localhost:4200"
