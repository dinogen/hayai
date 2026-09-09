#!/usr/bin/env bash
set -euo pipefail

# HAYAI v2 - Check-up diagnostico read-only del modello

on_error() {
    local exit_code=$?
    echo >&2
    echo "Diagnostica interrotta (exit code: $exit_code, riga: $1)." >&2
    echo "Rilancia con: bash -x ./avvia_diagnostic.sh" >&2
    exit "$exit_code"
}

trap 'on_error $LINENO' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="$SCRIPT_DIR/venv-tf/bin/python"
OUTPUT_DIR="${HAYAI_DIAGNOSTIC_OUTPUT_DIR:-$SCRIPT_DIR/diagnostic-output}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Errore: interprete non trovato: $PYTHON_BIN" >&2
    exit 1
fi

export PYTHONPATH="$SCRIPT_DIR/hayai-new"

PORTFOLIO="${HAYAI_PORTFOLIO:-main}"
VERSION="${HAYAI_MODEL_VERSION:-v2}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec "$PYTHON_BIN" -m app.cli --help
fi

mkdir -p "$OUTPUT_DIR"

if [[ $# -gt 0 ]]; then
    EXTRA_ARGS=("$@")
else
    EXTRA_ARGS=()
fi

echo "========================================================"
echo " HAYAI v2: Check-up diagnostico modello"
echo "========================================================"
echo "Portfolio: $PORTFOLIO"
echo "Modello:   $VERSION"
echo "Output:    $OUTPUT_DIR"
echo

echo "Avvio diagnostica read-only..."
"$PYTHON_BIN" -m app.cli diagnostic \
    --portfolio "$PORTFOLIO" \
    --version "$VERSION" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"

echo
echo "Diagnostica completata. File prodotti:"
echo "  $OUTPUT_DIR/hayai_diagnostic_report.md"
echo "  $OUTPUT_DIR/hayai_predictions.csv"

if [[ "${HAYAI_PAUSE_ON_EXIT:-0}" == "1" && -t 0 ]]; then
    read -r -p "Premi Invio per chiudere..."
fi
