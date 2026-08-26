#!/usr/bin/env bash
set -euo pipefail

# HAYAI v2 - FastAPI Backend
# Avvia il backend FastAPI in ascolto su http://127.0.0.1:8000
# Documentazione Swagger: http://127.0.0.1:8000/docs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Attiva l'ambiente virtuale e imposta il PYTHONPATH su hayai-new
source "$SCRIPT_DIR/venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/hayai-new"

echo "Backend in ascolto su http://127.0.0.1:8000 (Swagger: http://127.0.0.1:8000/docs)"
exec python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
