#!/usr/bin/env bash
set -euo pipefail

# HAYAI v2 - Angular Frontend
# Avvia il frontend Angular in ascolto su http://localhost:4200
# Assicurati che il backend sia avviato con avvia_backend.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/hayai-new/web"

echo "Frontend in ascolto su http://localhost:4200"
exec npm start
