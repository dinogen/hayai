#!/usr/bin/env bash
#
# HAYAI v2 - Install & enable the FastAPI systemd service on Raspberry Pi.
# Registers hayai-api.service so the backend starts automatically at boot.
#
# Usage (on the Raspberry Pi, from the repo root /opt/hayai/hayai-new):
#   sudo scripts/install_api_service.sh
#
# Requirements:
#   - App deployed at /opt/hayai/hayai-new with venv at /opt/hayai/venv
#   - The systemd user (dinogen in hayai-api.service) owns /opt/hayai and can
#     write to logs/
#   - MariaDB running (see doc-new-app/07-operativita-batch.md)
#   - .env file present in the app root

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_SRC="${ROOT_DIR}/deploy/hayai-api.service"
SERVICE_DST="/etc/systemd/system/hayai-api.service"

if [ ! -f "${SERVICE_SRC}" ]; then
  echo "Service file not found: ${SERVICE_SRC}" >&2
  exit 1
fi

if [ ! -f "${ROOT_DIR}/.env" ]; then
  echo "WARNING: ${ROOT_DIR}/.env not found. Create it from .env.example first." >&2
fi

install -m 0644 "${SERVICE_SRC}" "${SERVICE_DST}"
systemctl daemon-reload
systemctl enable hayai-api
systemctl restart hayai-api

sleep 2
systemctl --no-pager --full status hayai-api || true

echo ""
echo "Service installed. Check it starts at boot with:"
echo "  systemctl is-enabled hayai-api   # expect: enabled"
echo "  systemctl status hayai-api"
echo "  curl http://127.0.0.1:8000/api/health"
