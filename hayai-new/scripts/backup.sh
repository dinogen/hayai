#!/usr/bin/env bash
#
# HAYAI v2 - Daily MariaDB backup (scheduled via cron).
# Dumps the hayai database to backups/ and prunes old dumps (default: keep 14).
#
# Usage:
#   scripts/backup.sh [--keep 14]

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load DB credentials from .env (source-safe: values without spaces expected).
ENV_FILE="${ROOT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

DB_HOST="${DB_HOST:-127.0.0.1}"
DB_PORT="${DB_PORT:-3306}"
DB_NAME="${DB_NAME:-hayai}"
DB_USER="${DB_USER:-hayai}"
DB_PASSWORD="${DB_PASSWORD:-}"
KEEP="${KEEP:-14}"

BACKUP_DIR="${ROOT_DIR}/backups"
mkdir -p "${BACKUP_DIR}"

STAMP="$(date '+%Y%m%d_%H%M%S')"
DUMP_FILE="${BACKUP_DIR}/hayai_${STAMP}.sql.gz"

export MYSQL_PWD="${DB_PASSWORD}"
if mariadb-dump --host="${DB_HOST}" --port="${DB_PORT}" --user="${DB_USER}" \
    --single-transaction --routines --databases "${DB_NAME}" | gzip > "${DUMP_FILE}"; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Backup OK: ${DUMP_FILE}"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Backup FAILED"
  rm -f "${DUMP_FILE}"
  exit 1
fi

# Prune old dumps, keeping the most recent ${KEEP}.
ls -1t "${BACKUP_DIR}"/hayai_*.sql.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f
