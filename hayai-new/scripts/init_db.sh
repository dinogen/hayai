#!/usr/bin/env bash
#
# HAYAI v2 - First-time database initialization (MariaDB).
# Creates the database (if missing) and applies schema.sql + seed.sql.
#
# Usage:
#   scripts/init_db.sh            # create DB + schema + seed (safe on existing DB)
#   scripts/init_db.sh --drop     # drop the database first (full reset)
#   scripts/init_db.sh --no-seed  # create DB + schema only (no bootstrap data)
#
# Exit codes:
#   0  success
#   1  failure
#   2  invalid arguments or missing prerequisites

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

SCHEMA_FILE="${ROOT_DIR}/sql/schema.sql"
SEED_FILE="${ROOT_DIR}/sql/seed.sql"

DROP_FIRST=0
WITH_SEED=1

for arg in "$@"; do
  case "$arg" in
    --drop)    DROP_FIRST=1 ;;
    --no-seed) WITH_SEED=0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ ! -f "${SCHEMA_FILE}" ]; then
  echo "Schema file not found: ${SCHEMA_FILE}" >&2
  exit 2
fi
if [ "${WITH_SEED}" -eq 1 ] && [ ! -f "${SEED_FILE}" ]; then
  echo "Seed file not found: ${SEED_FILE}" >&2
  exit 2
fi

command -v mariadb >/dev/null 2>&1 || command -v mysql >/dev/null 2>&1 || {
  echo "mariadb/mysql client not found in PATH" >&2
  exit 2
}

export MYSQL_PWD="${DB_PASSWORD}"
MYSQL="mariadb"
command -v "${MYSQL}" >/dev/null 2>&1 || MYSQL="mysql"

CONNECT_ARGS=(--host="${DB_HOST}" --port="${DB_PORT}" --user="${DB_USER}")

if [ "${DROP_FIRST}" -eq 1 ]; then
  echo "Dropping database ${DB_NAME} (if exists)..."
  if ! "${MYSQL}" "${CONNECT_ARGS[@]}" -e "DROP DATABASE IF EXISTS \`${DB_NAME}\`;"; then
    echo "Failed to drop database ${DB_NAME}" >&2
    exit 1
  fi
fi

echo "Creating database ${DB_NAME} (utf8mb4_unicode_ci)..."
if ! "${MYSQL}" "${CONNECT_ARGS[@]}" -e "CREATE DATABASE IF NOT EXISTS \`${DB_NAME}\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"; then
  echo "Failed to create database ${DB_NAME}" >&2
  exit 1
fi

echo "Applying schema: ${SCHEMA_FILE}"
if ! "${MYSQL}" "${CONNECT_ARGS[@]}" "${DB_NAME}" < "${SCHEMA_FILE}"; then
  echo "Schema application FAILED" >&2
  exit 1
fi

if [ "${WITH_SEED}" -eq 1 ]; then
  echo "Applying seed data: ${SEED_FILE}"
  if ! "${MYSQL}" "${CONNECT_ARGS[@]}" "${DB_NAME}" < "${SEED_FILE}"; then
    echo "Seed application FAILED" >&2
    exit 1
  fi
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Database ${DB_NAME} initialized successfully."
