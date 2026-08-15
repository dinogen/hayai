#!/usr/bin/env bash
#
# HAYAI v2 - Nightly batch cycle (scheduled via cron on Raspberry Pi).
# Runs the sequential batch jobs and logs the whole cycle to logs/nightly.log.
#
# Usage:
#   scripts/run_nightly.sh [--skip-cleanup] [--portfolio main]
#
# Exit codes:
#   0  all jobs succeeded
#   1  at least one job failed

set -u


# Fixed paths for the Raspberry Pi deploy.
SCRIPT_DIR="/opt/hayai/hayai-new/scripts"
ROOT_DIR="/opt/hayai/hayai-new"


# Fixed virtualenv location for the Raspberry Pi deploy.
PYTHON="/opt/hayai/venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "venv not found at ${PYTHON}" >&2
  exit 2
fi


LOG_FILE="${ROOT_DIR}/logs/nightly.log"
PORTFOLIO="main"
SKIP_CLEANUP=0

PYTHON="/opt/hayai/venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "venv not found at ${PYTHON}" >&2
  exit 2
fi


for arg in "$@"; do
  case "$arg" in
    --skip-cleanup) SKIP_CLEANUP=1 ;;
    --portfolio=*)  PORTFOLIO="${arg#*=}" ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "${LOG_FILE}")"

# Ordered nightly jobs (dependencies: data -> metadata -> news -> ... -> summaries).
JOBS=(
  "data"
  "metadata"
  "news"
  "sentiment"
  "predict"
  "signal"
  "recommend"
  "nav"
  "summaries"
)
if [ "${SKIP_CLEANUP}" -eq 0 ]; then
  JOBS+=("cleanup")
fi

started=$(date '+%Y-%m-%d %H:%M:%S')
{
  echo "=============================================="
  echo "Nightly cycle started: ${started} (portfolio: ${PORTFOLIO})"
} >> "${LOG_FILE}"

FAILED=0

for job in "${JOBS[@]}"; do
  job_start=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[${job_start}] Running job: ${job}" >> "${LOG_FILE}"

  cd "${ROOT_DIR}" && "${PYTHON}" -m app.cli "${job}" --portfolio "${PORTFOLIO}" >> "${LOG_FILE}" 2>&1
  exit_code=$?

  if [ ${exit_code} -ne 0 ]; then
    FAILED=1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job FAILED: ${job} (exit ${exit_code})" >> "${LOG_FILE}"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job OK: ${job}" >> "${LOG_FILE}"
  fi
done

finished=$(date '+%Y-%m-%d %H:%M:%S')
if [ ${FAILED} -eq 0 ]; then
  echo "[${finished}] Nightly cycle completed successfully." >> "${LOG_FILE}"
  exit 0
else
  echo "[${finished}] Nightly cycle finished WITH ERRORS." >> "${LOG_FILE}"
  exit 1
fi
