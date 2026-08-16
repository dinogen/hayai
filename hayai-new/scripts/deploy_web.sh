#!/usr/bin/env bash
#
# HAYAI v2 - Deploy the Angular SPA and the nginx site on Raspberry Pi.
# Copies the production build to /var/www/hayai and enables the nginx site
# that serves the SPA and reverse-proxies /api/ to the FastAPI backend.
#
# Usage (on the Raspberry Pi, from the repo root /opt/hayai/hayai-new):
#   sudo scripts/deploy_web.sh
#
# Prerequisites:
#   - Production build done on dev PC (npm run build in web/) and deployed
#   - nginx installed and running
#   - hayai-api.service running (FastAPI on 127.0.0.1:8000)

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/web/dist/web/browser"
SITE_SRC="${ROOT_DIR}/deploy/nginx-hayai.conf"
SITE_DST="/etc/nginx/sites-available/hayai"
SITE_LINK="/etc/nginx/sites-enabled/hayai"
WWW_DIR="/var/www/hayai"

if [ ! -f "${BUILD_DIR}/index.html" ]; then
  echo "Build not found: ${BUILD_DIR}. Run 'npm run build' in web/ first." >&2
  exit 1
fi

if [ ! -f "${SITE_SRC}" ]; then
  echo "nginx site config not found: ${SITE_SRC}" >&2
  exit 1
fi

# 1. Copy the SPA build
mkdir -p "${WWW_DIR}"
cp -r "${BUILD_DIR}/." "${WWW_DIR}/"
echo "SPA copied to ${WWW_DIR}"

# 2. Install the nginx site
install -m 0644 "${SITE_SRC}" "${SITE_DST}"
ln -sf "${SITE_DST}" "${SITE_LINK}"

# 3. Remove the default site if it is still linked
if [ -f /etc/nginx/sites-enabled/default ]; then
  rm -f /etc/nginx/sites-enabled/default
  echo "Removed default nginx site"
fi

# 4. Test and reload nginx
nginx -t
systemctl reload nginx

echo ""
echo "Deployed. Open http://<raspberry-pi-ip>/ in the browser."
echo "Check: curl -s http://127.0.0.1/api/health"
