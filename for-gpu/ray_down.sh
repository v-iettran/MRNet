#!/usr/bin/env bash
# Tear down the 2-node Ray cluster (worker first, then head).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAY="${HERE}/.venv/bin/ray"
PEER="${PEER_HOST:-169.254.234.202}"
PEER_USER="${PEER_USER:-viet}"
REMOTE_DIR="/home/viet/AI-for-MIA/codes/for-gpu"

echo "[ray_down] stopping WORKER on ${PEER_USER}@${PEER}"
ssh -o BatchMode=yes -o ConnectTimeout=8 "${PEER_USER}@${PEER}" \
  "cd ${REMOTE_DIR} && .venv/bin/ray stop" || true

echo "[ray_down] stopping HEAD"
"${RAY}" stop || true
echo "[ray_down] done"
