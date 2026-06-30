#!/usr/bin/env bash
# Start a 2-node Ray cluster across both DGX Sparks over the QSFP cable.
#   head   = this node (spark-2e7f)  at HEAD_IP
#   worker = peer (spark-cdf0)        joins over the direct link
#
# Each node contributes its single GB10 (--num-gpus=1), so the cluster exposes
# 2 GPUs and tune_ray.py will run up to 2 trials concurrently.
#
# IMPORTANT: only bring the cluster up when the GPUs are free of non-Ray jobs.
# Ray doesn't know about processes it didn't launch, so it would schedule a
# trial onto a GPU already busy with (e.g.) a training run and cause contention.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAY="${HERE}/.venv/bin/ray"
HEAD_IP="${HEAD_IP:-169.254.34.253}"        # this node's QSFP cable address
PORT="${RAY_PORT:-6379}"
PEER="${PEER_HOST:-169.254.234.202}"
PEER_USER="${PEER_USER:-viet}"
REMOTE_DIR="/home/viet/AI-for-MIA/codes/for-gpu"

export PYTHONUNBUFFERED=1

echo "[ray_up] starting HEAD on ${HEAD_IP}:${PORT}"
"${RAY}" stop >/dev/null 2>&1 || true
"${RAY}" start --head --node-ip-address="${HEAD_IP}" --port="${PORT}" \
  --num-gpus=1 --num-cpus="$(nproc)" --dashboard-host=127.0.0.1

echo "[ray_up] starting WORKER on ${PEER_USER}@${PEER}"
ssh -o BatchMode=yes -o ConnectTimeout=8 "${PEER_USER}@${PEER}" \
  "cd ${REMOTE_DIR} && export PYTHONUNBUFFERED=1 && \
   .venv/bin/ray stop >/dev/null 2>&1; \
   .venv/bin/ray start --address=${HEAD_IP}:${PORT} \
     --num-gpus=1 --num-cpus=\$(nproc)"

sleep 4
echo "[ray_up] cluster status:"
"${RAY}" status
