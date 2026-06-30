#!/usr/bin/env bash
# Launch a for-gpu job on the SECOND DGX Spark (spark-cdf0) over the QSFP link,
# DETACHED so it survives this shell / SSH disconnects.
#
# The peer is a separate unified-memory GB10. Its dataset, MedViT files, code and
# venv live at the SAME absolute paths as this node (synced via sync_peer.sh), so
# the same commands "just work" there.
#
# Usage (args after the script name are passed verbatim to the python entrypoint):
#   ./run_peer.sh run_cnn.py --backbone medvit --tag strong_contrastive --contrastive ...
#   ./run_peer.sh run_tuning.py --variant medvit_strong_cbam ...
#
# Follow the log:  ssh viet@169.254.234.202 'tail -f ~/AI-for-MIA/codes/for-gpu/results/logs/peer_*.log'
set -euo pipefail

PEER="${PEER_HOST:-169.254.234.202}"          # QSFP cable (fast path)
PEER_USER="${PEER_USER:-viet}"
REMOTE_DIR="/home/viet/AI-for-MIA/codes/for-gpu"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <python-script> [args...]" >&2
  exit 2
fi

SCRIPT="$1"; shift
STAMP="$(date +%Y%m%d_%H%M%S)"
# Derive a readable log name from the --tag / --variant if present, else the script.
NAME="$(echo "$SCRIPT" | sed 's/\.py$//')"
LOG="${REMOTE_DIR}/results/logs/peer_${NAME}_${STAMP}.log"

# Build a single remote command. Thread caps + unbuffered output mirror the local
# launchers; expandable_segments reduces CUDA allocator fragmentation.
REMOTE_CMD=$(cat <<EOF
mkdir -p "${REMOTE_DIR}/results/logs"
cd "${REMOTE_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
setsid nohup .venv/bin/python ${SCRIPT} $* > "${LOG}" 2>&1 < /dev/null &
echo "peer PID \$!"
echo "peer log ${LOG}"
EOF
)

echo "Launching on peer ${PEER_USER}@${PEER}:"
echo "  ${SCRIPT} $*"
ssh -o BatchMode=yes -o ConnectTimeout=8 "${PEER_USER}@${PEER}" "${REMOTE_CMD}"
echo
echo "Follow: ssh ${PEER_USER}@${PEER} 'tail -f ${LOG}'"
