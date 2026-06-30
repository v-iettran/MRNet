#!/usr/bin/env bash
# Launch a CNN training (resnet50 or densenet121) DETACHED so it survives logout.
#
# Usage:
#   ./run_cnn.sh resnet50                 # full run, all defaults
#   ./run_cnn.sh densenet121              # full run
#   ./run_cnn.sh resnet50 --smoke-test    # quick end-to-end sanity check
#   ./run_cnn.sh resnet50 --epochs 30 ... # extra args pass through to run_cnn.py
#
# To run BOTH in parallel, just call this twice:
#   ./run_cnn.sh resnet50
#   ./run_cnn.sh densenet121
#
# Detached via setsid + nohup; logs stream to results/logs/.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <resnet50|densenet121> [extra args for run_cnn.py]" >&2
  exit 2
fi
BACKBONE="$1"; shift

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HERE}/.venv"
LOG_DIR="${HERE}/results/logs"
mkdir -p "${LOG_DIR}"

if [[ -x "${VENV}/bin/python" ]]; then
  PY="${VENV}/bin/python"
else
  PY="$(command -v python3 || true)"
  echo "WARNING: ${VENV} not found; falling back to '${PY}'." >&2
  echo "         Run ./setup_env.sh first to create a GB10-ready environment." >&2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Unbuffered stdout so per-epoch progress from src/training_utils' print()s shows
# up in the log promptly (redirected stdout is block-buffered otherwise).
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)"
HOST="$(hostname -s)"
LOG_FILE="${LOG_DIR}/cnn_${BACKBONE}_${HOST}_${STAMP}.log"
PID_FILE="${LOG_DIR}/cnn_${BACKBONE}_${HOST}_${STAMP}.pid"

echo "Launching ${BACKBONE} GPU training detached..."
echo "  python : ${PY}"
echo "  args   : --backbone ${BACKBONE} $*"
echo "  log    : ${LOG_FILE}"

setsid nohup "${PY}" "${HERE}/run_cnn.py" --backbone "${BACKBONE}" "$@" \
  >"${LOG_FILE}" 2>&1 </dev/null &

CHILD_PID=$!
echo "${CHILD_PID}" >"${PID_FILE}"

echo
echo "Started ${BACKBONE} (PID ${CHILD_PID}). It will keep running after you log out."
echo "  follow logs : tail -f ${LOG_FILE}"
echo "  check alive : ps -p ${CHILD_PID}"
echo "  stop it     : kill ${CHILD_PID}"
