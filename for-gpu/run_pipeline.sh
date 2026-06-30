#!/usr/bin/env bash
# Launch the FULL study pipeline (orchestrate.py) DETACHED so it survives logout.
#
# Phases (single node, 2-way parallel by default):
#   A) steps 1 & 2 : strong-aug baselines + CBAM + contrastive, for densenet121 & medvit
#   B) step 3      : auto-pick best variant per architecture, random-search tuning
#
# Usage:
#   ./run_pipeline.sh                 # full study, all defaults
#   ./run_pipeline.sh --smoke-test    # tiny end-to-end validation of the whole flow
#   ./run_pipeline.sh --epochs 40 ... # extra args pass through to orchestrate.py
#
# Detached via setsid + nohup; the orchestrator log streams to results/logs/.
# Each individual training/tuning job also gets its own log in results/logs/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HERE}/.venv"
LOG_DIR="${HERE}/results/logs"
mkdir -p "${LOG_DIR}"

if [[ -x "${VENV}/bin/python" ]]; then
  PY="${VENV}/bin/python"
else
  PY="$(command -v python3 || true)"
  echo "WARNING: ${VENV} not found; falling back to '${PY}'." >&2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)"
HOST="$(hostname -s)"
LOG_FILE="${LOG_DIR}/pipeline_${HOST}_${STAMP}.log"
PID_FILE="${LOG_DIR}/pipeline_${HOST}_${STAMP}.pid"

echo "Launching study pipeline detached..."
echo "  python : ${PY}"
echo "  args   : $*"
echo "  log    : ${LOG_FILE}"

setsid nohup "${PY}" "${HERE}/orchestrate.py" "$@" \
  >"${LOG_FILE}" 2>&1 </dev/null &

CHILD_PID=$!
echo "${CHILD_PID}" >"${PID_FILE}"

echo
echo "Started pipeline (PID ${CHILD_PID}). It will keep running after you log out."
echo "  follow orchestrator : tail -f ${LOG_FILE}"
echo "  per-job logs        : ls ${LOG_DIR}/pipe_*_${STAMP}.log"
echo "  stop everything     : kill -- -${CHILD_PID}   # kills the whole process group"
