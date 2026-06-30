#!/usr/bin/env bash
# Launch the MedViT GPU batch runner DETACHED so it survives logout/SSH drop.
#
# Usage:
#   ./run.sh                         # run all sections, all defaults
#   ./run.sh --sections plain,supcon # this node does plain + supcon
#   ./run.sh --sections augmentation # the OTHER DGX Spark does the aug sweep
#   ./run.sh --smoke-test            # quick end-to-end sanity check
#
# Any extra args are passed straight through to run_medvit.py.
#
# The job is started with setsid + nohup, fully detached from the terminal, so
# closing the SSH session does NOT kill it. Logs stream to results/logs/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HERE}/.venv"
LOG_DIR="${HERE}/results/logs"
mkdir -p "${LOG_DIR}"

# Pick the interpreter: prefer the for-gpu venv, then a CUDA-capable system python.
if [[ -x "${VENV}/bin/python" ]]; then
  PY="${VENV}/bin/python"
else
  PY="$(command -v python3 || true)"
  echo "WARNING: ${VENV} not found; falling back to '${PY}'." >&2
  echo "         Run ./setup_env.sh first to create a GB10-ready environment." >&2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAMP="$(date +%Y%m%d_%H%M%S)"
HOST="$(hostname -s)"
LOG_FILE="${LOG_DIR}/run_${HOST}_${STAMP}.log"
PID_FILE="${LOG_DIR}/run_${HOST}_${STAMP}.pid"

echo "Launching MedViT GPU runner detached..."
echo "  python : ${PY}"
echo "  args   : $*"
echo "  log    : ${LOG_FILE}"

# setsid -> new session (no controlling terminal); nohup -> ignore SIGHUP;
# trailing & -> background. stdout+stderr go to the log; stdin from /dev/null.
setsid nohup "${PY}" "${HERE}/run_medvit.py" "$@" \
  >"${LOG_FILE}" 2>&1 </dev/null &

CHILD_PID=$!
echo "${CHILD_PID}" >"${PID_FILE}"

echo
echo "Started (PID ${CHILD_PID}). It will keep running after you log out."
echo "  follow logs : tail -f ${LOG_FILE}"
echo "  check alive : ps -p ${CHILD_PID}"
echo "  stop it     : kill ${CHILD_PID}"
