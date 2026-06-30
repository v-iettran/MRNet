#!/usr/bin/env bash
# One-time environment setup for the MedViT GPU runner on a DGX Spark.
#
# The repo's existing AI-for-MIA/.venv is an x86 (Colab) build and CANNOT run on
# the DGX Spark's ARM64 (aarch64) CPU ("Exec format error"). This script builds a
# fresh, architecture-correct virtualenv at codes/for-gpu/.venv with a PyTorch
# build that targets this node's CUDA toolkit.
#
# Usage:
#   ./setup_env.sh            # auto-detect CUDA major version, install GPU torch
#   TORCH_INDEX=cu128 ./setup_env.sh   # force a specific PyTorch CUDA channel
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HERE}/.venv"

echo "== Architecture / CUDA check =="
ARCH="$(uname -m)"
echo "  arch: ${ARCH}"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -2 | sed 's/^/  /'
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/  GPU: /'
fi

# Choose a PyTorch CUDA channel. CUDA 13 -> cu130; CUDA 12.x -> cu128 (forward
# compatible). Override with the TORCH_INDEX env var if needed.
if [[ -z "${TORCH_INDEX:-}" ]]; then
  CUDA_MAJOR="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -1)"
  case "${CUDA_MAJOR}" in
    13) TORCH_INDEX="cu130" ;;
    12) TORCH_INDEX="cu128" ;;
    *)  TORCH_INDEX="cu130" ;;  # GB10 default
  esac
fi
echo "  PyTorch channel: ${TORCH_INDEX}"

echo "== Creating venv at ${VENV} =="
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "== Installing PyTorch (${TORCH_INDEX}, ${ARCH}) =="
# Try the CUDA-specific channel first; fall back to the default PyPI build.
if ! python -m pip install torch torchvision \
      --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"; then
  echo "  ${TORCH_INDEX} channel failed; trying default PyPI torch build..." >&2
  python -m pip install torch torchvision
fi

echo "== Installing project requirements =="
python -m pip install -r "${HERE}/requirements.txt"

echo "== Verifying torch sees the GPU =="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    cc = torch.cuda.get_device_capability(0)
    print("compute capability:", cc)
PY

echo
echo "Done. Next:  ./run.sh --smoke-test    (quick end-to-end check)"
echo "        then ./run.sh                 (full run, detached)"
