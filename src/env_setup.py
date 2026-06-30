"""Shared runtime/environment wiring for the headless GPU runners.

Owner: shared.

This module centralises the environment setup that used to be copy-pasted across
``for-gpu/run_cnn.py``, ``for-gpu/run_medvit.py`` and the other GPU scripts, so
all of them resolve paths and cap threads the SAME way. The notebooks rely on
``src.config`` for the equivalent path resolution (their Colab bootstrap cell
puts ``codes/`` on ``sys.path`` first).

Functions
---------
cap_cpu_threads()      Limit BLAS/OpenMP threads PER PROCESS so DataLoader
                       workers don't oversubscribe the CPU. MUST run before
                       torch/numpy are imported to take effect.
add_codes_to_path()    Put ``codes/`` on ``sys.path`` so ``import src...`` works.
resolve_data_dir()     Locate the MRNet data folder (``mrnet/`` or ``data/``) and
                       export ``MRNET_DATA_DIR``.
resolve_medvit_paths() Locate the MedViT repo + pretrained checkpoint and export
                       ``MEDVIT_REPO_DIR`` / ``MEDVIT_CKPT`` (read by src.config).
ensure_dataset_layout()Symlink the underscore CSV names ``src`` expects to the
                       hyphenated names found on the GPU machines.
gpu_results_dir()      The ``codes/for-gpu/results`` folder (metrics/logs/figures).
setup_runtime()        One call that does cap + path (+ optional data / MedViT).

IMPORTANT: import this module and call ``cap_cpu_threads()`` BEFORE importing
torch/numpy if you rely on the thread caps.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Folder layout (this file lives in codes/src/).
SRC_DIR = Path(__file__).resolve().parent
CODES_DIR = SRC_DIR.parent
PROJECT_ROOT = CODES_DIR.parent

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def cap_cpu_threads(n: int = 1) -> None:
    """Cap intra-op CPU threads per process and reduce CUDA fragmentation.

    On a many-core box each DataLoader worker would otherwise spawn one
    BLAS/OpenMP thread per core; with several workers that oversubscription
    stalls data loading. One thread per worker keeps loading parallel ACROSS
    workers without thrashing (the heavy math runs on the GPU). Only effective
    if called BEFORE torch/numpy import, hence ``setdefault``.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for var in _THREAD_VARS:
        os.environ.setdefault(var, str(n))


def add_codes_to_path() -> None:
    """Put ``codes/`` on ``sys.path`` so ``import src...`` resolves."""
    if str(CODES_DIR) not in sys.path:
        sys.path.insert(0, str(CODES_DIR))


def _first_existing(*candidates) -> Path | None:
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


def log(msg: str) -> None:
    """Timestamped, flushed log line (shared format across GPU runners)."""
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def resolve_data_dir() -> Path:
    """Locate the MRNet data folder and export ``MRNET_DATA_DIR``.

    Honours ``MRNET_DATA_DIR`` first, then ``<project>/mrnet`` (the GPU
    machines' layout), then ``<project>/data`` (the original Colab layout).
    """
    data_dir = _first_existing(
        Path(os.environ["MRNET_DATA_DIR"]) if os.environ.get("MRNET_DATA_DIR") else None,
        PROJECT_ROOT / "mrnet",
        PROJECT_ROOT / "data",
    )
    if data_dir is None:
        raise FileNotFoundError(
            "Could not find the MRNet data folder. Set MRNET_DATA_DIR to the "
            "folder that contains train/ valid/ and the label CSVs."
        )
    os.environ["MRNET_DATA_DIR"] = str(data_dir)
    return data_dir


def resolve_medvit_paths() -> dict:
    """Locate the MedViT repo + pretrained checkpoint and export them via env.

    ``src.config`` reads ``MEDVIT_REPO_DIR`` / ``MEDVIT_CKPT`` at import time and
    ``model_factory`` adds the repo to ``sys.path`` from config, so this must run
    BEFORE ``src.config`` is first imported.
    """
    repo = _first_existing(
        Path(os.environ["MEDVIT_REPO_DIR"]) if os.environ.get("MEDVIT_REPO_DIR") else None,
        PROJECT_ROOT / "MedViT",
        CODES_DIR / "MedViT",
    )
    ckpt = _first_existing(
        Path(os.environ["MEDVIT_CKPT"]) if os.environ.get("MEDVIT_CKPT") else None,
        PROJECT_ROOT / "MedViT_small_im1k.pth",
        CODES_DIR / "MedViT_small_im1k.pth",
    )
    if repo is None:
        raise FileNotFoundError(
            "Could not find the MedViT repo (expected MedViT/MedViT.py). Set "
            "MEDVIT_REPO_DIR."
        )
    if ckpt is None:
        raise FileNotFoundError(
            "Could not find MedViT_small_im1k.pth. Set MEDVIT_CKPT."
        )
    os.environ["MEDVIT_REPO_DIR"] = str(repo)
    os.environ["MEDVIT_CKPT"] = str(ckpt)
    return {"medvit_repo": repo, "medvit_ckpt": ckpt}


def ensure_dataset_layout(data_dir: Path,
                          tasks=("abnormal", "acl", "meniscus"),
                          splits=("train", "valid")) -> None:
    """Symlink the underscore CSV names ``src`` expects to the hyphen names.

    ``src/data_pipeline.py`` builds ``"{split}_{task}.csv"`` but the files on the
    GPU machines are ``"{split}-{task}.csv"``. Creates thin symlinks (falling
    back to a copy if symlinks aren't permitted) so shared ``src`` code is never
    touched. Idempotent.
    """
    data_dir = Path(data_dir)
    for split in splits:
        for task in tasks:
            want = data_dir / f"{split}_{task}.csv"
            have = data_dir / f"{split}-{task}.csv"
            if want.exists():
                continue
            if have.exists():
                try:
                    want.symlink_to(have.name)
                except OSError:
                    import shutil
                    shutil.copy(have, want)


def gpu_results_dir() -> Path:
    """The ``codes/for-gpu/results`` folder (metrics CSVs, logs, figures)."""
    return CODES_DIR / "for-gpu" / "results"


def setup_runtime(data: bool = False, medvit: bool = False,
                  threads: int = 1) -> dict:
    """One-shot environment wiring used at the top of the GPU runners.

    Always caps CPU threads and puts ``codes/`` on ``sys.path``. Optionally
    resolves the data folder and/or MedViT paths. Returns a dict with whatever
    was resolved (``data_dir`` / ``medvit_repo`` / ``medvit_ckpt``).
    """
    cap_cpu_threads(threads)
    add_codes_to_path()
    out: dict = {}
    if data:
        out["data_dir"] = resolve_data_dir()
    if medvit:
        out.update(resolve_medvit_paths())
    return out
