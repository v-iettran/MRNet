"""Project-wide configuration: paths, constants, experiment settings.

Owner: shared.

This file is the SINGLE SOURCE OF TRUTH for where things live and for the default
hyper-parameters of every experiment stage. Notebooks and the headless GPU runners
both import these constants so the two paths stay in lock-step.

Paths are resolved RELATIVE TO THIS FILE, so they work on Colab and locally as
long as the folder layout is preserved:

    AI-for-MIA/                 <- PROJECT_ROOT
    |-- codes/                  <- CODES_DIR
    |   |-- notebooks/          <- the .ipynb experiments run here
    |   |-- src/                <- this file lives here (config.py)
    |   |-- model_checkpoints/  <- trained weights (.pth) get written here
    |   `-- for-gpu/results/    <- metrics CSVs, logs and report figures
    |-- mrnet/                  <- MRNet data (train/ valid/ + label CSVs)
    |-- external_validation/    <- Rijeka KneeMRI external test set
    `-- MedViT/                 <- cloned MedViT backbone repo

If your data lives somewhere else, set the environment variable
``MRNET_DATA_DIR`` (e.g. in a notebook: ``os.environ['MRNET_DATA_DIR'] = '...'``)
before importing this module, and it will take precedence over the default.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Core directories (resolved from this file's location) ---
SRC_DIR = Path(__file__).resolve().parent
CODES_DIR = SRC_DIR.parent
PROJECT_ROOT = CODES_DIR.parent

# Data folder. On the original Colab layout MRNet lived in a sibling ``data/``
# folder; on the GPU machines it is ``mrnet/``. We honour MRNET_DATA_DIR first,
# then fall back to whichever of those exists, then to ``data`` for backwards
# compatibility.
def _default_data_dir() -> Path:
    env = os.environ.get("MRNET_DATA_DIR")
    if env:
        return Path(env)
    for cand in (PROJECT_ROOT / "mrnet", PROJECT_ROOT / "data"):
        if cand.exists():
            return cand
    return PROJECT_ROOT / "data"


DATA_DIR = _default_data_dir()

# --- MRNet on-disk layout (adjust the names if your extraction differs) ---
TRAIN_VOLUMES_DIR = DATA_DIR / "train"   # contains axial/ coronal/ sagittal/
VALID_VOLUMES_DIR = DATA_DIR / "valid"
# Label CSVs live directly in DATA_DIR, e.g. train_acl.csv, valid_acl.csv, ...

# --- External validation (Stage 5: zero-shot transfer test) ---
# Rijeka KneeMRI lives outside ``codes`` so it is shared across machines.
EXTERNAL_DATA_DIR = Path(
    os.environ.get("EXTERNAL_DATA_DIR", PROJECT_ROOT / "external_validation" / "rijeka")
)

# --- Output directories (created by the training/eval code when needed) ---
#
# Two clearly separated trees (this split is the whole point of the cleanup):
#   * model_checkpoints/  -> trained weights only (.pth), keyed by variant
#   * for-gpu/results/     -> metrics CSVs/JSON, logs, and report figures
#
# CHECKPOINTS_DIR is kept as an alias of MODEL_CHECKPOINTS_DIR so existing code
# (and embedded checkpoint paths) keep working after the rename from the old
# ``results/checkpoints`` location.
MODEL_CHECKPOINTS_DIR = CODES_DIR / "model_checkpoints"
CHECKPOINTS_DIR = MODEL_CHECKPOINTS_DIR
GPU_RESULTS_DIR = CODES_DIR / "for-gpu" / "results"

# Checkpoint sub-folder naming convention (documented for reproducibility):
#   {variant}_gpu/      best_{variant}.pth          -> Stage 1-3 trained weights
#   {variant}_tuning/   best_{variant}_tuned.pth     -> Stage 4 single-node tuning
#   {variant}_tuned/    best_{variant}_*.pth         -> Stage 4 Ray-distributed tuning

# Report figures, split by pipeline stage so each maps to a section/figure of the
# report. The visualization notebook and make_workflow_diagram.py write here.
_FIG_ROOT = GPU_RESULTS_DIR / "figures"
FIGURE_DIRS = {
    "pipeline": _FIG_ROOT / "00_pipeline",                  # Report Fig. 1
    "backbone": _FIG_ROOT / "01_backbone_screening",        # Stage 1, Report Fig. 2
    "augmentation": _FIG_ROOT / "02_augmentation",          # Stage 2, Report Fig. 3
    "ablations": _FIG_ROOT / "03_ablations",                # Stage 3, Report Fig. 4
    "tuning": _FIG_ROOT / "04_tuning",                      # Stage 4
    "external": _FIG_ROOT / "05_external_validation",       # Stage 5, Report Fig. 5
    "interpretability": _FIG_ROOT / "06_interpretability",  # Grad-CAM++/attention
}

# --- Dataset constants ---
PLANES = ("axial", "coronal", "sagittal")
TASKS = ("abnormal", "acl", "meniscus")

# --- Default experiment settings (see the report's "Experimental design") ---
DEFAULT_PLANE = "sagittal"   # main sweep runs on the sagittal plane
DEFAULT_TASK = "acl"         # ACL-tear detection is the headline task
SEED = 42                    # shared seed for reproducibility across the team

# --- Per-stage hyper-parameter presets -------------------------------------
# These mirror the values reported in the paper and are imported by both the
# notebooks and the GPU runners so the two never silently diverge. Each runner
# still exposes CLI flags to override them for ad-hoc experiments.
#
# BASELINE_DEFAULTS      -> 00_baseline_mrnet.ipynb  / run_baseline.py
# PLAIN_CNN_DEFAULTS     -> 01_resnet50, 02_densenet121 / run_cnn.py
# MEDVIT_NOTEBOOK_DEFAULTS-> 03_medvit.ipynb          / run_medvit.py (feature-extract)
# ORCHESTRATOR_DEFAULTS  -> orchestrate*.py (Stage 2-4 use STRONG augmentation)
BASELINE_DEFAULTS = {
    "optimizer": "Adam",
    "lr": 1e-5,
    "weight_decay": 0.1,
    "train_augment": "medium",
    "output_size": 256,
    "epochs": 50,
    "patience": 10,
    "accumulation_steps": 8,
}

PLAIN_CNN_DEFAULTS = {
    "optimizer": "Adam",
    "lr": 1e-4,
    "weight_decay": 0.0,
    "train_augment": "light",
    "output_size": 256,        # MedViT overrides to 224 (its required input)
    "epochs": 50,
    "patience": 10,
    "accumulation_steps": 8,
}

MEDVIT_NOTEBOOK_DEFAULTS = {
    "optimizer": "Adam",
    "lr": 1e-3,                # 1e-4 when fine-tuning the backbone end-to-end
    "weight_decay": 1e-4,
    "train_augment": "light",
    "output_size": 224,
    "freeze_backbone": True,   # notebook default: feature-extraction
    "dropout": 0.3,
}

# Stages 2-4 adopt STRONG augmentation for a consistent cross-model comparison
# (see report Stage 2). The orchestrators apply this to DenseNet121 and MedViT.
ORCHESTRATOR_DEFAULTS = {
    "train_augment": "strong",
    "epochs": 50,
    "patience": 10,
}

# --- MedViT ---
# Folder containing the cloned official repo's MedViT.py (at the project root).
MEDVIT_REPO_DIR = Path(os.environ.get("MEDVIT_REPO_DIR", PROJECT_ROOT / "MedViT"))
# Pretrained ImageNet checkpoint (MedViT_small_im1k.pth), also at the project root.
MEDVIT_CKPT = Path(os.environ.get("MEDVIT_CKPT", PROJECT_ROOT / "MedViT_small_im1k.pth"))


def label_csv(split: str, task: str) -> Path:
    """Return the path to a label CSV, e.g. ``label_csv("train", "acl")``.

    Args:
        split: "train" or "valid".
        task: one of ``TASKS`` ("abnormal" | "acl" | "meniscus").

    Returns:
        Path to ``<split>_<task>.csv`` inside ``DATA_DIR``.
    """
    return DATA_DIR / f"{split}_{task}.csv"
