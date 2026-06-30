#!/usr/bin/env python3
"""GPU batch runner for the backbone sweep (ResNet50 / DenseNet121 / MedViT).

Owner: Viet (GPU runner)  |  Backbones: Caolan/Viet  |  Training loop: Ilaria
Plane: sagittal  |  Sweep task: acl

Headless, background-friendly version of:
  * ``codes/notebooks/01_resnet50.ipynb``
  * ``codes/notebooks/02_densenet121.ipynb``
  * ``codes/notebooks/03_medvit.ipynb`` (fine-tuned through the SAME loop here,
    so all three backbones are directly comparable; MedViT runs at 224px).

Re-tuned for the DGX Spark (NVIDIA GB10, ARM64) so the team doesn't have to fight
Colab's 5-hour GPU cap. One backbone per invocation (``--backbone``), so the two
models can be trained **in parallel** as two detached processes.

Training logic is intentionally identical to the notebooks: it calls the shared
``src.training_utils.run_training`` (Ilaria's loop) with the exact same model,
optimizer, scheduler, epochs and class-weighting the notebooks use. The model is
**fine-tuned end-to-end** (``build_model`` default ``freeze_backbone=False``),
matching the notebooks. The only changes are operational:
  * proper GPU, no Colab time cap, runs unattended in the background;
  * TF32 matmuls + cuDNN autotuning + capped CPU threads (avoids the dataloader
    thread-oversubscription stall we hit before);
  * results written to per-model CSVs so teammates can see progress.

Outputs (under ``codes/for-gpu/results/``):
  <backbone>_history.csv   -> per-epoch train/val loss + AUC + full val metrics
  <backbone>_summary.csv   -> best-epoch row (full metric suite)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Environment wiring. MUST run before importing torch/numpy/src.
#
# The actual logic lives in src/env_setup.py so every GPU script wires the
# environment identically. We put codes/ on sys.path, import env_setup, then cap
# CPU threads BEFORE torch/numpy are imported anywhere. The path resolvers are
# re-exported below so the many scripts that do `import run_cnn as rc` keep
# using `rc.resolve_medvit_paths()` / `rc.resolve_data_dir()` / ... unchanged.
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../codes/for-gpu
CODES_DIR = HERE.parent                          # .../codes
PROJECT_ROOT = CODES_DIR.parent                  # .../AI-for-MIA

if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from src import env_setup

env_setup.cap_cpu_threads()      # caps OMP/MKL/... before torch import

# Re-export the shared env helpers under the names callers already use.
_first_existing = env_setup._first_existing
resolve_medvit_paths = env_setup.resolve_medvit_paths
resolve_data_dir = env_setup.resolve_data_dir
ensure_dataset_layout = env_setup.ensure_dataset_layout


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------
log = env_setup.log              # shared timestamped, flushed logger


def configure_gpu(deterministic: bool = False):
    import torch

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cudnn.benchmark = not deterministic
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"device: cuda ({name}, {total_gb:.0f} GB)")
    else:
        log("device: cpu (no CUDA visible — this will be slow)")
    return device


def write_csv(rows, path: Path, columns=None) -> None:
    import pandas as pd

    if isinstance(rows, dict):
        rows = [rows]
    df = pd.DataFrame(rows)
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log(f"wrote {path}  ({len(df)} rows)")


def use_batch_stat_bn(model) -> int:
    """Make every BatchNorm layer use batch statistics in train AND eval.

    Sets ``track_running_stats=False`` (and drops the running buffers) so BN
    normalizes with the current forward's statistics instead of an EMA. This is
    the right choice when each forward is a bag of many instances (here, an
    exam's ~20-60 slices); it sidesteps the unusable running stats produced by
    the ``batch_size=1``-exam training regime. Returns the number of BN layers.
    """
    import torch.nn as nn

    count = 0
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
            count += 1
    return count


def history_to_rows(history: dict) -> list:
    """Flatten ``run_training`` history (lists + per-epoch val_metrics) to rows."""
    rows = []
    n = len(history.get("val_auc", []))
    for i in range(n):
        vm = history["val_metrics"][i] if i < len(history.get("val_metrics", [])) else {}
        row = {
            "epoch": i,
            "train_loss": history["train_loss"][i],
            "train_auc": history["train_auc"][i],
            "val_loss": history["val_loss"][i],
            "val_auc": history["val_auc"][i],
        }
        # full val metric suite (accuracy/precision/recall/f1/sens/spec)
        for k, v in vm.items():
            row[f"val_{k}"] = v
        rows.append(row)
    return rows


def best_epoch_summary(history: dict) -> dict:
    """Return the best-epoch (max val AUC) row as a flat dict for the summary CSV."""
    val_auc = history.get("val_auc", [])
    if not val_auc:
        return {}
    best_i = int(max(range(len(val_auc)), key=lambda i: val_auc[i]))
    vm = history["val_metrics"][best_i] if best_i < len(history.get("val_metrics", [])) else {}
    summary = {
        "best_epoch": best_i,
        "train_loss": history["train_loss"][best_i],
        "train_auc": history["train_auc"][best_i],
        "val_loss": history["val_loss"][best_i],
        "val_auc": val_auc[best_i],
    }
    for k, v in vm.items():
        summary[f"val_{k}"] = v
    return summary


# --------------------------------------------------------------------------
# Training (mirrors 01_resnet50.ipynb / 02_densenet121.ipynb).
# --------------------------------------------------------------------------
def embed_build_config(checkpoint_path: Path, build_config: dict) -> None:
    """Re-save a run_training checkpoint with the build recipe attached.

    run_training only stores weights + metrics. For reliable inference later we
    also embed exactly how to rebuild the model (backbone, cbam, input size, BN
    mode, ...), so loading is self-contained.
    """
    import torch

    if not checkpoint_path.exists():
        return
    ck = torch.load(checkpoint_path, map_location="cpu")
    ck["build_config"] = build_config
    torch.save(ck, checkpoint_path)


def train_backbone(args, device, results_dir: Path):
    import torch
    from src import config
    from src.data_pipeline import build_dataloaders, set_seed
    from src.model_factory import build_model
    from src.training_utils import run_training

    regime = "contrastive+finetune" if args.contrastive else "finetune"
    log(f"=== Training {args.variant} ({regime}, {args.task}, {args.plane}, "
        f"aug={args.train_augment}) ===")
    set_seed(config.SEED)

    train_loader, val_loader = build_dataloaders(
        root_dir=args.data_dir, task=args.task, plane=args.plane,
        train_augment=args.train_augment, batch_size=1,
        num_workers=args.num_workers, output_size=args.output_size,
        pin_memory=False if args.no_pin_memory else None,
    )

    # Fine-tune end-to-end (build_model defaults freeze_backbone=False).
    #
    # Gradient checkpointing (--grad-checkpoint): recomputes backbone activations
    # during backward instead of storing them, trading ~30% compute for a large
    # drop in peak memory. The old worry was that checkpointing double-updates
    # BatchNorm running stats on the recompute -- but in this pipeline BN uses
    # batch statistics (track_running_stats=False, below), so there are NO running
    # stats to corrupt. It's therefore safe, and essential on the GB10's UNIFIED
    # memory pool (CPU+GPU share ~121 GB) to keep parallel jobs from OOM-ing.
    model = build_model(backbone=args.backbone, use_cbam=args.use_cbam,
                        dropout=args.dropout,
                        use_checkpoint=args.grad_checkpoint).to(device)

    # --- BatchNorm fix for the batch_size=1-exam regime ---------------------
    # Each training "batch" is ONE exam's slices (one patient), so BatchNorm's
    # running_mean/running_var EMA never converges to a usable population
    # estimate. In eval() mode (which uses those running stats) DenseNet121's
    # validation AUC collapses to ~0.28 with the logits exploding, even though
    # the learned features score ~0.92 when BN uses the current batch's slice
    # statistics. Since every forward here contains many slices (~20-60), batch
    # statistics are well-defined, so we make BatchNorm ALWAYS use them
    # (track_running_stats=False) for both train and val — consistent and stable.
    if not args.bn_running_stats:
        n = use_batch_stat_bn(model)
        log(f"BatchNorm: using batch statistics (track_running_stats=False) "
            f"on {n} layers")

    # --- Optional Stage 1: Supervised Contrastive pretraining ---------------
    # SupCon pretrains the encoder (backbone + slice pooling) from the ImageNet
    # init, then we fine-tune the WHOLE model end-to-end below (Stage 2). This is
    # the "fine-tune with contrastive learning" regime: contrastive representation
    # learning first, supervised fine-tuning second. use_amp=False sidesteps the
    # degenerate no-positive-pair batch that crashes the GradScaler.
    if args.contrastive:
        from src.contrastive_learning import pretrain_encoder

        log(f"[{args.variant}] Stage 1: SupCon pretraining "
            f"({args.supcon_epochs} epochs, batch {args.supcon_batch})...")
        model, supcon_hist = pretrain_encoder(
            model, train_loader, epochs=args.supcon_epochs,
            supcon_batch=args.supcon_batch, temperature=0.07, lr=args.lr,
            device=device, use_amp=False, verbose=True,
        )
        write_csv(supcon_hist, results_dir / f"{args.variant}_supcon_pretrain.csv")
        log(f"[{args.variant}] Stage 1 done; starting Stage 2 supervised fine-tune")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5,
    )

    ckpt_dir = config.CHECKPOINTS_DIR / f"{args.variant}_gpu"
    history = run_training(
        model, train_loader, val_loader, optimizer, scheduler, device,
        num_epochs=args.epochs,
        accumulation_steps=args.accumulation_steps,
        early_stopping_patience=args.patience,
        checkpoint_dir=str(ckpt_dir),
        task_name=args.variant,
    )

    # Make the best checkpoint self-contained for inference.
    build_config = {
        "backbone": args.backbone, "use_cbam": args.use_cbam,
        "dropout": args.dropout, "output_size": args.output_size,
        "bn_running_stats": args.bn_running_stats,
        "contrastive": args.contrastive, "train_augment": args.train_augment,
        "task": args.task, "plane": args.plane,
    }
    best_ckpt = ckpt_dir / f"best_{args.variant}.pth"
    embed_build_config(best_ckpt, build_config)

    # Stream results to per-variant CSVs.
    write_csv(history_to_rows(history), results_dir / f"{args.variant}_history.csv")
    summary = best_epoch_summary(history)
    summary.update({"variant": args.variant, "backbone": args.backbone,
                    "use_cbam": args.use_cbam, "contrastive": args.contrastive,
                    "train_augment": args.train_augment, "task": args.task,
                    "plane": args.plane, "checkpoint": str(best_ckpt)})
    write_csv(summary, results_dir / f"{args.variant}_summary.csv")

    best = summary.get("val_auc", float("nan"))
    log(f"{args.variant} done — best val AUC: {best:.4f}")


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="GPU runner for the ResNet50 / DenseNet121 sweep (background-friendly).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backbone", required=True,
                   choices=["resnet50", "densenet121", "medvit"],
                   help="Which backbone to train. One per process so they can "
                        "run in parallel. 'medvit' uses the same fine-tuning "
                        "protocol as the CNNs (for an apples-to-apples comparison) "
                        "but at 224px (MedViT's required input size).")
    p.add_argument("--task", default="acl", choices=["abnormal", "acl", "meniscus"])
    p.add_argument("--plane", default="sagittal", choices=["axial", "coronal", "sagittal"])
    p.add_argument("--train-augment", default="light",
                   choices=["none", "light", "medium", "strong"])
    p.add_argument("--output-size", type=int, default=None,
                   help="Slice crop/pad size. Default: 256 for the CNNs (fully "
                        "convolutional) and 224 for MedViT (its required input).")
    p.add_argument("--use-cbam", action="store_true",
                   help="Insert CBAM channel+spatial attention blocks into the "
                        "backbone's conv stages (resnet50/densenet121/medvit).")
    p.add_argument("--contrastive", action="store_true",
                   help="Stage 1: Supervised-Contrastive pretrain the encoder "
                        "from the ImageNet init, then Stage 2 fine-tune the whole "
                        "model end-to-end (supcon -> fine-tune regime).")
    p.add_argument("--supcon-epochs", type=int, default=10,
                   help="SupCon pretraining epochs (only with --contrastive).")
    p.add_argument("--supcon-batch", type=int, default=8,
                   help="Exams accumulated per SupCon step (only with --contrastive).")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Dropout before the classification head.")
    p.add_argument("--tag", default=None,
                   help="Variant suffix for output naming, e.g. 'strong_cbam'. "
                        "Outputs/checkpoints are keyed by '<backbone>[_<tag>]'.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10,
                   help="Early-stopping patience on val AUC.")
    p.add_argument("--accumulation-steps", type=int, default=8)
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Enable gradient checkpointing of the backbone. OFF by "
                        "default (only needed on memory-constrained GPUs, not "
                        "the GB10).")
    p.add_argument("--bn-running-stats", action="store_true",
                   help="Keep standard BatchNorm running-stat behaviour. OFF by "
                        "default: in the batch_size=1-exam regime the running "
                        "stats are unusable (DenseNet121 val AUC collapses to "
                        "~0.28), so by default BN uses batch statistics instead.")
    p.add_argument("--num-workers", type=int, default=min(6, (os.cpu_count() or 4)))
    p.add_argument("--no-pin-memory", action="store_true",
                   help="Disable pinned host memory. Recommended on unified-memory "
                        "systems (GB10) where pinning just duplicates buffers.")
    p.add_argument("--results-dir", default=str(HERE / "results"))
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="Tiny fast run (few exams, 1 epoch) to validate the pipeline.")
    return p.parse_args()


def apply_smoke_test(args):
    args.epochs = 1
    args.patience = 1
    args.supcon_epochs = min(args.supcon_epochs, 1)
    args.results_dir = str(HERE / "results" / "smoke")
    os.environ["FORGPU_SMOKE_LIMIT"] = os.environ.get("FORGPU_SMOKE_LIMIT", "12")


def maybe_patch_for_smoke():
    limit = int(os.environ.get("FORGPU_SMOKE_LIMIT", "0"))
    if limit <= 0:
        return
    from src import data_pipeline as dp

    orig_init = dp.MRNetDataset.__init__

    def _balanced_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        # Keep a class-BALANCED subset so the smoke run has both labels present
        # (capping to the first N exams can yield an all-negative slice, which
        # makes the class-weight computation fail). Real runs are untouched.
        half = max(limit // 2, 1)
        pos_idx = [i for i, y in enumerate(self.labels) if int(y) == 1][:half]
        neg_idx = [i for i, y in enumerate(self.labels) if int(y) == 0][:half]
        keep = sorted(pos_idx + neg_idx)
        self.paths = [self.paths[i] for i in keep]
        self.labels = [self.labels[i] for i in keep]

    dp.MRNetDataset.__init__ = _balanced_init
    log(f"[smoke] capping each dataset to ~{limit} exams (class-balanced)")


def main():
    args = parse_args()
    if args.smoke_test:
        apply_smoke_test(args)

    # Variant label used to key all outputs/checkpoints.
    args.variant = args.backbone if not args.tag else f"{args.backbone}_{args.tag}"

    # Per-backbone default input size: MedViT needs 224, CNNs use 256.
    if args.output_size is None:
        args.output_size = 224 if args.backbone == "medvit" else 256

    data_dir = resolve_data_dir()
    ensure_dataset_layout(data_dir)
    args.data_dir = str(data_dir)

    medvit_paths = {}
    if args.backbone == "medvit":
        medvit_paths = resolve_medvit_paths()

    log("=" * 70)
    log(f"GPU runner — {args.variant}")
    log(f"  data_dir  : {data_dir}")
    if medvit_paths:
        log(f"  medvit    : {medvit_paths['medvit_repo']} | {medvit_paths['medvit_ckpt']}")
    log(f"  task/plane: {args.task} / {args.plane}")
    log(f"  input size: {args.output_size}  | augment: {args.train_augment}")
    log(f"  cbam={args.use_cbam}  contrastive={args.contrastive}  dropout={args.dropout}")
    log(f"  lr/epochs : {args.lr} / {args.epochs} (patience {args.patience})")
    log(f"  workers   : {args.num_workers}")
    log("=" * 70)

    device = configure_gpu(deterministic=args.deterministic)
    if args.smoke_test:
        maybe_patch_for_smoke()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "started": datetime.now(timezone.utc).astimezone().isoformat(),
        "variant": args.variant, "backbone": args.backbone,
        "use_cbam": args.use_cbam, "contrastive": args.contrastive,
        "dropout": args.dropout, "train_augment": args.train_augment,
        "task": args.task, "plane": args.plane,
        "lr": args.lr, "epochs": args.epochs, "patience": args.patience,
        "supcon_epochs": args.supcon_epochs, "supcon_batch": args.supcon_batch,
        "accumulation_steps": args.accumulation_steps,
        "num_workers": args.num_workers, "output_size": args.output_size,
        "hostname": os.uname().nodename, "data_dir": args.data_dir,
        "smoke_test": args.smoke_test,
    }
    (results_dir / f"run_metadata_{args.variant}.json").write_text(json.dumps(meta, indent=2))

    t0 = time.time()
    try:
        train_backbone(args, device, results_dir)
    except Exception:
        import traceback
        log(f"!!! {args.variant} FAILED:\n{traceback.format_exc()}")
        raise
    log(f"DONE ({args.variant}) in {(time.time() - t0) / 60:.1f} min. Results in {results_dir}")


if __name__ == "__main__":
    main()
