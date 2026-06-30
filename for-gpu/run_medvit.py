#!/usr/bin/env python3
"""GPU batch runner for the MedViT notebook (codes/notebooks/03_medvit.ipynb).

Owner: Viet  |  Plane: sagittal  |  Sweep task: acl

This is a headless, background-friendly version of the ``03_medvit`` notebook,
re-tuned for the DGX Spark (NVIDIA GB10, ARM64) instead of a Colab T4. It runs
the same three experiments and writes the results of *each section* to its own
CSV file under ``codes/for-gpu/results/``:

  section "plain"         -> plain_medvit_history.csv   + plain_medvit_summary.csv
  section "augmentation"  -> augmentation_comparison.csv (+ per-preset history)
  section "supcon"        -> supcon_pretrain_history.csv + supcon_linear_history.csv
                             + supcon_summary.csv

The three sections are independent, so they can be split across the two DGX
Sparks (see ``--sections`` / ``--presets`` and the README). The job is designed
to be launched detached (see ``run.sh``) and keeps running after you log out.

Logic is intentionally identical to the notebook (same model, data, losses,
schedule, seed). The "optimization" is in *how* it runs, not *what* it computes:
  * proper GPU (GB10) with no 5-hour Colab cap, run unattended in the background;
  * TF32 matmuls + cuDNN autotuning + high float32 matmul precision;
  * more dataloader workers (the box has many CPU cores);
  * results streamed to CSV per section so teammates can see partial progress;
  * env wiring for this machine's data / MedViT paths done automatically.
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
# Environment wiring. MUST happen before importing torch / src so that the
# CUDA allocator config and the path overrides in src.config take effect.
# The shared logic lives in src/env_setup.py (identical across all GPU scripts).
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../codes/for-gpu
CODES_DIR = HERE.parent                          # .../codes
PROJECT_ROOT = CODES_DIR.parent                  # .../AI-for-MIA

if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from src import env_setup

env_setup.cap_cpu_threads()      # caps OMP/MKL/... before torch import

# Re-export so the rest of the file keeps its original names.
_first_existing = env_setup._first_existing
ensure_dataset_layout = env_setup.ensure_dataset_layout


def resolve_paths() -> dict:
    """Locate the MRNet data folder, the MedViT repo and the pretrained ckpt.

    Thin wrapper over env_setup so MedViT trains through the SAME wiring as the
    CNN runner. Exports MRNET_DATA_DIR / MEDVIT_REPO_DIR / MEDVIT_CKPT.
    """
    data_dir = env_setup.resolve_data_dir()
    medvit = env_setup.resolve_medvit_paths()
    return {"data_dir": data_dir, "medvit_repo": medvit["medvit_repo"],
            "medvit_ckpt": medvit["medvit_ckpt"]}


# --------------------------------------------------------------------------
# Small helpers (logging + metrics + CSV).
# --------------------------------------------------------------------------
log = env_setup.log              # shared timestamped, flushed logger


def configure_gpu(deterministic: bool = False):
    """Enable Blackwell/Ampere-friendly fast paths and report the device."""
    import torch

    # One intra-op thread: GPU does the heavy lifting, and forked DataLoader
    # workers inherit this, avoiding CPU thread oversubscription (see top of file).
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # TF32 + high matmul precision: big speedup on tensor cores, negligible
        # effect on AUC for this task.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        # Autotune convolution algorithms. Slice counts vary per exam, so there
        # is some re-tuning, but the conv H/W are fixed (224) so it still helps.
        torch.backends.cudnn.benchmark = not deterministic
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"device: cuda ({name}, {total_gb:.0f} GB)")
    else:
        log("device: cpu (no CUDA visible — this will be slow)")
    return device


def evaluate_full_metrics(model, loader, device, threshold=0.5, use_amp=True):
    """Validate and return the full metric suite from Project Pipeline section 4.

    Returns a dict with: loss, auc, accuracy, precision, recall, f1,
    sensitivity, specificity (plus the confusion-matrix counts). Mirrors
    ``src.training_utils.validate`` but adds the thresholded metrics the report
    needs, computed from the same predictions.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    amp_on = bool(use_amp) and str(device).startswith("cuda")
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    running_loss, num_batches = 0.0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for image, label in loader:
            image = image.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_on):
                logits = model(image)
                label_d = label.to(device, non_blocking=True).view_as(logits)
                loss = criterion(logits, label_d)
            running_loss += loss.item()
            num_batches += 1
            all_probs.append(torch.sigmoid(logits.float()).view(-1).cpu().numpy())
            all_labels.append(label.view(-1).cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    def _safe_div(a, b):
        return float(a) / float(b) if b else float("nan")

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)          # == sensitivity
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall) \
        if (precision + recall) else float("nan")
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")

    return {
        "loss": running_loss / max(num_batches, 1),
        "auc": auc,
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sensitivity": recall,
        "specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
    }


def write_csv(rows, path: Path, columns=None) -> None:
    """Write a list-of-dicts (or single dict) to CSV via pandas."""
    import pandas as pd

    if isinstance(rows, dict):
        rows = [rows]
    df = pd.DataFrame(rows)
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log(f"wrote {path.relative_to(PROJECT_ROOT)}  ({len(df)} rows)")


# --------------------------------------------------------------------------
# Sections (mirror the notebook cells).
# --------------------------------------------------------------------------
def make_criterion(args, device):
    import torch
    from src.data_pipeline import get_pos_weight

    pos_weight = get_pos_weight(root_dir=args.data_dir, task=args.task, train=True).to(device)
    log(f"pos_weight ({args.task}): {pos_weight.item():.4f}")
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def section_plain(args, device, results_dir):
    """Plain MedViT transfer learning (notebook cell 5)."""
    import torch
    from src import config
    from src.data_pipeline import build_dataloaders, set_seed
    from src.model_factory import build_model
    from src.training_utils import fit, EarlyStopping

    log("=== Section: plain MedViT transfer learning ===")
    set_seed(config.SEED)

    train_loader, valid_loader = build_dataloaders(
        root_dir=args.data_dir, task=args.task, plane=args.plane,
        train_augment="light", batch_size=1, num_workers=args.num_workers,
        output_size=224,
    )
    criterion = make_criterion(args, device)

    model = build_model(backbone="medvit", pretrained=True, num_classes=1,
                        dropout=0.3, freeze_backbone=not args.finetune)
    lr = 1e-4 if args.finetune else 1e-3  # lower LR when fine-tuning the backbone
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4,
    )

    ckpt = config.CHECKPOINTS_DIR / f"medvit_{args.task}_gpu.pt"
    _, history = fit(
        model, train_loader, valid_loader, optimizer, device,
        num_epochs=args.plain_epochs, criterion=criterion,
        accumulation_steps=args.accumulation_steps,
        early_stopping=EarlyStopping(patience=5, mode="max"),
        checkpoint_path=str(ckpt),
    )

    write_csv(history, results_dir / "plain_medvit_history.csv")
    final = evaluate_full_metrics(model, valid_loader, device)
    final.update({"task": args.task, "plane": args.plane,
                  "finetune": args.finetune, "checkpoint": str(ckpt)})
    write_csv(final, results_dir / "plain_medvit_summary.csv")
    log(f"plain MedViT best val AUC: {final['auc']:.4f}")


def section_augmentation(args, device, results_dir):
    """MedViT under each augmentation preset (notebook cell 7)."""
    import torch
    from src import config
    from src.data_pipeline import build_dataloaders, set_seed
    from src.model_factory import build_model
    from src.training_utils import fit, EarlyStopping

    log("=== Section: augmentation comparison ===")
    criterion = make_criterion(args, device)

    presets = args.presets or ["none", "light", "medium", "strong"]
    rows = []
    for preset in presets:
        log(f"--- augmentation preset: {preset} ---")
        set_seed(config.SEED)  # same init + data order => fair comparison
        tl, vl = build_dataloaders(
            root_dir=args.data_dir, task=args.task, plane=args.plane,
            train_augment=preset, batch_size=1, num_workers=args.num_workers,
            output_size=224,
        )
        m = build_model(backbone="medvit", pretrained=True, num_classes=1,
                        dropout=0.3, freeze_backbone=not args.finetune)
        lr = 1e-4 if args.finetune else 1e-3
        opt = torch.optim.Adam(
            [p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4,
        )
        _, history = fit(
            m, tl, vl, opt, device, num_epochs=args.aug_epochs, criterion=criterion,
            accumulation_steps=args.accumulation_steps,
            early_stopping=EarlyStopping(patience=4, mode="max"),
            checkpoint_path=str(config.CHECKPOINTS_DIR / f"medvit_{args.task}_{preset}_gpu.pt"),
            verbose=False,
        )
        write_csv(history, results_dir / f"augmentation_{preset}_history.csv")
        metrics = evaluate_full_metrics(m, vl, device)
        metrics = {"preset": preset, **metrics}
        rows.append(metrics)
        log(f"preset {preset}: val AUC {metrics['auc']:.4f}, acc {metrics['accuracy']:.4f}")
        # Stream the comparison table after each preset so partial results survive.
        write_csv(rows, results_dir / "augmentation_comparison.csv")

    log("augmentation comparison complete")


def section_supcon(args, device, results_dir):
    """MedViT + Supervised Contrastive Learning (notebook cell 9)."""
    from src import config
    from src.data_pipeline import build_dataloaders, set_seed
    from src.model_factory import build_model
    from src.contrastive_learning import pretrain_encoder, train_linear_classifier

    log("=== Section: MedViT + Supervised Contrastive Learning ===")
    set_seed(config.SEED)

    train_loader, valid_loader = build_dataloaders(
        root_dir=args.data_dir, task=args.task, plane=args.plane,
        train_augment="light", batch_size=1, num_workers=args.num_workers,
        output_size=224,
    )
    criterion = make_criterion(args, device)

    encoder = build_model(backbone="medvit", pretrained=True, num_classes=1, dropout=0.0)

    # use_amp=False for the contrastive pretraining: when a contrastive batch
    # happens to contain NO positive pairs (e.g. the trailing partial batch is
    # two exams with different labels), src.SupConLoss returns a constant zero
    # tensor that is detached from the model parameters. Under AMP, the
    # GradScaler then has nothing to unscale and `scaler.step()` raises
    #   "AssertionError: No inf checks were recorded for this optimizer".
    # Without the scaler, that same batch is simply a harmless no-op step. The
    # GB10 has ample memory, so dropping fp16 here costs only a little speed and
    # makes the run robust to that degenerate batch.
    encoder, supcon_hist = pretrain_encoder(
        encoder, train_loader,
        epochs=args.supcon_epochs, supcon_batch=args.supcon_batch,
        temperature=0.07, lr=1e-4, device=device, use_amp=False,
    )
    write_csv(supcon_hist, results_dir / "supcon_pretrain_history.csv")

    encoder, probe_hist = train_linear_classifier(
        encoder, train_loader, valid_loader,
        epochs=args.probe_epochs, lr=1e-3, criterion=criterion, device=device,
    )
    write_csv(probe_hist, results_dir / "supcon_linear_history.csv")

    final = evaluate_full_metrics(encoder, valid_loader, device)
    final.update({"task": args.task, "plane": args.plane})
    write_csv(final, results_dir / "supcon_summary.csv")
    log(f"MedViT + SupCon val AUC: {final['auc']:.4f}")


SECTIONS = {
    "plain": section_plain,
    "augmentation": section_augmentation,
    "supcon": section_supcon,
}


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="GPU batch runner for the MedViT notebook (background-friendly).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sections", default="all",
                   help="Comma-separated subset of {plain,augmentation,supcon} or 'all'. "
                        "Split these across the two DGX Sparks.")
    p.add_argument("--presets", default=None,
                   help="Comma-separated augmentation presets to run (default: "
                        "none,light,medium,strong). Lets you shard the augmentation "
                        "sweep across nodes.")
    p.add_argument("--task", default="acl", choices=["abnormal", "acl", "meniscus"])
    p.add_argument("--plane", default="sagittal", choices=["axial", "coronal", "sagittal"])
    p.add_argument("--num-workers", type=int, default=min(8, (os.cpu_count() or 4)))
    p.add_argument("--accumulation-steps", type=int, default=8,
                   help="Gradient accumulation (effective batch size in exams).")
    p.add_argument("--finetune", action="store_true",
                   help="Fine-tune the MedViT backbone end-to-end instead of "
                        "feature-extraction. GB10 has the memory for it; uses a "
                        "lower LR. Off by default to match the notebook.")
    # Epoch budgets (defaults match the notebook).
    p.add_argument("--plain-epochs", type=int, default=15)
    p.add_argument("--aug-epochs", type=int, default=10)
    p.add_argument("--supcon-epochs", type=int, default=10)
    p.add_argument("--probe-epochs", type=int, default=15)
    p.add_argument("--supcon-batch", type=int, default=8)
    p.add_argument("--results-dir", default=str(HERE / "results"))
    p.add_argument("--deterministic", action="store_true",
                   help="Disable cuDNN autotuning for more reproducible timing.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Tiny run (few exams, 1-2 epochs) to validate the pipeline "
                        "end-to-end on the GPU. Writes to results/smoke/.")
    return p.parse_args()


def apply_smoke_test(args):
    """Shrink everything so a full end-to-end pass takes a couple of minutes."""
    args.plain_epochs = 1
    args.aug_epochs = 1
    args.supcon_epochs = 1
    args.probe_epochs = 1
    args.supcon_batch = 4
    args.presets = args.presets or ["none", "light"]
    args.results_dir = str(HERE / "results" / "smoke")
    os.environ["FORGPU_SMOKE_LIMIT"] = os.environ.get("FORGPU_SMOKE_LIMIT", "12")


def maybe_patch_for_smoke():
    """If smoke-testing, cap each dataset to a handful of exams via a Dataset patch."""
    limit = int(os.environ.get("FORGPU_SMOKE_LIMIT", "0"))
    if limit <= 0:
        return
    from src import data_pipeline as dp

    orig_len = dp.MRNetDataset.__len__

    def _capped_len(self):
        return min(orig_len(self), limit)

    dp.MRNetDataset.__len__ = _capped_len
    log(f"[smoke] capping each dataset to {limit} exams")


def main():
    args = parse_args()
    if args.smoke_test:
        apply_smoke_test(args)

    paths = resolve_paths()
    ensure_dataset_layout(paths["data_dir"])
    args.data_dir = str(paths["data_dir"])

    log("=" * 70)
    log("MedViT GPU batch runner")
    log(f"  data_dir   : {paths['data_dir']}")
    log(f"  medvit_repo: {paths['medvit_repo']}")
    log(f"  medvit_ckpt: {paths['medvit_ckpt']}")
    log(f"  task/plane : {args.task} / {args.plane}")
    log(f"  workers    : {args.num_workers}")
    log(f"  finetune   : {args.finetune}")
    log("=" * 70)

    device = configure_gpu(deterministic=args.deterministic)
    if args.smoke_test:
        maybe_patch_for_smoke()

    # Normalise selections.
    if args.sections.strip().lower() == "all":
        wanted = list(SECTIONS.keys())
    else:
        wanted = [s.strip() for s in args.sections.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in SECTIONS]
    if unknown:
        raise SystemExit(f"Unknown section(s): {unknown}. Choose from {list(SECTIONS)}.")
    if args.presets and isinstance(args.presets, str):
        args.presets = [p.strip() for p in args.presets.split(",") if p.strip()]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Record run metadata for reproducibility.
    meta = {
        "started": datetime.now(timezone.utc).astimezone().isoformat(),
        "sections": wanted, "task": args.task, "plane": args.plane,
        "finetune": args.finetune, "num_workers": args.num_workers,
        "accumulation_steps": args.accumulation_steps,
        "epochs": {"plain": args.plain_epochs, "aug": args.aug_epochs,
                   "supcon": args.supcon_epochs, "probe": args.probe_epochs},
        "hostname": os.uname().nodename,
        "data_dir": args.data_dir, "smoke_test": args.smoke_test,
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    t0 = time.time()
    for name in wanted:
        try:
            SECTIONS[name](args, device, results_dir)
        except Exception:
            import traceback
            log(f"!!! section '{name}' FAILED:\n{traceback.format_exc()}")
            # Keep going so one failed section doesn't kill the others.
    log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min. Results in {results_dir}")


if __name__ == "__main__":
    main()
