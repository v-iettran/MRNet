#!/usr/bin/env python3
"""GPU runner for the original MRNet baseline (AlexNet + max-pool).

Owner: Viet (GPU runner)  |  Baseline architecture: Ilaria (00_baseline_mrnet.ipynb)
Plane: sagittal  |  Task: acl (the project's external-validation task)

Headless, background-friendly port of ``codes/notebooks/00_baseline_mrnet.ipynb``
so the reference baseline is trained on the DGX Spark through the SAME shared
pipeline (``src.data_pipeline`` / ``src.training_utils``) as every other model,
and can be evaluated on the external Rijeka set by ``eval_external.py``.

Architecture (Bien et al. 2018, identical to the notebook):
  pretrained AlexNet per slice -> 4096-d feature -> MAX-pool across slices -> FC->1.
Note this is the parameter-free max-pool baseline; it deliberately does NOT use the
attention pooling of ``MRNetModel`` (that's what the later models add).

The trained checkpoint is saved inference-ready (``build_config`` embedded, with
``backbone="alexnet_baseline"``) so ``eval_external.py`` can rebuild it without
any notebook code. AlexNet has no BatchNorm, so the batch-stat BN fix used for the
CNN/MedViT runs is unnecessary here.

Defaults mirror the notebook's documented baseline: Adam(lr=1e-5, wd=0.1),
ReduceLROnPlateau(patience=4, factor=0.3), 50 epochs, patience 10, 'medium'
augmentation. (The experimental models use 'strong'; the baseline keeps the
notebook's 'medium' so it faithfully reproduces the published reference. Override
with --train-augment strong if you want it matched to the tuned models.)

Outputs (under ``codes/for-gpu/results/``):
  alexnet_baseline_history.csv  -> per-epoch train/val loss + AUC + full metrics
  alexnet_baseline_summary.csv  -> best-epoch row
checkpoint -> codes/model_checkpoints/alexnet_baseline_gpu/best_alexnet_baseline.pth
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Reuse run_cnn's environment wiring (CPU-thread caps + sys.path) and helpers.
import run_cnn as rc

log = rc.log


# --------------------------------------------------------------------------
# Baseline model. The architecture now lives in src.model_factory so the
# interactive notebook (00_baseline_mrnet.ipynb), this trainer, and
# eval_external.py all build the EXACT same model. Imported lazily (inside the
# function) so torch is only pulled in after run_cnn caps the CPU threads.
# --------------------------------------------------------------------------
def build_alexnet_baseline():
    """Build the AlexNet+max-pool baseline (defined in src.model_factory)."""
    from src.model_factory import build_baseline_model
    return build_baseline_model()


def train_baseline(args, device, results_dir: Path):
    import torch
    from src import config
    from src.data_pipeline import build_dataloaders, set_seed
    from src.training_utils import run_training

    log(f"=== Training {args.variant} (AlexNet+maxpool baseline, {args.task}, "
        f"{args.plane}, aug={args.train_augment}) ===")
    set_seed(config.SEED)

    train_loader, val_loader = build_dataloaders(
        root_dir=args.data_dir, task=args.task, plane=args.plane,
        train_augment=args.train_augment, batch_size=1,
        num_workers=args.num_workers, output_size=args.output_size,
        pin_memory=False if args.no_pin_memory else None,
    )

    model = build_alexnet_baseline().to(device)

    # Mirror the notebook's optimiser/scheduler exactly.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.3, threshold=1e-4,
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

    # Inference-ready checkpoint. AlexNet has no BatchNorm, so bn_running_stats
    # is irrelevant; eval rebuilds the architecture purely from `backbone`.
    build_config = {
        "backbone": "alexnet_baseline", "use_cbam": False,
        "dropout": 0.0, "output_size": args.output_size,
        "bn_running_stats": True, "contrastive": False,
        "train_augment": args.train_augment, "task": args.task,
        "plane": args.plane,
    }
    best_ckpt = ckpt_dir / f"best_{args.variant}.pth"
    rc.embed_build_config(best_ckpt, build_config)

    rc.write_csv(rc.history_to_rows(history), results_dir / f"{args.variant}_history.csv")
    summary = rc.best_epoch_summary(history)
    summary.update({"variant": args.variant, "backbone": "alexnet_baseline",
                    "use_cbam": False, "contrastive": False,
                    "train_augment": args.train_augment, "task": args.task,
                    "plane": args.plane, "checkpoint": str(best_ckpt)})
    rc.write_csv(summary, results_dir / f"{args.variant}_summary.csv")

    log(f"{args.variant} done — best val AUC: {summary.get('val_auc', float('nan')):.4f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="GPU runner for the AlexNet+max-pool MRNet baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task", default="acl", choices=["abnormal", "acl", "meniscus"])
    p.add_argument("--plane", default="sagittal",
                   choices=["axial", "coronal", "sagittal"])
    p.add_argument("--train-augment", default="medium",
                   choices=["none", "light", "medium", "strong"],
                   help="Notebook baseline uses 'medium'. Use 'strong' to match "
                        "the tuned models' augmentation.")
    p.add_argument("--output-size", type=int, default=256)
    p.add_argument("--tag", default=None,
                   help="Variant suffix; outputs keyed by 'alexnet_baseline[_<tag>]'.")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--accumulation-steps", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=min(6, (rc.os.cpu_count() or 4)))
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--results-dir", default=str(rc.HERE / "results"))
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.patience = 1
        args.results_dir = str(rc.HERE / "results" / "smoke")
        rc.os.environ.setdefault("FORGPU_SMOKE_LIMIT", "12")

    args.variant = "alexnet_baseline" if not args.tag else f"alexnet_baseline_{args.tag}"

    data_dir = rc.resolve_data_dir()
    rc.ensure_dataset_layout(data_dir)
    args.data_dir = str(data_dir)

    log("=" * 70)
    log(f"GPU runner — {args.variant}")
    log(f"  data_dir  : {data_dir}")
    log(f"  task/plane: {args.task} / {args.plane}")
    log(f"  input size: {args.output_size}  | augment: {args.train_augment}")
    log(f"  lr/wd     : {args.lr} / {args.weight_decay}")
    log(f"  epochs    : {args.epochs} (patience {args.patience}) | workers {args.num_workers}")
    log("=" * 70)

    device = rc.configure_gpu(deterministic=args.deterministic)
    if args.smoke_test:
        rc.maybe_patch_for_smoke()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "started": datetime.now(timezone.utc).astimezone().isoformat(),
        "variant": args.variant, "backbone": "alexnet_baseline",
        "train_augment": args.train_augment, "task": args.task, "plane": args.plane,
        "lr": args.lr, "weight_decay": args.weight_decay, "epochs": args.epochs,
        "patience": args.patience, "accumulation_steps": args.accumulation_steps,
        "num_workers": args.num_workers, "output_size": args.output_size,
        "hostname": rc.os.uname().nodename, "data_dir": args.data_dir,
        "smoke_test": args.smoke_test,
    }
    (results_dir / f"run_metadata_{args.variant}.json").write_text(json.dumps(meta, indent=2))

    t0 = time.time()
    try:
        train_baseline(args, device, results_dir)
    except Exception:
        import traceback
        log(f"!!! {args.variant} FAILED:\n{traceback.format_exc()}")
        raise
    log(f"DONE ({args.variant}) in {(time.time() - t0) / 60:.1f} min. Results in {results_dir}")


if __name__ == "__main__":
    main()
