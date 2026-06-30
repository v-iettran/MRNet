#!/usr/bin/env python3
"""Step 3: hyperparameter tuning for the best variant of an architecture.

Owner: Viet (GPU runner)  |  Search logic: Ilaria (src/tuning.py)

Random-search over optimiser / lr / weight-decay / dropout / accumulation for a
single *winning* variant (e.g. ``densenet121_strong_cbam`` or
``medvit_strong_contrastive``). Each trial **warm-starts from that variant's best
checkpoint** and continues training under the trial's hyperparameters, so we are
literally tuning our best model rather than retraining from ImageNet (which, for
the contrastive variant, would also mean repeating SupCon every trial).

The model is rebuilt exactly as the winning variant was (backbone, CBAM, input
size, BatchNorm batch-stat fix) by reading the ``build_config`` embedded in its
checkpoint. The best trial's checkpoint is saved inference-ready.

Background-friendly; one architecture per invocation so the two can run in
parallel. Outputs under ``codes/for-gpu/results/``:
  <variant>_tuning_results.csv  -> per-trial hyperparameters + best val AUC
  <variant>_tuning_best.json    -> winning hyperparameters + checkpoint path
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Reuse all the environment wiring + helpers from run_cnn (its module-level code
# caps CPU threads and puts codes/ on sys.path before torch is imported).
import run_cnn as rc

log = rc.log


def load_build_config(variant: str):
    """Read the embedded build_config + weights path for a finished variant."""
    import torch
    from src import config

    ckpt_path = config.CHECKPOINTS_DIR / f"{variant}_gpu" / f"best_{variant}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint for variant '{variant}' not found at {ckpt_path}. "
            f"Run step 1/2 for this variant first."
        )
    ck = torch.load(ckpt_path, map_location="cpu")
    bc = ck.get("build_config")
    if bc is None:
        raise KeyError(
            f"{ckpt_path} has no embedded build_config (was it trained with the "
            f"updated run_cnn.py?)."
        )
    return ck, bc, ckpt_path


def make_model_class(build_config: dict, warm_start_state, device,
                     use_checkpoint: bool = False):
    """Return a zero-arg callable producing a fresh, ready-to-train model.

    Rebuilds the winning architecture, applies the BatchNorm batch-stat fix, and
    warm-starts from the winning checkpoint's weights so tuning refines the best
    model. tuning.apply_dropout adjusts dropout per trial afterwards.
    """
    from src.model_factory import build_model

    def _factory():
        model = build_model(
            backbone=build_config["backbone"],
            use_cbam=build_config.get("use_cbam", False),
            dropout=build_config.get("dropout", 0.0),
            # Gradient checkpointing trades ~25% compute for much lower memory.
            # Off by default: when a job has a GB10 to itself there's ample memory,
            # so we skip the recompute and train faster. Enable with
            # --grad-checkpoint if sharing a box. Safe either way because BN uses
            # batch stats (no running-stat double-update on the backward recompute).
            use_checkpoint=use_checkpoint,
        )
        if not build_config.get("bn_running_stats", False):
            rc.use_batch_stat_bn(model)
        if warm_start_state is not None:
            model.load_state_dict(warm_start_state)
        return model.to(device)

    return _factory


def main():
    p = argparse.ArgumentParser(
        description="Random-search hyperparameter tuning for one winning variant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant", required=True,
                   help="Winning variant name, e.g. 'densenet121_strong_cbam'.")
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--tuning-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-warm-start", action="store_true",
                   help="Train each trial from ImageNet init instead of warm-"
                        "starting from the winning checkpoint.")
    p.add_argument("--num-workers", type=int, default=min(6, (rc.os.cpu_count() or 4)))
    p.add_argument("--no-pin-memory", action="store_true",
                   help="Disable pinned host memory (recommended on unified-memory "
                        "GB10).")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Gradient-checkpoint the backbone (saves memory, ~25%% "
                        "slower). Off by default since a dedicated GB10 has ample "
                        "memory; enable only when sharing a box.")
    p.add_argument("--results-dir", default=str(rc.HERE / "results"))
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    if args.smoke_test:
        args.n_trials = 2
        args.tuning_epochs = 1
        args.patience = 1
        args.results_dir = str(rc.HERE / "results" / "smoke")
        rc.os.environ["FORGPU_SMOKE_LIMIT"] = rc.os.environ.get("FORGPU_SMOKE_LIMIT", "12")

    data_dir = rc.resolve_data_dir()
    rc.ensure_dataset_layout(data_dir)
    # Export MedViT repo/ckpt paths BEFORE importing src.config (it reads them at
    # import time and model_factory adds the repo to sys.path from config).
    try:
        rc.resolve_medvit_paths()
    except FileNotFoundError:
        pass

    import torch  # noqa: F401  (ensures torch import after env wiring)
    from src import config
    from src.data_pipeline import build_dataloaders, set_seed
    from src import tuning

    ck, build_config, ckpt_path = load_build_config(args.variant)

    log("=" * 70)
    log(f"Hyperparameter tuning — variant '{args.variant}'")
    log(f"  build_config: {build_config}")
    log(f"  warm-start  : {not args.no_warm_start}  (from {ckpt_path})")
    log(f"  trials/epochs/patience: {args.n_trials} / {args.tuning_epochs} / {args.patience}")
    log("=" * 70)

    device = rc.configure_gpu()
    if args.smoke_test:
        rc.maybe_patch_for_smoke()

    set_seed(config.SEED)
    train_loader, val_loader = build_dataloaders(
        root_dir=str(data_dir),
        task=build_config.get("task", "acl"),
        plane=build_config.get("plane", "sagittal"),
        train_augment=build_config.get("train_augment", "strong"),
        batch_size=1, num_workers=args.num_workers,
        output_size=build_config.get("output_size", 256),
        pin_memory=False if args.no_pin_memory else None,
    )

    warm_state = None if args.no_warm_start else ck["model_state_dict"]
    model_class = make_model_class(build_config, warm_state, device,
                                   use_checkpoint=args.grad_checkpoint)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = config.CHECKPOINTS_DIR / f"{args.variant}_tuning"
    results_csv = str(results_dir / f"{args.variant}_tuning_results.csv")

    t0 = time.time()
    best_config = tuning.random_search(
        model_class=model_class,
        train_loader=train_loader, val_loader=val_loader, device=device,
        n_trials=args.n_trials, seed=args.seed,
        checkpoint_dir=str(ckpt_dir), results_csv=results_csv,
        task_name=args.variant,
        epochs=args.tuning_epochs, patience=args.patience,
    )

    # Locate the winning trial's checkpoint (tuning names it by the config label)
    # and re-save it inference-ready with the full build + hyperparameter recipe.
    best_ckpt_path = None
    if best_config is not None:
        label = (f"{args.variant}_opt{best_config['optimizer']}_lr{best_config['lr']}_"
                 f"wd{best_config['weight_decay']}_do{best_config['dropout']}_"
                 f"acc{best_config['accumulation_steps']}")
        cand = ckpt_dir / f"best_{label}.pth"
        if cand.exists():
            final = ckpt_dir / f"best_{args.variant}_tuned.pth"
            merged = {**build_config, **{f"hp_{k}": v for k, v in best_config.items()}}
            rc.embed_build_config(cand, merged)
            import shutil
            shutil.copy(cand, final)
            best_ckpt_path = str(final)

    (results_dir / f"{args.variant}_tuning_best.json").write_text(json.dumps({
        "variant": args.variant, "best_config": best_config,
        "best_checkpoint": best_ckpt_path, "build_config": build_config,
        "n_trials": args.n_trials, "tuning_epochs": args.tuning_epochs,
    }, indent=2))

    log(f"DONE tuning ({args.variant}) in {(time.time() - t0) / 60:.1f} min. "
        f"Best config: {best_config}")
    log(f"Best tuned checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
