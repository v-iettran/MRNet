#!/usr/bin/env python3
"""
tune_ray.py
-----------
Distributed hyperparameter tuning across BOTH DGX Sparks with Ray.

Each trial = one (variant, hyperparameter) combination, trained on a single GB10
and warm-started from that variant's Phase-A winning checkpoint. Ray schedules
trials across every GPU in the cluster (1 GPU per trial), so the two nodes run
trials in parallel with dynamic load balancing -- a free GPU immediately grabs
the next queued trial regardless of which variant it belongs to.

Why Ray *core* (remote tasks) instead of Ray Tune's Tuner:
  The Sparks share no filesystem, and Tune's experiment/checkpoint syncing wants
  shared (or cloud) storage on a multi-node cluster. With plain @ray.remote tasks
  each trial writes its checkpoint to node-local disk, returns its metric + the
  node it ran on, and the driver rsyncs only the winning checkpoints back. Simple
  and robust over the QSFP link.

Usage (run on the head node after ./ray_up.sh):
  ./.venv/bin/python tune_ray.py --variants densenet121_strong_cbam medvit_strong \
      --n-trials 10 --tuning-epochs 20 --patience 5

Per-trial early stopping still happens inside run_training (patience). Cross-trial
pruning (ASHA) is not used here because it needs per-epoch reporting; can be added
later via the Tune API if wanted.
"""
import os
import sys
import csv
import json
import socket
import argparse
import itertools
import random
from pathlib import Path

import ray

# Absolute path to this for-gpu folder. Ray workers chdir here on every node;
# both nodes share the same absolute repo layout, so resolving from __file__
# yields the identical path everywhere (no hard-coded user path needed).
FORGPU_DIR = str(Path(__file__).resolve().parent)
PEER_HOST = os.environ.get("PEER_HOST", "169.254.234.202")
PEER_USER = os.environ.get("PEER_USER", "viet")


# ---------------------------------------------------------------------------
# Remote trial: runs on whichever node Ray places it (1 GPU reserved per trial)
# ---------------------------------------------------------------------------
@ray.remote(num_gpus=1, num_cpus=6)
def run_trial(variant: str, hp: dict, epochs: int, patience: int,
              num_workers: int, checkpoint_root: str) -> dict:
    # Make the for-gpu package importable and wire env BEFORE importing torch.
    os.chdir(FORGPU_DIR)
    if FORGPU_DIR not in sys.path:
        sys.path.insert(0, FORGPU_DIR)

    import run_cnn as rc  # module import caps CPU threads + adds codes/ to path
    rc.resolve_data_dir()
    try:
        rc.resolve_medvit_paths()
    except FileNotFoundError:
        pass

    import torch
    from src import config as cfg
    from src.data_pipeline import build_dataloaders, set_seed
    from src.model_factory import build_model
    from src import tuning

    node = socket.gethostname()
    gpu_ids = ray.get_gpu_ids()

    # --- load the variant's winning build_config + warm-start weights ---------
    ckpt_path = cfg.CHECKPOINTS_DIR / f"{variant}_gpu" / f"best_{variant}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"[{node}] warm-start checkpoint missing for '{variant}': {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu")
    bc = ck["build_config"]
    warm_state = ck["model_state_dict"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.SEED)

    train_loader, val_loader = build_dataloaders(
        root_dir=os.environ["MRNET_DATA_DIR"],
        task=bc.get("task", "acl"), plane=bc.get("plane", "sagittal"),
        train_augment=bc.get("train_augment", "strong"),
        batch_size=1, num_workers=num_workers,
        output_size=bc.get("output_size", 256), pin_memory=False,
    )

    def model_class():
        m = build_model(backbone=bc["backbone"], use_cbam=bc.get("use_cbam", False),
                        dropout=bc.get("dropout", 0.0), use_checkpoint=False)
        if not bc.get("bn_running_stats", False):
            rc.use_batch_stat_bn(m)
        m.load_state_dict(warm_state)
        return m

    ckpt_dir = Path(checkpoint_root) / variant
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_auc = tuning.evaluate_config(
        config=hp, model_class=model_class,
        train_loader=train_loader, val_loader=val_loader, device=device,
        checkpoint_dir=str(ckpt_dir), task_name=variant,
        epochs=epochs, patience=patience,
    )

    # evaluate_config / run_training name the checkpoint deterministically:
    label = (f"{variant}_opt{hp['optimizer']}_lr{hp['lr']}_wd{hp['weight_decay']}_"
             f"do{hp['dropout']}_acc{hp['accumulation_steps']}")
    trial_ckpt = ckpt_dir / f"best_{label}.pth"

    return {
        "variant": variant, **hp,
        "val_auc": round(float(best_auc), 4),
        "node": node, "gpu_ids": str(gpu_ids),
        "checkpoint": str(trial_ckpt) if trial_ckpt.exists() else "",
    }


# ---------------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------------
def sample_configs(search_space: dict, n_trials: int, seed: int) -> list:
    """Unique configs where possible, else sample with replacement (parity with
    src.tuning.random_search)."""
    keys = list(search_space)
    combos = [dict(zip(keys, c)) for c in itertools.product(*search_space.values())]
    rng = random.Random(seed)
    rng.shuffle(combos)
    if n_trials <= len(combos):
        return combos[:n_trials]
    extra = [{k: rng.choice(search_space[k]) for k in keys}
             for _ in range(n_trials - len(combos))]
    return combos + extra


def pull_checkpoint(remote_node: str, remote_path: str, local_dir: Path) -> str:
    """rsync a winning checkpoint from a worker node back to the head."""
    import subprocess
    if not remote_path:
        return ""
    if remote_node == socket.gethostname():
        return remote_path  # already local
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / Path(remote_path).name
    src = f"{PEER_USER}@{PEER_HOST}:{remote_path}"
    try:
        subprocess.run(["rsync", "-a", src, str(dest)], check=True,
                       timeout=300)
        return str(dest)
    except Exception as exc:  # pragma: no cover
        print(f"  !! failed to pull {src}: {exc}")
        return ""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", required=True,
                   help="Winning variant names to tune, e.g. "
                        "densenet121_strong_cbam medvit_strong")
    p.add_argument("--n-trials", type=int, default=10,
                   help="Trials PER variant.")
    p.add_argument("--tuning-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default=str(Path(FORGPU_DIR) / "results"))
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = results_dir / "tuning_ray_checkpoints"

    # Order matters: importing run_cnn caps CPU threads AND puts codes/ on
    # sys.path, which is what makes `from src import ...` resolvable on the head.
    if FORGPU_DIR not in sys.path:
        sys.path.insert(0, FORGPU_DIR)
    import run_cnn  # noqa: F401  (env wiring on head for any local trials)
    from src import config as cfg
    from src import tuning as _t
    search_space = _t.SEARCH_SPACE

    # Connect to the running cluster (started by ray_up.sh).
    ray.init(address="auto")
    cluster = ray.cluster_resources()
    n_gpus = int(cluster.get("GPU", 0))
    print("=" * 78)
    print(f"Ray cluster: {len(ray.nodes())} node(s), {n_gpus} GPU(s) "
          f"-> up to {n_gpus} trials run concurrently")
    print(f"Variants: {args.variants} | {args.n_trials} trials each "
          f"| {args.tuning_epochs} epochs (patience {args.patience})")
    print("=" * 78)

    # Build + submit every trial; Ray queues them and fills GPUs as they free.
    futures, meta = [], []
    for variant in args.variants:
        for cfg in sample_configs(search_space, args.n_trials, args.seed):
            futures.append(run_trial.remote(
                variant, cfg, args.tuning_epochs, args.patience,
                args.num_workers, str(ckpt_root)))
            meta.append((variant, cfg))

    print(f"Submitted {len(futures)} trials. Waiting for completion...\n")

    rows = []
    remaining = list(futures)
    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1)
        try:
            r = ray.get(done[0])
            rows.append(r)
            print(f"  [{len(rows)}/{len(futures)}] {r['variant']} on {r['node']} "
                  f"-> val_auc={r['val_auc']}  ({r['optimizer']} lr={r['lr']} "
                  f"wd={r['weight_decay']} do={r['dropout']} acc={r['accumulation_steps']})")
        except Exception as exc:
            print(f"  !! a trial failed: {exc}")

    # Write the full results table.
    out_csv = results_dir / "tuning_ray_all_trials.csv"
    if rows:
        cols = ["variant", "optimizer", "lr", "weight_decay", "dropout",
                "accumulation_steps", "val_auc", "node", "gpu_ids", "checkpoint"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in sorted(rows, key=lambda r: r["val_auc"], reverse=True):
                w.writerow(r)
        print(f"\nAll trials -> {out_csv}")

    # Per-variant winner: pull its checkpoint to the head + record build info.
    summary = {}
    for variant in args.variants:
        vrows = [r for r in rows if r["variant"] == variant and r["checkpoint"]]
        if not vrows:
            print(f"  !! no successful trial for {variant}")
            continue
        best = max(vrows, key=lambda r: r["val_auc"])
        # Winning checkpoints are collected into the shared model_checkpoints tree
        # (config.CHECKPOINTS_DIR), NOT under for-gpu/results, so eval_external.py
        # finds pre-tuned and tuned weights in one place.
        local_ckpt = pull_checkpoint(
            best["node"], best["checkpoint"],
            cfg.CHECKPOINTS_DIR / f"{variant}_tuned")
        summary[variant] = {
            "best_val_auc": best["val_auc"],
            "best_config": {k: best[k] for k in
                            ("optimizer", "lr", "weight_decay", "dropout",
                             "accumulation_steps")},
            "ran_on": best["node"],
            "checkpoint": local_ckpt,
        }
        print(f"  WINNER {variant}: val_auc={best['val_auc']} "
              f"cfg={summary[variant]['best_config']} (ckpt -> {local_ckpt})")

    with open(results_dir / "tuning_ray_best.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest-per-variant -> {results_dir / 'tuning_ray_best.json'}")
    ray.shutdown()


if __name__ == "__main__":
    main()
