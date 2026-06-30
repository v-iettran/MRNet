#!/usr/bin/env python3
"""
launch_ray_tuning.py
--------------------
Autonomous launcher for the distributed Ray hyperparameter search.

It waits until both GPUs are free of the current Phase-A/manual jobs, then:
  1. (binding constraint) waits for the peer's `medvit_strong_contrastive` to finish
     -- this frees the peer GPU AND finalises the MedViT winner;
  2. stops the now-redundant local manual densenet tuning (Ray re-tunes it fresh);
  3. syncs the peer's contrastive summary + checkpoint back to the head;
  4. auto-selects the best densenet and MedViT variant by validation AUC;
  5. syncs the winning checkpoints to the peer (so trials there can warm-start);
  6. brings up the 2-node Ray cluster, runs tune_ray.py for BOTH winners across
     both GPUs, then tears the cluster down.

Designed to be launched detached:  setsid nohup ... &
"""
import os
import csv
import time
import socket
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
# Trained weights now live in codes/model_checkpoints (was results/checkpoints).
CKPT_ROOT = HERE.parent / "model_checkpoints"
PEER = os.environ.get("PEER_HOST", "169.254.234.202")
PEER_USER = os.environ.get("PEER_USER", "viet")
# Peer shares the same absolute repo layout, so the local for-gpu path doubles
# as the remote one.
PEER_FORGPU = str(HERE)

DENSENET_VARIANTS = ["densenet121_strong", "densenet121_strong_cbam",
                     "densenet121_strong_contrastive"]
MEDVIT_VARIANTS = ["medvit_strong", "medvit_strong_cbam",
                   "medvit_strong_contrastive"]


def log(m):
    print(f"[ray-launcher {time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def _ssh(cmd, **kw):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                           f"{PEER_USER}@{PEER}", cmd],
                          capture_output=True, text=True, **kw)


def _bracketize(p: str) -> str:
    """Wrap the first char in a regex class so ``pgrep -f`` doesn't match the
    wrapping shell command that *contains* the pattern string (classic
    self-match pitfall): '[r]un_cnn...' matches 'run_cnn...' but not the literal
    '[r]un_cnn...' in the ssh command line."""
    return f"[{p[0]}]{p[1:]}" if p else p


def peer_running(pattern: str) -> bool:
    try:
        r = _ssh(f"pgrep -f '{_bracketize(pattern)}' >/dev/null && echo RUN || echo IDLE",
                 timeout=20)
        # Only a clean 'RUN' counts as running; empty stdout (ssh hiccup) or
        # 'IDLE' both read as not-running so we never wedge forever.
        return r.returncode == 0 and r.stdout.strip() == "RUN"
    except Exception as e:
        log(f"  (peer check failed, will retry next poll: {e})")
        return True


def local_running(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern],
                          capture_output=True).returncode == 0


def summary_row(variant: str):
    f = RESULTS / f"{variant}_summary.csv"
    if not f.exists():
        return None
    try:
        with open(f) as fh:
            return next(csv.DictReader(fh))
    except Exception:
        return None


def val_auc(variant: str):
    row = summary_row(variant)
    if not row:
        return None
    try:
        return float(row["val_auc"])
    except Exception:
        return None


def pick_winner(variants):
    scored = [(val_auc(v), v) for v in variants if val_auc(v) is not None]
    if not scored:
        return None
    scored.sort(reverse=True)
    for auc, v in scored:
        log(f"    {v}: val_auc={auc}")
    return scored[0][1]


def main():
    log("=" * 70)
    log("Autonomous Ray-tuning launcher started.")

    # 1. Wait for the peer contrastive job (binding constraint).
    log("waiting for peer medvit_strong_contrastive to finish...")
    while peer_running("run_cnn.py.*strong_contrastive"):
        time.sleep(60)
    log("peer contrastive finished -> peer GPU free, MedViT winner decided.")

    # 2. Stop the redundant local manual densenet tuning (Ray redoes it fresh).
    if local_running("run_tuning.py --variant densenet121_strong_cbam"):
        log("stopping redundant manual densenet tuning (Ray will redo it fresh)...")
        subprocess.run(["pkill", "-f",
                        "run_tuning.py --variant densenet121_strong_cbam"])
        time.sleep(5)

    # 3. Sync the peer's contrastive summary + checkpoint back to the head.
    log("syncing peer contrastive results -> head")
    subprocess.run(
        ["rsync", "-a",
         f"{PEER_USER}@{PEER}:{PEER_FORGPU}/results/medvit_strong_contrastive_summary.csv",
         str(RESULTS) + "/"], check=False)
    subprocess.run(
        ["rsync", "-a",
         f"{PEER_USER}@{PEER}:{PEER_FORGPU}/results/medvit_strong_contrastive_history.csv",
         str(RESULTS) + "/"], check=False)
    local_dir = CKPT_ROOT / "medvit_strong_contrastive_gpu"
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a",
         f"{PEER_USER}@{PEER}:{CKPT_ROOT}/medvit_strong_contrastive_gpu/",
         str(local_dir) + "/"], check=False)

    # 4. Auto-select winners by validation AUC.
    log("selecting winners:")
    dense = pick_winner(DENSENET_VARIANTS)
    medv = pick_winner(MEDVIT_VARIANTS)
    winners = [w for w in (dense, medv) if w]
    log(f"winners -> densenet={dense}, medvit={medv}")
    if not winners:
        log("!! no winners found; aborting.")
        return

    # 5. Sync winning checkpoints to the peer so its trials can warm-start.
    for w in winners:
        row = summary_row(w)
        ckpt = row.get("checkpoint") if row else None
        if not ckpt:
            continue
        ckdir = str(Path(ckpt).parent) + "/"
        _ssh(f"mkdir -p {ckdir}")
        subprocess.run(["rsync", "-a", ckdir,
                        f"{PEER_USER}@{PEER}:{ckdir}"], check=False)
        log(f"  synced {w} checkpoint dir to peer")

    # 6. Ray up -> tune both winners across both GPUs -> Ray down.
    log("bringing up 2-node Ray cluster")
    subprocess.run([str(HERE / "ray_up.sh")], check=True)
    try:
        log(f"launching Ray tuning across both GPUs for: {winners}")
        subprocess.run(
            [str(HERE / ".venv/bin/python"), str(HERE / "tune_ray.py"),
             "--variants", *winners,
             "--n-trials", "10", "--tuning-epochs", "20", "--patience", "5",
             "--num-workers", "6", "--results-dir", str(RESULTS)],
            check=False)
    finally:
        log("tearing down Ray cluster")
        subprocess.run([str(HERE / "ray_down.sh")], check=False)

    log("DONE. Best configs -> results/tuning_ray_best.json, "
        "all trials -> results/tuning_ray_all_trials.csv")
    log("=" * 70)


if __name__ == "__main__":
    main()
