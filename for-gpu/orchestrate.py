#!/usr/bin/env python3
"""End-to-end pipeline orchestrator for the attention / contrastive / tuning study.

Owner: Viet (GPU runner)

Runs the whole multi-step study unattended on a single DGX Spark (GB10), with up
to ``--max-parallel`` trainings at a time. Designed to be launched detached (see
``run_pipeline.sh``) and keep running after logout.

Steps (all on ACL / sagittal, STRONG augmentation):

  Phase A  (steps 1 & 2, independent -> queued together, 2-way parallel)
    * <bb>_strong              plain fine-tune baseline (fair, strong-aug)
    * <bb>_strong_cbam         + CBAM attention                (step 1)
    * <bb>_strong_contrastive  SupCon-pretrain -> fine-tune     (step 2)
    for bb in {densenet121, medvit}  -> 6 jobs

  Phase B  (step 3, after Phase A finishes)
    * auto-pick the best-val-AUC variant per architecture, then random-search
      hyperparameters on it (2 jobs, 2-way parallel).

Each job is a detached-friendly subprocess writing its own log under
``results/logs/``. A combined comparison table is written at the end of each
phase so partial progress is always visible.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
BACKBONES = ["densenet121", "medvit"]


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[orchestrator {ts}] {msg}", flush=True)


def mem_available_gb() -> float:
    """Live free memory from /proc/meminfo (no external deps).

    On the GB10 this single pool is BOTH CPU RAM and GPU VRAM (unified memory),
    so it's the right number to gate scheduling on.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def est_memory_gb(backbone: str, contrastive: bool) -> int:
    """Conservative peak-memory estimate per job (with grad checkpointing ON).

    Used by the governor to avoid co-scheduling jobs that would exhaust the
    unified memory pool. Deliberately on the high side so the budget is safe.
    """
    if backbone == "medvit":
        return 42 if contrastive else 34
    return 22 if contrastive else 14  # densenet121


def read_val_auc(results_dir: Path, variant: str):
    """Read best val AUC from a variant's summary CSV (None if missing)."""
    path = results_dir / f"{variant}_summary.csv"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            row = next(csv.DictReader(f))
        return float(row["val_auc"])
    except (StopIteration, KeyError, ValueError):
        return None


class Job:
    def __init__(self, name: str, argv: list, log_path: Path, est_gb: int = 15):
        self.name = name
        self.argv = argv
        self.log_path = log_path
        self.est_gb = est_gb
        self.proc = None
        self.rc = None

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        fh = open(self.log_path, "w")
        log(f"START {self.name}  ->  {self.log_path.name}")
        self.proc = subprocess.Popen(self.argv, cwd=str(HERE), env=env,
                                     stdout=fh, stderr=subprocess.STDOUT)
        self._fh = fh

    def poll(self):
        if self.proc is None or self.rc is not None:
            return self.rc
        code = self.proc.poll()
        if code is not None:
            self.rc = code
            try:
                self._fh.close()
            except Exception:
                pass
            status = "OK" if code == 0 else f"FAILED (exit {code})"
            log(f"DONE  {self.name}  -> {status}")
        return self.rc


def run_pool(jobs: list, max_parallel: int, budget_gb: float = 70.0,
             headroom_gb: float = 15.0, warmup_s: float = 120.0,
             poll_s: float = 15.0):
    """Run jobs with a unified-memory governor.

    A new job starts only when ALL hold:
      * fewer than ``max_parallel`` jobs are running;
      * committed estimate (sum of running ``est_gb``) + this job's estimate
        stays within ``budget_gb``  -> prevents co-running two heavy jobs;
      * live MemAvailable minus this job's estimate keeps ``headroom_gb`` free;
      * the previously started job has had ``warmup_s`` to ramp its memory, so
        the live check above reflects real usage before we add more.

    A job is always allowed to run ALONE (even if its estimate exceeds the
    budget) to avoid deadlock. Blocks until every job finishes.
    """
    pending = list(jobs)
    running = []
    last_start = 0.0
    last_hold_log = 0.0
    while pending or running:
        for job in list(running):
            if job.poll() is not None:
                running.remove(job)

        started = False
        if pending and len(running) < max_parallel:
            job = pending[0]
            committed = sum(j.est_gb for j in running)
            avail = mem_available_gb()
            warmed = (time.time() - last_start) >= warmup_s
            fits_budget = (committed + job.est_gb) <= budget_gb
            fits_actual = (avail - job.est_gb) >= headroom_gb
            ok = (not running) or (warmed and fits_budget and fits_actual)
            if ok:
                pending.pop(0)
                job.start()
                log(f"  (committed~{committed + job.est_gb}GB/{budget_gb:.0f}GB budget, "
                    f"MemAvailable {avail:.0f}GB)")
                running.append(job)
                last_start = time.time()
                started = True
            elif time.time() - last_hold_log > 300:
                reason = ("warming up" if not warmed else
                          "budget" if not fits_budget else "low memory")
                log(f"HOLD {job.name} ({reason}; running={[j.name for j in running]}, "
                    f"committed~{committed}GB, MemAvailable {avail:.0f}GB)")
                last_hold_log = time.time()

        if not started:
            time.sleep(poll_s)
    return jobs


def build_phaseA_jobs(args, logs_dir: Path, stamp: str) -> list:
    results_dir = Path(args.results_dir)
    jobs = []
    for bb in BACKBONES:
        # Memory-safety flags applied to every job (unified-memory pool):
        #   --grad-checkpoint : recompute activations in backward (safe now that
        #                       BN uses batch stats, no running-stat double-update)
        #   --num-workers     : fewer prefetching workers -> less host memory
        #   --no-pin-memory   : pinning is pointless on unified memory
        common = [PY, "run_cnn.py", "--backbone", bb,
                  "--train-augment", "strong",
                  "--epochs", str(args.epochs), "--patience", str(args.patience),
                  "--grad-checkpoint", "--num-workers", str(args.num_workers),
                  "--no-pin-memory",
                  "--results-dir", args.results_dir]
        if args.smoke_test:
            common.append("--smoke-test")
        variants = [
            ("strong", []),                                  # baseline
            ("strong_cbam", ["--use-cbam"]),                 # step 1
            ("strong_contrastive", ["--contrastive",         # step 2
                                    "--supcon-epochs", str(args.supcon_epochs),
                                    "--supcon-batch", str(args.supcon_batch)]),
        ]
        for tag, extra in variants:
            variant = f"{bb}_{tag}"
            # Resume-friendly: skip variants that already finished (summary CSV
            # written only on successful completion). Use --force to re-run.
            if not args.force and (results_dir / f"{variant}_summary.csv").exists():
                log(f"SKIP {variant} (already complete; --force to re-run)")
                continue
            argv = common + ["--tag", tag] + extra
            est = est_memory_gb(bb, contrastive="--contrastive" in extra)
            jobs.append(Job(variant, argv, logs_dir / f"pipe_{variant}_{stamp}.log",
                            est_gb=est))
    # Interleave backbones so a fast DenseNet job pairs with a slow MedViT job.
    jobs.sort(key=lambda j: (j.name.split("_", 1)[1], j.name))
    return jobs


def select_winners(results_dir: Path) -> dict:
    """Pick the highest val-AUC variant per architecture from Phase A."""
    winners = {}
    for bb in BACKBONES:
        scored = []
        for tag in ("strong", "strong_cbam", "strong_contrastive"):
            variant = f"{bb}_{tag}"
            auc = read_val_auc(results_dir, variant)
            if auc is not None:
                scored.append((auc, variant))
            log(f"  {variant}: val AUC = {auc}")
        if scored:
            scored.sort(reverse=True)
            winners[bb] = scored[0][1]
            log(f"  -> winner for {bb}: {winners[bb]} (AUC {scored[0][0]:.4f})")
        else:
            log(f"  !! no completed variant for {bb}; skipping tuning")
    return winners


def build_phaseB_jobs(args, winners: dict, logs_dir: Path, stamp: str) -> list:
    results_dir = Path(args.results_dir)
    jobs = []
    for bb, variant in winners.items():
        if not args.force and (results_dir / f"{variant}_tuning_best.json").exists():
            log(f"SKIP tuning {variant} (already complete; --force to re-run)")
            continue
        argv = [PY, "run_tuning.py", "--variant", variant,
                "--n-trials", str(args.n_trials),
                "--tuning-epochs", str(args.tuning_epochs),
                "--patience", str(args.tuning_patience),
                "--num-workers", str(args.num_workers), "--no-pin-memory",
                "--results-dir", args.results_dir]
        if args.smoke_test:
            argv.append("--smoke-test")
        bb = "medvit" if variant.startswith("medvit") else "densenet121"
        est = est_memory_gb(bb, contrastive=False)
        jobs.append(Job(f"{variant}_tuning", argv,
                        logs_dir / f"pipe_{variant}_tuning_{stamp}.log", est_gb=est))
    return jobs


def write_comparison(results_dir: Path):
    """Aggregate every *_summary.csv into one comparison table."""
    rows = []
    for path in sorted(results_dir.glob("*_summary.csv")):
        try:
            with open(path) as f:
                for row in csv.DictReader(f):
                    rows.append(row)
        except Exception:
            continue
    if not rows:
        return
    cols = ["variant", "backbone", "use_cbam", "contrastive", "train_augment",
            "val_auc", "val_accuracy", "val_f1", "val_sensitivity",
            "val_specificity", "best_epoch", "checkpoint"]
    out = results_dir / "comparison_all.csv"
    present = [c for c in cols if any(c in r for r in rows)]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=present, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: float(r.get("val_auc", 0) or 0), reverse=True):
            w.writerow(r)
    log(f"wrote comparison table -> {out}")


def main():
    p = argparse.ArgumentParser(description="Pipeline orchestrator (single-node, parallel).")
    p.add_argument("--max-parallel", type=int, default=2)
    p.add_argument("--budget-gb", type=float, default=70.0,
                   help="Unified-memory budget. Sum of running jobs' estimates "
                        "stays under this, so two heavy jobs never co-run.")
    p.add_argument("--headroom-gb", type=float, default=15.0,
                   help="Keep at least this much live MemAvailable free when "
                        "admitting a new job.")
    p.add_argument("--num-workers", type=int, default=3,
                   help="Dataloader workers per job (lower = less host memory).")
    p.add_argument("--supcon-batch", type=int, default=4,
                   help="Exams accumulated per SupCon step (lower = less peak "
                        "memory in the contrastive stage).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--supcon-epochs", type=int, default=10)
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--tuning-epochs", type=int, default=20)
    p.add_argument("--tuning-patience", type=int, default=5)
    p.add_argument("--results-dir", default=str(HERE / "results"))
    p.add_argument("--skip-phase-a", action="store_true",
                   help="Skip training; go straight to selecting winners + tuning.")
    p.add_argument("--force", action="store_true",
                   help="Re-run variants even if a completed summary already "
                        "exists (default: resume by skipping finished variants).")
    p.add_argument("--smoke-test", action="store_true",
                   help="Tiny end-to-end validation of the whole pipeline.")
    args = p.parse_args()

    # run_cnn/run_tuning --smoke-test force their outputs into results/smoke/, so
    # the orchestrator must read winners + write the comparison from there too.
    if args.smoke_test:
        args.results_dir = str(HERE / "results" / "smoke")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = results_dir / "logs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log("=" * 60)
    log(f"PIPELINE START  (max_parallel={args.max_parallel}, smoke={args.smoke_test})")
    log(f"  memory governor: budget={args.budget_gb:.0f}GB headroom={args.headroom_gb:.0f}GB "
        f"(unified pool MemTotal {mem_available_gb():.0f}GB avail now)")
    log(f"  phase A: {len(BACKBONES) * 3} trainings (baseline / cbam / contrastive)")
    log(f"  epochs={args.epochs} patience={args.patience} supcon_epochs={args.supcon_epochs} "
        f"supcon_batch={args.supcon_batch} workers={args.num_workers} grad_checkpoint=ON")
    log("=" * 60)

    t0 = time.time()
    if not args.skip_phase_a:
        log("### PHASE A: steps 1 & 2 (strong-aug baselines + CBAM + contrastive)")
        jobsA = build_phaseA_jobs(args, logs_dir, stamp)
        run_pool(jobsA, args.max_parallel, budget_gb=args.budget_gb,
                 headroom_gb=args.headroom_gb)
        write_comparison(results_dir)
        log(f"PHASE A complete in {(time.time() - t0) / 60:.1f} min")
    else:
        log("### PHASE A skipped (--skip-phase-a)")

    log("### Selecting best variant per architecture")
    winners = select_winners(results_dir)

    if winners:
        log(f"### PHASE B: step 3 hyperparameter tuning on {list(winners.values())}")
        jobsB = build_phaseB_jobs(args, winners, logs_dir, stamp)
        run_pool(jobsB, args.max_parallel, budget_gb=args.budget_gb,
                 headroom_gb=args.headroom_gb)
    else:
        log("### PHASE B skipped: no winners to tune")

    write_comparison(results_dir)
    log(f"PIPELINE COMPLETE in {(time.time() - t0) / 60:.1f} min. "
        f"See {results_dir / 'comparison_all.csv'}")


if __name__ == "__main__":
    main()
