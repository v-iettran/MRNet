# MedViT GPU runner (DGX Spark)

Headless, background-friendly version of `codes/notebooks/03_medvit.ipynb`,
re-tuned for the **DGX Spark (NVIDIA GB10, ARM64)** instead of a Colab T4.

It runs the **same three experiments** as the notebook and writes the results of
**each section to its own CSV** under `results/`:

| Section        | Notebook cell | Output CSV(s) |
| -------------- | ------------- | ------------- |
| `plain`        | Plain MedViT transfer learning | `plain_medvit_history.csv`, `plain_medvit_summary.csv` |
| `augmentation` | Augmentation preset comparison | `augmentation_comparison.csv`, `augmentation_<preset>_history.csv` |
| `supcon`       | MedViT + Supervised Contrastive Learning | `supcon_pretrain_history.csv`, `supcon_linear_history.csv`, `supcon_summary.csv` |

`*_history.csv` = per-epoch training/validation curves.
`*_summary.csv` = final held-out metrics (AUC, accuracy, precision, recall, F1,
sensitivity, specificity, confusion-matrix counts).

## Why a script instead of the notebook?

Colab caps GPU sessions at ~5 h, so the notebook can't finish unattended. This
runner:

- runs **detached in the background** (survives SSH disconnect / logout);
- uses the GB10 fully: TF32 matmuls, cuDNN autotuning, high float32 matmul
  precision, more dataloader workers, mixed precision (already in `src`);
- streams **per-section CSVs** so teammates can see partial results as they land;
- auto-wires this machine's paths (data in `../../mrnet`, MedViT repo + checkpoint
  at the project root, hyphenated label CSVs) **without editing shared `src/`**.

The *logic* is identical to the notebook (same model, data, losses, seed,
schedule) — only the execution is optimized.

## Quick start

```bash
cd codes/for-gpu

# 1. One-time environment setup (the repo's .venv is x86/Colab and won't run on ARM).
./setup_env.sh

# 2. Quick end-to-end sanity check (~couple of minutes, tiny data slice).
./run.sh --smoke-test

# 3. Full run, detached. Keeps going after you log out.
./run.sh
```

Follow progress:

```bash
tail -f results/logs/run_*.log
```

## Splitting across the two DGX Sparks

The three sections are independent, so put roughly half the work on each node.

**Node A:**
```bash
./run.sh --sections plain,supcon
```

**Node B:**
```bash
./run.sh --sections augmentation
```

You can shard the augmentation sweep too (4 presets), e.g.:

```bash
# Node A
./run.sh --sections augmentation --presets none,light
# Node B
./run.sh --sections augmentation --presets medium,strong
```

Point both nodes at a shared results folder (`--results-dir /shared/path`) if you
want all CSVs collected in one place; otherwise each node writes locally and you
merge afterwards. Per-section CSV names don't collide across sections.

## Useful flags

```
--sections plain,augmentation,supcon   # which experiments to run (default: all)
--presets none,light,medium,strong     # augmentation presets to sweep
--task acl|abnormal|meniscus           # default: acl (the sweep task)
--plane sagittal|coronal|axial         # default: sagittal
--finetune                             # fine-tune the MedViT backbone end-to-end
                                       #   (GB10 has the memory; off by default to
                                       #    match the notebook's feature-extraction)
--num-workers N                        # dataloader workers (default: up to 8)
--accumulation-steps N                 # effective batch size in exams (default: 8)
--plain-epochs / --aug-epochs / --supcon-epochs / --probe-epochs
--results-dir PATH                     # where CSVs/logs go (default: ./results)
--smoke-test                           # tiny fast run to validate the pipeline
```

## Files

- `run_medvit.py` — the runner (all three sections + metrics + CSV logging).
- `run.sh` — launches it detached (setsid + nohup) with logging + PID file.
- `setup_env.sh` — builds the ARM64/CUDA venv and verifies the GPU.
- `requirements.txt` — Python deps (torch installed separately by `setup_env.sh`).
