# AI for Medical Image Analysis — ACL-tear detection on knee MRI

Code for our study on detecting anterior cruciate ligament (ACL) tears from knee
MRI. We train a transfer-learned CNN/transformer with learned slice-attention
pooling on the Stanford **MRNet** dataset, then test how well it generalises to
an independent hospital's data (**Rijeka KneeMRI**) with no fine-tuning.

Two ways to run everything:

- **Notebooks** ([`notebooks/`](notebooks/)) — interactive, Colab-friendly, one
  notebook per pipeline stage. Best for reading and exploring.
- **GPU runners** ([`for-gpu/`](for-gpu/)) — headless scripts that run the *same*
  `src/` code on a server for the full-scale experiments and figures.

Both import the shared library in [`src/`](src/), so the interactive and headless
paths never diverge.

---

## Five-stage pipeline

The experiment is organised into five stages. Each stage maps to a notebook, a
GPU script, and a figure folder.

| Stage | What it does | Notebook | GPU script | Figures |
|---|---|---|---|---|
| Baseline | Reproduce the Bien et al. (2018) AlexNet + max-pool baseline | [`notebooks/00_baseline_mrnet.ipynb`](notebooks/00_baseline_mrnet.ipynb) | `for-gpu/run_baseline.py` | `figures/01_backbone_screening/` |
| 1 — Backbone screening | Compare ResNet50 / DenseNet121 / MedViT | [`01_resnet50.ipynb`](notebooks/01_resnet50.ipynb), [`02_densenet121.ipynb`](notebooks/02_densenet121.ipynb), [`03_medvit.ipynb`](notebooks/03_medvit.ipynb) | `for-gpu/run_cnn.py`, `run_medvit.py` | `figures/01_backbone_screening/` |
| 2 — Augmentation | Pick the augmentation strength (none/light/medium/strong) | `02_densenet121.ipynb` | `for-gpu/orchestrate.py` | `figures/02_augmentation/` |
| 3 — Ablations | CBAM attention and contrastive (SupCon) pre-training | `02_densenet121.ipynb`, `03_medvit.ipynb` | `for-gpu/orchestrate.py` | `figures/03_ablations/` |
| 4 — Hyperparameter tuning | Random search on the winning variant | [`04_hyperparameter_tuning.ipynb`](notebooks/04_hyperparameter_tuning.ipynb) | `for-gpu/run_tuning.py`, `tune_ray.py` | `figures/04_tuning/` |
| 5 — External validation | Zero-shot test on Rijeka KneeMRI | (see note below) | `for-gpu/eval_external.py` | `figures/05_external_validation/` |
| — Interpretability | Grad-CAM++ and slice-attention maps | [`05_interpretability.ipynb`](notebooks/05_interpretability.ipynb) | — | `figures/06_interpretability/` |
| — Results | Render all report figures | [`06_results_visualization.ipynb`](notebooks/06_results_visualization.ipynb) | — | all `figures/0N_*` |

### Why there is no "cross-validation" notebook

Generalisation is assessed with the **Rijeka zero-shot external test** (Stage 5),
not k-fold cross-validation or multi-view fusion. The old CV scaffold is archived
in [`notebooks/archive/`](notebooks/archive/) with an explanation. `eval_external.py`
loads the best checkpoint and applies it to Rijeka with no retraining; the metric
code is shared with training via [`src/evaluation.py`](src/evaluation.py).

---

## Repository layout

```
codes/
├── README.md              <- this file
├── GRADING.md             <- rubric quick-map for markers
├── notebooks/             <- interactive experiments (00–06) + archive/
├── src/                   <- shared library (single source of truth)
│   ├── config.py          <- all paths + per-stage hyper-parameter presets
│   ├── data_pipeline.py   <- MRNetDataset, transforms, build_dataloaders()
│   ├── model_factory.py   <- MRNetModel, backbones, SliceAttentionPool, baseline
│   ├── attention_modules.py <- CBAM
│   ├── training_utils.py  <- loss, training loop, early stopping, metrics
│   ├── tuning.py          <- hyper-parameter search
│   ├── evaluation.py      <- shared metric definitions
│   ├── interpretability.py<- Grad-CAM++ / attention overlays
│   └── env_setup.py       <- shared runtime wiring (threads, paths)
├── for-gpu/               <- headless runners + shell launchers + setup_env.sh
│   └── results/           <- metrics CSV/JSON, logs, and figures/0N_* (NOT weights)
└── model_checkpoints/     <- trained weights (.pth), keyed by variant
```

Two output trees are kept deliberately separate:

- `model_checkpoints/` holds **only** trained weights. Naming convention:
  `{variant}_gpu/` (Stages 1–3), `{variant}_tuning/` (single-node Stage 4) and
  `{variant}_tuned/` (Ray-distributed Stage 4).
- `for-gpu/results/` holds everything else: metrics CSV/JSON, run logs, and the
  stage-split `figures/`.

Paths are centralised in [`src/config.py`](src/config.py) (`CHECKPOINTS_DIR`,
`GPU_RESULTS_DIR`, `EXTERNAL_DATA_DIR`, `FIGURE_DIRS`), resolved relative to the
repo so the same code works locally and on Colab.

---

## Setup

Expected folder layout (siblings of `codes/`):

```
AI-for-MIA/
├── codes/                 <- this repo
├── mrnet/                 <- MRNet data: train/ valid/ + {split}_{task}.csv
├── external_validation/rijeka/  <- Rijeka KneeMRI (for Stage 5)
├── MedViT/                <- cloned MedViT repo (for the MedViT backbone)
└── MedViT_small_im1k.pth  <- MedViT ImageNet checkpoint
```

If your data lives elsewhere, set `MRNET_DATA_DIR` (and optionally
`EXTERNAL_DATA_DIR`, `MEDVIT_REPO_DIR`, `MEDVIT_CKPT`) before importing
`src.config`.

### Notebooks

Open a notebook and run the bootstrap cell first; it puts `codes/` on the path
and imports `src.config`, which locates the data folder. Then run top to bottom.

### GPU

```bash
cd codes/for-gpu
./setup_env.sh                 # one-time: builds an arch-correct .venv + PyTorch
pip install -r requirements.txt

# single experiment
.venv/bin/python run_cnn.py --backbone densenet121

# the full study, detached (survives logout)
./run_pipeline.sh              # add --smoke-test for a quick end-to-end check
```

---

## Reproducing the report figures

1. Run the GPU pipeline (`run_pipeline.sh`) to train all variants and tune the
   winner; weights land in `model_checkpoints/`, metrics in `for-gpu/results/`.
2. Run `for-gpu/eval_external.py` to score the best checkpoints on Rijeka; it
   writes `for-gpu/results/external_validation_rijeka.csv`.
3. Run [`notebooks/06_results_visualization.ipynb`](notebooks/06_results_visualization.ipynb)
   to render every figure into the stage folders under
   `for-gpu/results/figures/0N_*`, and
   [`05_interpretability.ipynb`](notebooks/05_interpretability.ipynb) for the
   Grad-CAM++ overlays.

---

## Troubleshooting

- **"Data folder not found"** — set `os.environ['MRNET_DATA_DIR']` before
  importing `src.config`, or place `mrnet/` next to `codes/`.
- **MedViT import errors** — make sure the `MedViT/` repo and
  `MedViT_small_im1k.pth` exist at the project root, or set `MEDVIT_REPO_DIR` /
  `MEDVIT_CKPT`.
- **CUDA out of memory when fine-tuning** — `MRNetModel` already gradient-
  checkpoints the backbone over slice-chunks; lower `slice_chunk` in
  `build_model()` if needed.
- **`Exec format error` on the GPU box** — the Colab `.venv` is x86; rebuild with
  `for-gpu/setup_env.sh` on the target machine.
```
