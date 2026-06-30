#!/usr/bin/env python3
"""External validation on the Rijeka KneeMRI ACL dataset.

Owner: Viet  |  Metrics/eval loop: src/evaluation.py (Sonia)

Evaluates our best PRE-tuned and POST-tuned DenseNet121 and MedViT models on an
*independent* dataset (Rijeka KneeMRI, Stajduhar et al. 2017) to estimate true
out-of-distribution generalisation. This is a zero-shot transfer: the models were
trained on MRNet (sagittal ACL) and are applied to Rijeka volumes with NO
fine-tuning.

Why a separate driver (and not src/evaluation.py directly): evaluation.py is a
metrics/eval library (its ``evaluate_model`` + ``compute_metrics`` are reused
here verbatim), but it has no data loader for Rijeka's ``.pck`` volumes and no
model-loading/CLI. This script supplies those and calls into evaluation.py so the
exact same metric definitions are used as everywhere else in the project.

Rijeka specifics:
  * volumes are pickled uint16 arrays shaped (slices, H, W) (typically 320x320);
  * ``metadata.csv`` column ``aclDiagnosis`` is 0=intact, 1=partial, 2=complete
    tear. We binarise to ACL-injury (0 vs {1,2}) to match MRNet's ACL task;
  * only the volumes actually present on disk are scored (the public archive is
    a subset of the metadata rows).

Each model is rebuilt exactly as it was trained (backbone, CBAM, input size,
BatchNorm batch-stat fix) from the ``build_config`` embedded in its checkpoint.
The hyperparameter-tuned checkpoints don't carry a build_config (Ray's tuner
didn't embed one), so they reuse their warm-start parent's config -- valid
because tuning only changed optimiser/lr/wd/dropout/accumulation, none of which
alter the module graph or the saved tensors.

Output: results/external_validation_rijeka.csv  (one row per model).
"""
from __future__ import annotations

import argparse
import csv
import pickle
import warnings
from pathlib import Path

import run_cnn as rc  # env wiring (CPU threads, sys.path) BEFORE torch/src import

rc.resolve_medvit_paths()  # MedViT repo/ckpt must be on env before `import src`

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config, evaluation
from src.data_pipeline import MRNetTransform
from src.model_factory import build_model, build_baseline_model

HERE = Path(__file__).resolve().parent
CODES_DIR = HERE.parent
# Both pre-tuned (Stage 1-3) and tuned (Stage 4) weights now live under the
# single ``codes/model_checkpoints`` tree (config.CHECKPOINTS_DIR). Tuned Ray
# winners are in ``{variant}_tuned`` sub-folders alongside the ``{variant}_gpu``
# pre-tuned ones.
PRETUNED_CKPT_ROOT = config.CHECKPOINTS_DIR
TUNED_CKPT_ROOT = config.CHECKPOINTS_DIR
RIJEKA_DIR = config.EXTERNAL_DATA_DIR


# --------------------------------------------------------------------------
# Rijeka dataset
# --------------------------------------------------------------------------
class RijekaDataset(torch.utils.data.Dataset):
    """Yields (image, label) for available Rijeka volumes.

    image: float tensor (slices, 3, H, W) after the SAME MRNetTransform used at
           training/validation time (center-crop/pad -> z-score -> 3 channels).
    label: float tensor (1,)  with 1 = ACL injury (aclDiagnosis in {1,2}).
    """

    def __init__(self, records, transform):
        self.records = records          # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(path, "rb") as fh:
                vol = pickle.load(fh)
        vol = np.asarray(vol).astype(np.float32)   # (slices, H, W)
        image = self.transform(vol)                 # (slices, 3, H, W)
        return image, torch.FloatTensor([label])


def build_records(rijeka_dir: Path):
    """Map metadata rows to on-disk volumes and binarise the label.

    Returns (records, stats) where records is a list of (path, label) and stats
    summarises coverage and class balance.
    """
    import pandas as pd

    meta = pd.read_csv(rijeka_dir / "metadata.csv")
    # filename -> absolute path, across vol01..vol0N subfolders.
    on_disk = {p.name: p for p in rijeka_dir.rglob("*.pck")}

    records, missing = [], 0
    pos = neg = 0
    for _, row in meta.iterrows():
        fname = str(row["volumeFilename"]).strip()
        path = on_disk.get(fname)
        if path is None:
            missing += 1
            continue
        label = 1 if int(row["aclDiagnosis"]) >= 1 else 0
        records.append((str(path), label))
        pos += label
        neg += 1 - label

    stats = {
        "meta_rows": len(meta), "scored": len(records), "missing": missing,
        "pos": pos, "neg": neg,
    }
    return records, stats


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_model(build_config: dict, state_dict: dict, device):
    """Rebuild the model from its build_config + the BN batch-stat fix, load
    weights, and switch to eval. Inference uses no gradient checkpointing."""
    if build_config["backbone"] == "alexnet_baseline":
        # The AlexNet+max-pool baseline deliberately skips MRNetModel's attention
        # pooling, so it has its own factory in src.model_factory. It has no
        # BatchNorm, so the batch-stat fix below never applies.
        model = build_baseline_model()
        model.load_state_dict(state_dict)
        return model.to(device).eval()

    model = build_model(
        backbone=build_config["backbone"],
        use_cbam=build_config.get("use_cbam", False),
        dropout=build_config.get("dropout", 0.0),
        use_checkpoint=False,
    )
    # Training validated with BN in batch-stat mode (track_running_stats=False);
    # replicate it so eval-time normalisation matches and the state_dict (which
    # has no BN running buffers) loads strictly.
    if not build_config.get("bn_running_stats", False):
        rc.use_batch_stat_bn(model)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def resolve_models():
    """Describe the four models to evaluate.

    For the tuned checkpoints (no embedded build_config) we reuse the parent
    variant's config -- the architecture is identical; tuning changed only
    training hyperparameters.
    """
    def tuned_ckpt(folder):
        cands = sorted((TUNED_CKPT_ROOT / folder).glob("*.pth"))
        return cands[0] if cands else None

    dense_parent = PRETUNED_CKPT_ROOT / "densenet121_strong_cbam_gpu" / "best_densenet121_strong_cbam.pth"
    medvit_parent = PRETUNED_CKPT_ROOT / "medvit_strong_gpu" / "best_medvit_strong.pth"
    baseline_ckpt = PRETUNED_CKPT_ROOT / "alexnet_baseline_gpu" / "best_alexnet_baseline.pth"

    return [
        {"name": "alexnet_baseline", "backbone": "alexnet_baseline",
         "ckpt": baseline_ckpt, "config_from": "self"},
        {"name": "densenet121_cbam_pretuned", "backbone": "densenet121",
         "ckpt": dense_parent, "config_from": "self"},
        {"name": "densenet121_cbam_postuned", "backbone": "densenet121",
         "ckpt": tuned_ckpt("densenet121_strong_cbam_tuned"),
         "config_from": dense_parent},
        {"name": "medvit_pretuned", "backbone": "medvit",
         "ckpt": medvit_parent, "config_from": "self"},
        {"name": "medvit_postuned", "backbone": "medvit",
         "ckpt": tuned_ckpt("medvit_strong_tuned"),
         "config_from": medvit_parent},
    ]


def load_manifest(path: Path):
    """Load an explicit model list (JSON) produced by the orchestrator.

    Each entry: {name, backbone, ckpt, config_from}. config_from is either
    "self" (use the checkpoint's own build_config) or a path to a parent
    checkpoint whose build_config should be borrowed (for tuned models).
    """
    import json
    entries = json.loads(Path(path).read_text())
    specs = []
    for e in entries:
        specs.append({
            "name": e["name"], "backbone": e["backbone"],
            "ckpt": Path(e["ckpt"]),
            "config_from": e["config_from"] if e["config_from"] == "self"
                           else Path(e["config_from"]),
        })
    return specs


def load_build_config(spec):
    """Return the build_config for a spec, falling back to the parent's."""
    ck = torch.load(spec["ckpt"], map_location="cpu")
    state = ck["model_state_dict"]
    bc = ck.get("build_config")
    if bc is None:
        if spec["config_from"] == "self":
            raise KeyError(f"{spec['ckpt']} has no build_config and no parent.")
        parent = torch.load(spec["config_from"], map_location="cpu")
        bc = parent["build_config"]
        rc.log(f"  {spec['name']}: no embedded config; using parent "
               f"{Path(spec['config_from']).parent.name}")
    return bc, state, ck.get("val_auc")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rijeka-dir", default=str(RIJEKA_DIR))
    p.add_argument("--manifest", default=None,
                   help="JSON list of models to evaluate (overrides the built-in "
                        "resolve_models set).")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold for the thresholded metrics.")
    p.add_argument("--results-dir", default=str(HERE / "results"))
    args = p.parse_args()

    device = rc.configure_gpu()
    rc.log(f"Device: {device}")

    rijeka_dir = Path(args.rijeka_dir)
    records, stats = build_records(rijeka_dir)
    rc.log(f"Rijeka: scored {stats['scored']}/{stats['meta_rows']} volumes "
           f"({stats['missing']} not on disk) | injury(+)={stats['pos']} "
           f"intact(-)={stats['neg']} | prevalence={stats['pos']/max(stats['scored'],1):.3f}")

    models = load_manifest(args.manifest) if args.manifest else resolve_models()
    rows = []
    for spec in models:
        if spec["ckpt"] is None or not Path(spec["ckpt"]).exists():
            rc.log(f"!! {spec['name']}: checkpoint missing, skipping ({spec['ckpt']})")
            continue

        rc.log(f"=== {spec['name']} ===")
        build_config, state, val_auc = load_build_config(spec)
        out_size = build_config.get("output_size", 256)

        # Each backbone has its OWN input resolution (DenseNet 256, MedViT 224),
        # so the transform/loader is rebuilt per model. No augmentation at eval.
        transform = MRNetTransform(augment="none", normalize="zscore",
                                   output_size=out_size, repeat_channels=True)
        loader = DataLoader(
            RijekaDataset(records, transform), batch_size=1, shuffle=False,
            num_workers=args.num_workers, pin_memory=False,
            persistent_workers=args.num_workers > 0,
        )

        model = load_model(build_config, state, device)
        metrics, y_true, y_prob = evaluation.evaluate_model(
            model, loader, device, threshold=args.threshold)

        rc.log(f"  input={out_size}px cbam={build_config.get('use_cbam')} "
               f"| MRNet-val AUC={val_auc} -> Rijeka {metrics}")
        row = {"model": spec["name"], "backbone": spec["backbone"],
               "input_px": out_size, "use_cbam": build_config.get("use_cbam"),
               "mrnet_val_auc": val_auc, "n_scored": stats["scored"], **metrics}
        rows.append(row)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Persist + print a comparison table.
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "external_validation_rijeka.csv"
    if rows:
        cols = ["model", "backbone", "input_px", "use_cbam", "mrnet_val_auc",
                "n_scored", "AUC", "Accuracy", "Precision", "F1",
                "Sensitivity", "Specificity"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        rc.log(f"Wrote {out_csv}")

        print("\n" + "=" * 78)
        print("EXTERNAL VALIDATION (Rijeka KneeMRI, zero-shot transfer from MRNet)")
        print("=" * 78)
        hdr = f"{'model':<28}{'AUC':>7}{'Acc':>7}{'F1':>7}{'Sens':>7}{'Spec':>7}"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f"{r['model']:<28}{r['AUC']:>7}{r['Accuracy']:>7}{r['F1']:>7}"
                  f"{r['Sensitivity']:>7}{r['Specificity']:>7}")
    else:
        rc.log("No models evaluated.")


if __name__ == "__main__":
    main()
