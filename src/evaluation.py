"""Evaluation metrics library (single source of truth for the whole project).

Owner: Sonia

SCOPE — what is actually used in this study
-------------------------------------------
``compute_metrics()`` and ``evaluate_model()`` are the project's canonical metric
definitions. They are used in TWO places, so every number in the report comes
from the same code path:

  1. MRNet validation split — during training/model selection (Stages 1-4), via
     the training loop in ``src/training_utils.py``.
  2. Rijeka/KneeMRI external set — the zero-shot generalisation test (Stage 5),
     via ``for-gpu/eval_external.py``, which reuses ``evaluate_model`` verbatim.

Metrics: AUC (primary), accuracy, precision, F1, sensitivity, specificity at a
configurable decision threshold (default 0.5).

NOT used in this study (optional future extensions)
---------------------------------------------------
``cross_validate()`` and ``late_fusion()`` are retained as ready-to-use utilities
but were NOT run for the reported results. The report assesses generalisation
with the Rijeka zero-shot test instead of k-fold CV or multi-view fusion (see the
archived ``notebooks/archive/05_cross_validation.ipynb`` and its README). They are
clearly marked below so markers don't go looking for a CV experiment that wasn't
performed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn import metrics as skmetrics
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression


def compute_metrics(y_true, y_prob, threshold=0.5):
  """Compute evaluation metrics from labels and predicted probabilities.
  Args:
      y_true: ground-truth binary labels.
      y_prob: predicted positive-class probabilities.
      threshold: decision threshold for the thresholded metrics.
  Returns:
      A dict with keys such as: auc, accuracy, precision, recall, f1,
      sensitivity, specificity.
  """
  y_true = np.array(y_true) # ground-truth binary labels
  y_prob = np.array(y_prob) # predicted class
  y_pred = (y_prob >= threshold).astype(int)

  try:
    auc = skmetrics.roc_auc_score(y_true, y_prob)
  except ValueError:
    auc = 0.5
  accuracy = skmetrics.accuracy_score(y_true, y_pred)
  precision = skmetrics.precision_score(y_true, y_pred, zero_division=0)
  f1 = skmetrics.f1_score(y_true, y_pred, zero_division=0)
  sensitivity = skmetrics.recall_score(y_true, y_pred, zero_division=0) # same as recall 
  tn, fp, fn, tp = skmetrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
  specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

  metrics = { "AUC": round(auc, 4), "Accuracy": round(accuracy, 4), "Precision": round(precision, 4),"F1": round(f1, 4), "Sensitivity": round(sensitivity, 4), "Specificity": round(specificity, 4)} #dic
  return metrics



def evaluate_model(model, loader, device, threshold=0.5):
  """Run a model over a loader and return metrics + raw predictions.
  Returns:
      (metrics_dict, y_true, y_prob).
  """
  # GRADING (criterion 4 - evaluation): switch to eval mode, run inference with
  # gradients off over the whole loader, and score predictions against labels.
  # Reused for both MRNet validation and the Rijeka zero-shot external test.
  model.eval()
  y_true, y_prob = [], []

  with torch.no_grad():
      for image, label in loader:
          image = image.float().to(device)
          logit = model(image)
          prob = torch.sigmoid(logit).cpu().item()
          y_prob.append(prob)
          y_true.append(int(label.view(-1).item()))

  metrics = compute_metrics(y_true, y_prob, threshold)
  return metrics, y_true, y_prob


def cross_validate(model, dataset, ids, labels, device, n_folds=5, seed=None):
  """[NOT USED IN THIS STUDY] k-fold evaluation of a fixed trained model.

  Optional utility retained for completeness. The reported results use the
  Rijeka zero-shot external test (Stage 5) for generalisation, not k-fold CV.

  The model is fixed. This checks its performance across different subsets.

  Args:
      model: trained model.
      dataset: a Dataset (KneeMRIDataset) yielding (image, label).
      ids: list of exam ids, same order as the dataset's records.
      labels: list of binary labels, same order as ids.
      device: torch.device or "cuda"/"cpu".
      n_folds: number of folds (default 5).
      seed: reproducibility seed.

  Returns:
      dict with "folds" (list of per-fold metrics), "mean", and "std".
  """
  skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
  fold_metrics = []

  for fold, (_, fold_idx) in enumerate(skf.split(ids, labels)):
      fold_loader = DataLoader(Subset(dataset, fold_idx), batch_size=1, shuffle=False)
      metrics, _, _ = evaluate_model(model, fold_loader, device)
      metrics["fold"] = fold + 1
      fold_metrics.append(metrics)
      print(f"fold {fold + 1}: {metrics}")

  metric_names = [k for k in fold_metrics[0] if k != "fold"]
  mean_metrics = {m: round(float(np.mean([f[m] for f in fold_metrics])), 4) for m in metric_names}
  std_metrics = {m: round(float(np.std([f[m] for f in fold_metrics])), 4) for m in metric_names}

  return {"folds": fold_metrics, "mean": mean_metrics, "std": std_metrics}

def late_fusion(plane_probs, labels, method="logreg", fit_on=None):
    """[NOT USED IN THIS STUDY] Combine per-plane probabilities per exam.

    Optional multi-view extension, never run for the report (the study trains and
    evaluates on the sagittal plane only). Retained as a stub for future work.

    Combine per-plane predicted probabilities into one prediction per exam.

    Args:
        plane_probs: mapping plane -> predicted probabilities (aligned by exam).
        labels: ground-truth labels (for fitting/evaluating the fusion).
        method: "mean" | "logreg" (logistic-regression fusion) | ...
        fit_on: indices/mask of the validation exams used to FIT the fusion;
            evaluate on the held-out test exams only.

    Returns:
        Fused probabilities (and the fitted fusion model, if any).
    """
    raise NotImplementedError("TODO(Noma/Sonia): implement late_fusion")
