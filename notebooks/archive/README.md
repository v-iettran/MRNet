# Archived notebooks

These notebooks are kept for provenance but are **not part of the reproducible
pipeline**. Nothing in the active workflow imports or depends on them.

## `05_cross_validation.ipynb`

A scaffold for k-fold cross-validation, a held-out test split, and multi-view
(per-plane) late fusion. **None of this was used in the final study.**

The report instead validates generalisation with a **zero-shot external test**
on the Rijeka KneeMRI dataset (Stage 5): the models trained on MRNet are applied
to an independent cohort from a different institution with no retraining. That
path lives in:

- [`../../for-gpu/eval_external.py`](../../for-gpu/eval_external.py) — the driver
  (Rijeka loader + model loading + CLI), and
- [`../../src/evaluation.py`](../../src/evaluation.py) — the shared metric
  definitions (`compute_metrics`, `evaluate_model`).

The cross-validation helpers (`cross_validate`, `make_test_split`,
`late_fusion`) remain in `src/evaluation.py` as optional, clearly-marked future
extensions, but were not exercised for the reported results.
