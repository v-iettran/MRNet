# ACL-Tear Detection — Model & Generalization Showcase

Recruiter/client-facing showcase for the knee-MRI ACL project. Renders the full
story straight from committed results — **no model weights or patient data needed**:

1. **Does it generalize?** Internal (MRNet) vs. zero-shot external (Rijeka) AUC.
2. **Interpretability** — Grad-CAM++ overlays, correct vs. incorrect, per model.
3. **Ablations & method** — augmentation sweep, CBAM/contrastive, Ray tuning.

## Run
```bash
pip install -r app/requirements.txt
streamlit run app/showcase.py
```
