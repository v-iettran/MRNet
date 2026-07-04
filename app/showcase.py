"""
ACL-Tear Detection — Model & Generalization Showcase
====================================================

A recruiter/client-facing showcase for the MRNet knee-MRI project. It tells the
three-act story straight from committed results (no model weights or patient data
required):

  1. Which architecture wins on internal validation
  2. Does that winner still work on a *different hospital's* scanner (Rijeka)
  3. Where does the model actually look? (Grad-CAM++ interpretability)

Run:  streamlit run app/showcase.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "for-gpu" / "results"
FIG = RESULTS / "figures"

st.set_page_config(page_title="ACL-Tear Detection · Model Showcase", page_icon="🦵", layout="wide")

# ------------------------------------------------------------------ styling --
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1200px;}
      h1,h2,h3 {letter-spacing:-0.01em;}
      .hero h1 {margin:0 0 .2rem 0; font-size:1.95rem;}
      .sub {color:#64748b; font-size:1.02rem;}
      .pill {display:inline-block; padding:.16rem .6rem; border-radius:999px;
             font-size:.78rem; font-weight:600;}
      .pill-win  {background:#dcfce7; color:#166534;}
      .pill-warn {background:#fee2e2; color:#991b1b;}
      .pill-tech {background:#e0e7ff; color:#3730a3; margin-right:.3rem;}
      .lead {font-size:1.05rem; color:#334155; line-height:1.55;}
      [data-testid="stMetricValue"] {font-size:1.75rem;}
      .cap {color:#64748b; font-size:.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------- data loaders -
@st.cache_data
def load_external() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "external_validation_rijeka.csv")
    pretty = {
        "alexnet_baseline": "AlexNet (baseline)",
        "densenet121_cbam_pretuned": "DenseNet121 + CBAM (pre-tune)",
        "densenet121_cbam_postuned": "DenseNet121 + CBAM (tuned)",
        "medvit_pretuned": "MedViT (pre-tune)",
        "medvit_postuned": "MedViT (tuned)",
    }
    df["name"] = df["model"].map(pretty).fillna(df["model"])
    df["internal_auc"] = df["mrnet_val_auc"]
    df["external_auc"] = df["AUC"]
    df["drop"] = (df["internal_auc"] - df["external_auc"]).round(3)
    return df


@st.cache_data
def load_augmentation() -> pd.DataFrame | None:
    p = RESULTS / "augmentation_comparison.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["preset"] = df["preset"].str.capitalize()
    return df[["preset", "auc", "accuracy", "f1", "sensitivity", "specificity"]]


def fig(*parts: str) -> Path | None:
    p = FIG.joinpath(*parts)
    return p if p.exists() else None


ext = load_external()

# --------------------------------------------------------------------- hero ---
st.markdown(
    """<div class="hero"><h1>🦵 ACL-Tear Detection on Knee MRI</h1></div>
    <div class="sub">Transfer-learned CNN/transformer with learned slice-attention ·
    screened 3 backbones · stress-tested on a second hospital's scanner</div>""",
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="pill pill-tech">PyTorch</span>'
    '<span class="pill pill-tech">Transfer learning</span>'
    '<span class="pill pill-tech">Contrastive (SupCon)</span>'
    '<span class="pill pill-tech">CBAM attention</span>'
    '<span class="pill pill-tech">Ray tuning</span>'
    '<span class="pill pill-tech">Grad-CAM++</span>',
    unsafe_allow_html=True,
)
st.write("")

# best / worst by external AUC (the metric that matters for deployment)
best = ext.loc[ext["external_auc"].idxmax()]
base = ext[ext["model"] == "alexnet_baseline"].iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Best internal AUC", f"{best['internal_auc']:.3f}", best["name"].split(" (")[0])
k2.metric("Held on unseen scanner", f"{best['external_auc']:.3f}", f"−{best['drop']:.3f} vs internal",
          delta_color="off")
k3.metric("Baseline on unseen scanner", f"{base['external_auc']:.3f}", f"−{base['drop']:.3f} collapse",
          delta_color="inverse")
k4.metric("External test scans", f"{int(best['n_scored'])}", "Rijeka KneeMRI", delta_color="off")

st.divider()

tab1, tab2, tab3 = st.tabs(
    ["🎯 Does it generalize?", "🔬 Interpretability", "⚙️ Ablations & method"]
)

# ============================================================ TAB 1 — GENERALIZE
with tab1:
    st.markdown(
        '<p class="lead">The real test isn\'t validation accuracy — it\'s whether the model '
        'still works on images from a hospital and scanner it never trained on. '
        'Every model was frozen and applied <b>zero-shot</b> to the external Rijeka dataset.</p>',
        unsafe_allow_html=True,
    )

    chart_df = ext.set_index("name")[["internal_auc", "external_auc"]].rename(
        columns={"internal_auc": "Internal (MRNet)", "external_auc": "External (Rijeka, zero-shot)"}
    )
    st.bar_chart(chart_df, color=["#c7d2fe", "#4f46e5"], horizontal=True)
    st.markdown(
        '<span class="cap">Bars: validation AUC on the original data vs. the same frozen model on a '
        'different hospital\'s scans. Smaller drop = more robust.</span>',
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("**Per-model breakdown**")
    show = ext[["name", "internal_auc", "external_auc", "drop", "Sensitivity", "Specificity"]].copy()
    show.columns = ["Model", "Internal AUC", "External AUC", "AUC drop", "Sensitivity", "Specificity"]
    st.dataframe(
        show.sort_values("External AUC", ascending=False),
        hide_index=True,
        width="stretch",
        column_config={
            "External AUC": st.column_config.NumberColumn(format="%.3f"),
            "Internal AUC": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.info(
        f"**Takeaway:** the tuned DenseNet121 + CBAM retains **{best['external_auc']:.3f}** AUC on unseen-scanner "
        f"data, while the AlexNet baseline collapses to **{base['external_auc']:.3f}**. "
        "Architecture and attention choices — not just training accuracy — drive real-world robustness."
    )

    c1, c2 = st.columns(2)
    if (f := fig("05_external_validation", "internal_vs_external_auc.png")):
        c1.image(str(f), caption="Internal vs external AUC", width="stretch")
    if (f := fig("05_external_validation", "roc_and_scores.png")):
        c2.image(str(f), caption="ROC & score distributions (external)", width="stretch")

# ========================================================== TAB 2 — INTERPRET
with tab2:
    st.markdown(
        '<p class="lead">Grad-CAM++ overlays show <b>where</b> the model looks when it calls a tear. '
        'Comparing correct vs. incorrect cases is how you build trust in a medical model — '
        'and how you catch one that\'s right for the wrong reasons.</p>',
        unsafe_allow_html=True,
    )

    model_map = {
        "DenseNet121 + CBAM (tuned)": "densenet121_cbam_postuned",
        "MedViT (tuned)": "medvit_postuned",
        "AlexNet (baseline)": "alexnet_baseline",
    }
    pick = st.selectbox("Model", list(model_map))
    slug = model_map[pick]

    cols = st.columns(2)
    if (f := fig("06_interpretability", f"{slug}_correct.png")):
        cols[0].markdown("**✅ Correct prediction**")
        cols[0].image(str(f), width="stretch")
    if (f := fig("06_interpretability", f"{slug}_incorrect.png")):
        cols[1].markdown("**❌ Incorrect prediction**")
        cols[1].image(str(f), width="stretch")

    if (f := fig("06_interpretability", f"{slug}_slice_attention.png")):
        st.markdown("**Learned slice-attention** — which MRI slices the model weighted most")
        st.image(str(f), width="stretch")
    elif (f := fig("06_interpretability", f"{slug}_interpretability.png")):
        st.image(str(f), width="stretch")

# ============================================================ TAB 3 — ABLATIONS
with tab3:
    st.markdown(
        '<p class="lead">The final model was chosen by systematic ablation, not guesswork: '
        'backbone screening, an augmentation sweep, CBAM / contrastive pre-training, '
        'then Ray-distributed hyperparameter search.</p>',
        unsafe_allow_html=True,
    )

    aug = load_augmentation()
    if aug is not None:
        st.markdown("**Augmentation strength sweep** (validation)")
        st.bar_chart(aug.set_index("preset")[["auc", "f1"]], color=["#4f46e5", "#a5b4fc"])
        st.markdown(
            '<span class="cap">"Strong" augmentation gave the best AUC/specificity balance; '
            'aggressive light/medium presets inflated recall by predicting almost everything positive.</span>',
            unsafe_allow_html=True,
        )
        st.dataframe(aug, hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    if (f := fig("03_ablations", "ablation_bars.png")):
        c1.image(str(f), caption="CBAM / contrastive ablations", width="stretch")
    if (f := fig("03_ablations", "medvit_cbam_collapse.png")):
        c2.image(str(f), caption="MedViT + CBAM training collapse (a negative result worth keeping)",
                 width="stretch")

    st.markdown(
        "**Best configuration** (Ray search): DenseNet121 + CBAM · AdamW · lr 1e-4 · "
        "weight-decay 0.1 · dropout 0.5 · grad-accum 8 → **0.963 tuned val AUC**."
    )

st.divider()
st.caption(
    "Built by Viet Tran · MSc AI for Medicine, UCD. Screening → augmentation → ablation → tuning → "
    "external validation → interpretability, on one reproducible pipeline."
)
