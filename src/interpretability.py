"""Interpretability: Grad-CAM / Grad-CAM++ (CNNs and the MedViT hybrid).

Owner: Noma + Sonia

Goal (see Project Pipeline.md, section 7): check whether predictions are
driven by clinically relevant anatomical regions.

Design notes
------------
Every model in this project is wrapped by ``MRNetModel`` (per-slice backbone ->
``SliceAttentionPool`` over the variable slice stack -> FC head), so a single
"prediction" is one logit per *exam*, not per slice. Visualising a heatmap
therefore needs two pieces of information:

* *which* slice to look at  -> taken from the slice-attention weights
  (``MRNetModel.pool``) when present, else the slice with the most CAM energy;
* *where* in that slice the model looked -> Grad-CAM / Grad-CAM++ on the last
  spatial feature map.

Grad-CAM++ works for both families here: DenseNet121 ends in a conv feature map
before global average pooling, and MedViT (despite being a conv+transformer
hybrid) *also* ends in a ``(B, C, H, W)`` map -> BatchNorm -> GAP with no class
token, so the same CAM machinery applies directly (no ``reshape_transform``
needed, unlike a plain ViT). This makes the two architectures directly
comparable under one explanation method.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Layer / attention helpers
# ---------------------------------------------------------------------------
def get_target_layer(model, backbone):
    """Return the conv layer to hook for Grad-CAM on a given backbone.

    Args:
        model: a built model (``MRNetModel`` or the AlexNet baseline).
        backbone: "densenet121" | "medvit" | "alexnet_baseline" | "resnet50".

    Returns:
        The ``nn.Module`` whose output is the last spatial feature map.
    """
    name = (backbone or "").lower()
    if name == "densenet121":
        # Last dense block (wrapped with CBAM when use_cbam=True): the final
        # 1024-channel spatial map. We hook here rather than features.norm5
        # because torchvision applies an in-place ``F.relu`` to norm5's output,
        # which is incompatible with a backward hook on that tensor.
        return model.backbone.features.denseblock4
    if name == "medvit":
        # MedViT keeps a (B, C, H, W) map right up to GAP; norm is the last one
        # and its output is consumed out-of-place by avgpool.
        return model.backbone.norm
    if name == "resnet50":
        return model.backbone.layer4
    if name in ("alexnet_baseline", "alexnet"):
        # Final feature-stage module (a MaxPool); its output feeds avgpool
        # out-of-place, unlike the conv layers which are followed by in-place
        # ReLUs that would clash with the backward hook.
        return model.features[-1]
    # Generic fallback: the last Conv2d anywhere in the model.
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise ValueError(f"No conv layer found for backbone={backbone!r}")
    return last


def get_slice_attention(model, exam, device="cuda"):
    """Return the per-slice attention weights ``(num_slices,)`` or ``None``.

    Only ``MRNetModel`` has a ``pool`` (``SliceAttentionPool``); the AlexNet
    max-pool baseline has no learned slice weights, so this returns ``None``.
    """
    if not hasattr(model, "pool"):
        return None
    store = {}

    def hook(module, inputs, output):
        slice_features = inputs[0]
        scores = module.attention(slice_features)
        store["w"] = torch.softmax(scores, dim=0).detach().squeeze(-1).cpu()

    handle = model.pool.register_forward_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            model(exam.to(device).float())
    finally:
        handle.remove()
    return store.get("w")


# ---------------------------------------------------------------------------
# Core CAM computation
# ---------------------------------------------------------------------------
def compute_cam(model, exam, target_layer, device="cuda", target_class=None,
                plusplus=True):
    """Compute per-slice CAM heatmaps for one exam.

    Args:
        model: the model to explain (in any mode; switched to eval here).
        exam: input exam tensor ``(num_slices, C, H, W)`` or
            ``(1, num_slices, C, H, W)``.
        target_layer: conv layer to hook (see ``get_target_layer``).
        device: "cuda" / "cpu".
        target_class: 1 to explain ACL-injury evidence, 0 to explain the
            no-injury direction. ``None`` -> the model's own predicted class.
        plusplus: True for Grad-CAM++, False for vanilla Grad-CAM.

    Returns:
        ``(cam, attn, prob)`` where
        * ``cam``  : ``(num_slices, h, w)`` CPU tensor, per-slice min-max
          normalised to ``[0, 1]``;
        * ``attn`` : ``(num_slices,)`` slice-attention weights or ``None``;
        * ``prob`` : scalar sigmoid probability of the positive class.
    """
    model.eval()
    acts, grads, attn = {}, {}, {}

    def fwd_hook(module, inputs, output):
        acts["v"] = output

    def bwd_hook(module, grad_input, grad_output):
        grads["v"] = grad_output[0]

    handles = [
        target_layer.register_forward_hook(fwd_hook),
        target_layer.register_full_backward_hook(bwd_hook),
    ]
    if hasattr(model, "pool"):
        def pool_hook(module, inputs, output):
            scores = module.attention(inputs[0])
            attn["w"] = torch.softmax(scores, dim=0).detach().squeeze(-1).cpu()
        handles.append(model.pool.register_forward_hook(pool_hook))

    try:
        exam = exam.to(device).float()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logit = model(exam).reshape(())  # scalar logit for the exam
            if target_class is None:
                target_class = 1 if logit.item() >= 0 else 0
            score = logit if target_class == 1 else -logit
            score.backward()

        A = acts["v"].detach()   # (S, C, h, w)
        G = grads["v"].detach()  # (S, C, h, w)

        if plusplus:
            # Grad-CAM++ (Chattopadhyay et al. 2018) pixel-wise weighting.
            G2 = G * G
            G3 = G2 * G
            sum_A = A.sum(dim=(2, 3), keepdim=True)
            denom = 2.0 * G2 + sum_A * G3
            denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
            alpha = G2 / denom
            weights = (alpha * F.relu(G)).sum(dim=(2, 3), keepdim=True)
        else:
            weights = G.mean(dim=(2, 3), keepdim=True)

        cam = F.relu((weights * A).sum(dim=1))  # (S, h, w)
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        prob = torch.sigmoid(logit.detach()).item()
        return cam.cpu(), attn.get("w"), prob
    finally:
        for h in handles:
            h.remove()


def grad_cam(model, exam, target_layer, target_class=None, device="cuda"):
    """Vanilla Grad-CAM heatmaps ``(num_slices, h, w)`` for one exam."""
    cam, _, _ = compute_cam(model, exam, target_layer, device=device,
                            target_class=target_class, plusplus=False)
    return cam


def grad_cam_plusplus(model, exam, target_layer, target_class=None,
                      device="cuda"):
    """Grad-CAM++ heatmaps ``(num_slices, h, w)`` for one exam."""
    cam, _, _ = compute_cam(model, exam, target_layer, device=device,
                            target_class=target_class, plusplus=True)
    return cam


def explain_exam(model, exam, target_layer, device="cuda", plusplus=True,
                 target_class=None, select_slice="auto"):
    """Explain one exam and pick a representative slice to display.

    Returns a dict with ``cam`` (per-slice maps), ``attn`` (slice weights or
    ``None``), ``prob``, and ``slice`` (the chosen slice index). The slice is
    chosen from the attention weights when available, otherwise from the slice
    with the highest CAM energy.
    """
    cam, attn, prob = compute_cam(model, exam, target_layer, device=device,
                                  target_class=target_class, plusplus=plusplus)
    if select_slice == "auto":
        if attn is not None:
            sel = int(torch.as_tensor(attn).reshape(-1).argmax())
        else:
            sel = int(cam.flatten(1).sum(dim=1).argmax())
    else:
        sel = int(select_slice)
    return {"cam": cam, "attn": attn, "prob": prob, "slice": sel}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def overlay_heatmap(slice_image, heatmap, alpha=0.5, colormap="jet"):
    """Overlay a CAM heatmap on a grayscale MRI slice.

    Args:
        slice_image: 2D array (any scale) for the underlying slice.
        heatmap: 2D CAM in ``[0, 1]``; resized to the slice if shapes differ.
        alpha: blending factor for the colored overlay.
        colormap: matplotlib colormap name.

    Returns:
        An ``(H, W, 3)`` float RGB image in ``[0, 1]``.
    """
    try:
        from matplotlib import colormaps
        cmap = colormaps[colormap]
    except Exception:  # older matplotlib
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)

    img = np.asarray(slice_image, dtype=np.float32)
    span = float(img.max() - img.min())
    img = (img - img.min()) / (span + 1e-8)

    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.shape != img.shape:
        hm_t = torch.from_numpy(hm)[None, None]
        hm = F.interpolate(hm_t, size=img.shape, mode="bilinear",
                           align_corners=False)[0, 0].numpy()

    colored = cmap(hm)[..., :3]
    gray = np.stack([img, img, img], axis=-1)
    out = (1.0 - alpha) * gray + alpha * colored
    return np.clip(out, 0.0, 1.0)


def attention_rollout(model, exam, device="cuda", **kwargs):
    """Attention rollout for pure-ViT models.

    Not used in this project: MedViT only has spatial-reduction self-attention
    in its LTB blocks (non-square attention, no class token), so rollout is not
    directly applicable. We explain MedViT with Grad-CAM++ instead (see module
    docstring). Kept as a stable stub for the notebook scaffold.
    """
    raise NotImplementedError(
        "attention_rollout is intentionally unused; MedViT is explained with "
        "grad_cam_plusplus (see interpretability module docstring)."
    )
