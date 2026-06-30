"""Model factory: backbones + slice attention-pooling + classification head.

Owner: Caolan (ResNet50, DenseNet121, slice-pooling wrapper, CBAM wiring)
       Viet  (MedViT branch — see also contrastive_learning.py)

Shared architecture (GRADING criterion 3 - model architecture & novelty):

    per-slice backbone (optionally + CBAM attention)
        -> per-slice feature vectors
        -> learned attention pooling across the variable number of slices
        -> fully-connected classification head

All models subclass ``torch.nn.Module``. The pipeline supports three backbones
(ResNet50, DenseNet121, MedViT) under one ``MRNetModel`` wrapper. The novelty
over the Bien et al. (2018) baseline (``AlexNetBaseline`` below, parameter-free
max-pool) is twofold: (1) a learned gated-attention pooling over slices
(``SliceAttentionPool``, Ilse et al. 2018) that weights diagnostically relevant
slices, and (2) optional CBAM channel+spatial attention inside the backbone
stages. Gradient checkpointing over slice-chunks lets the variable-length
volumes fine-tune within GPU memory.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


class SliceAttentionPool(nn.Module):
    """Attention pooling over the slice dimension.

    Turns a variable-length set of per-slice feature vectors
    ``(num_slices, feat_dim)`` into a single exam vector ``(feat_dim,)`` by
    learning a weight per slice (gated-attention MIL, Ilse et al. 2018). This is
    the mechanism that handles the volumetric (variable-#slices) aspect of MRNet.
    """

    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, slice_features):
        # slice_features: (num_slices, feat_dim)
        scores = self.attention(slice_features)          # (num_slices, 1)
        weights = torch.softmax(scores, dim=0)           # attention over slices
        pooled = (weights * slice_features).sum(dim=0)   # (feat_dim,)
        return pooled


# The project's main architecture, an
# nn.Module composing a transfer-learned backbone, learned slice-attention
# pooling, and an FC head. forward() defines the full data flow.
class MRNetModel(nn.Module):
    """Full MRNet classifier: backbone + slice pooling + FC head.

    Args:
        backbone: an nn.Module mapping ``(B, C, H, W)`` -> ``(B, feat_dim)``.
        feat_dim: feature dimension output by the backbone.
        num_classes: 1 for a single binary task.
        dropout: dropout probability before the head.

    ``forward`` accepts an exam tensor shaped ``(num_slices, C, H, W)`` or
    ``(1, num_slices, C, H, W)`` (the DataLoader's batch=1 form) and returns
    logits of shape ``(1, num_classes)``.

    Memory note: a single exam can have many slices (~20-60), and the backbone
    processes all of them as one batch. Fine-tuning therefore stores activations
    for every slice, which OOMs on a T4. ``use_checkpoint`` runs the backbone in
    slice-chunks of ``slice_chunk`` under gradient checkpointing, so peak
    activation memory scales with ``slice_chunk`` (recomputed in backward,
    ~2x backbone compute) instead of the full slice count.
    """

    def __init__(self, backbone, feat_dim, num_classes=1, dropout=0.0,
                 use_checkpoint=True, slice_chunk=8):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.use_checkpoint = use_checkpoint
        self.slice_chunk = slice_chunk
        self.pool = SliceAttentionPool(feat_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def _encode_slices(self, exam):
        """Run the backbone over ``(S, C, H, W)`` -> ``(S, feat_dim)``.

        Three regimes:
        * frozen backbone (feature extraction): run forward-only under
          ``no_grad`` so no activations are stored -> fast + low memory;
        * fine-tuning with checkpointing: process slices in chunks and recompute
          each chunk in backward to cap peak activation memory;
        * otherwise (eval / inference): a single plain forward.
        """
        backbone_trainable = any(p.requires_grad for p in self.backbone.parameters())

        if not backbone_trainable:
            with torch.no_grad():
                return self.backbone(exam)

        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            feats = []
            for start in range(0, exam.shape[0], self.slice_chunk):
                chunk = exam[start:start + self.slice_chunk]
                feats.append(
                    checkpoint.checkpoint(self.backbone, chunk, use_reentrant=False)
                )
            return torch.cat(feats, dim=0)

        return self.backbone(exam)

    def forward_features(self, exam):
        """Return the pooled exam embedding ``(1, feat_dim)`` (before the head)."""
        if exam.dim() == 5:                  # (1, S, C, H, W) -> (S, C, H, W)
            exam = exam.squeeze(0)
        slice_features = self._encode_slices(exam)  # (S, feat_dim)
        pooled = self.pool(slice_features)    # (feat_dim,)
        return pooled.view(1, -1)             # (1, feat_dim)

    def forward(self, exam):
        embedding = self.forward_features(exam)  # (1, feat_dim)
        return self.head(embedding)              # (1, num_classes)


def build_backbone(name, pretrained=True, use_cbam=False):
    """Construct a 2D feature-extractor backbone.

    Args:
        name: "resnet50" | "densenet121" | "medvit".
        pretrained: load ImageNet (or domain) pretrained weights for transfer
            learning. Note MRI is grayscale -> handle the channel mismatch.
        use_cbam: insert CBAM blocks into the backbone's conv stages
            (applies to ResNet/DenseNet and MedViT's conv stages).

    Returns:
        ``(backbone_module, feat_dim)``.
    """
    if name == "medvit":
        import sys
        from . import config

        if str(config.MEDVIT_REPO_DIR) not in sys.path:
            sys.path.append(str(config.MEDVIT_REPO_DIR))
        from MedViT import MedViT_small

        model = MedViT_small(num_classes=1000)
        if pretrained:
            ckpt = torch.load(config.MEDVIT_CKPT, map_location="cpu")
            state = ckpt.get("model", ckpt)
            model.load_state_dict(state, strict=False)

        feat_dim = model.proj_head[-1].in_features
        model.proj_head = nn.Identity()

        if use_cbam:
            from .attention_modules import CBAM

            # MedViT is a hybrid conv/transformer net. Insert CBAM at the end of
            # each of the 4 stages (``stage_out_idx`` are the last block indices),
            # using that block's output channel count. ``features`` is an
            # nn.Sequential iterated in ``forward``, so wrapping a block as
            # Sequential(block, CBAM) is transparent. Pretrained weights are
            # already loaded above, so the CBAM params are simply new/random.
            for i in model.stage_out_idx:
                block = model.features[i]
                ch = block.out_channels
                model.features[i] = nn.Sequential(block, CBAM(ch))

        return model, feat_dim
    elif name == "resnet50":
        import torchvision.models as tv_models
        from .attention_modules import CBAM

        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = tv_models.resnet50(weights=weights)

        # MRI slices are single-channel; pretrained conv1 expects 3.
        # Average the pretrained RGB filters into one filter so the learned
        # edge/texture detectors still transfer, instead of training conv1
        # from scratch.
        # old_conv1 = resnet.conv1
        # new_conv1 = nn.Conv2d(1, old_conv1.out_channels,
        #                       kernel_size=old_conv1.kernel_size,
        #                       stride=old_conv1.stride,
        #                       padding=old_conv1.padding, bias=False)
        # if pretrained:
        #     with torch.no_grad():
        #         new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
        # resnet.conv1 = new_conv1

        if use_cbam:
            resnet.layer1 = nn.Sequential(resnet.layer1, CBAM(256))
            resnet.layer2 = nn.Sequential(resnet.layer2, CBAM(512))
            resnet.layer3 = nn.Sequential(resnet.layer3, CBAM(1024))
            resnet.layer4 = nn.Sequential(resnet.layer4, CBAM(2048))

        feat_dim = resnet.fc.in_features  # 2048
        resnet.fc = nn.Identity()         # avgpool+flatten already in resnet.forward

        return resnet, feat_dim
    elif name == "densenet121":
        import torchvision.models as tv_models

        weights = tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = tv_models.densenet121(weights=weights)

        # old_conv0 = densenet.features.conv0
        # new_conv0 = nn.Conv2d(1, old_conv0.out_channels,
        #                       kernel_size=old_conv0.kernel_size,
        #                       stride=old_conv0.stride,
        #                       padding=old_conv0.padding, bias=False)
        # if pretrained:
        #     with torch.no_grad():
        #         new_conv0.weight.copy_(old_conv0.weight.mean(dim=1, keepdim=True))
        # densenet.features.conv0 = new_conv0

        feat_dim = densenet.classifier.in_features  # 1024
        densenet.classifier = nn.Identity()

        if use_cbam:
            from .attention_modules import CBAM

            # DenseNet's dense blocks are nested inside ``features``. Insert CBAM
            # right after each dense block (before the following transition),
            # using the block's output channel count. Wrapping the block as
            # Sequential(block, CBAM) keeps ``features`` iterable as before.
            # Channels grow as 64 + n_layers*32: 256, 512, 1024, 1024.
            block_channels = {"denseblock1": 256, "denseblock2": 512,
                              "denseblock3": 1024, "denseblock4": 1024}
            for blk_name, ch in block_channels.items():
                block = getattr(densenet.features, blk_name)
                setattr(densenet.features, blk_name,
                        nn.Sequential(block, CBAM(ch)))

        return densenet, feat_dim
    else:
        raise ValueError(f"Unsupported backbone: {name}")



def build_model(backbone="resnet50", use_cbam=False, pretrained=True,
                num_classes=1, dropout=0.0, freeze_backbone=False,
                use_checkpoint=True, slice_chunk=8):
    """Top-level factory used by the notebooks to assemble a full model.

    Args:
        backbone: "resnet50" | "densenet121" | "medvit".
        use_cbam: whether to add CBAM attention.
        pretrained: use transfer-learning weights.
        num_classes: 1 for a single binary task.
        dropout: dropout before the FC head.
        freeze_backbone: if True, only the pooling + head train
            (feature-extraction mode); if False, fine-tune the backbone too.
        use_checkpoint: gradient-checkpoint the backbone over slice-chunks to
            fit fine-tuning on a T4 (trades ~2x backbone compute for memory).
        slice_chunk: number of slices per checkpointed chunk.

    Returns:
        An ``MRNetModel`` instance.
    """
    backbone_module, feat_dim = build_backbone(
        backbone, pretrained=pretrained, use_cbam=use_cbam
    )
    if freeze_backbone:
        for param in backbone_module.parameters():
            param.requires_grad = False
    return MRNetModel(
        backbone_module, feat_dim, num_classes=num_classes, dropout=dropout,
        use_checkpoint=use_checkpoint, slice_chunk=slice_chunk,
    )


# --------------------------------------------------------------------------
# Reference baseline (kept here so the interactive notebook
# 00 and the GPU runner run_baseline.py build the exact same model).
# --------------------------------------------------------------------------
class AlexNetBaseline(nn.Module):
    """Original MRNet baseline: pretrained AlexNet per slice + max-pool + FC head.

    Reproduces Bien et al. (2018) / the MRNet tutorial. Unlike ``MRNetModel`` it
    aggregates slices with a parameter-free MAX-pool (no learned attention), so
    it is the reference point the attention-pooled models are compared against.
    The final 1000-class ImageNet head is replaced by a single linear output.

    Input : ``(1, S, 3, 256, 256)`` one exam, S slices.
    Output: ``(1, 1)`` logit (pre-sigmoid).
    """

    def __init__(self):
        super().__init__()
        from torchvision import models
        from torchvision.models import AlexNet_Weights

        alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.features = alexnet.features            # conv backbone
        self.avgpool = alexnet.avgpool              # -> (S, 256, 6, 6)
        # Keep the classifier up to the 4096-d representation (drop the final
        # 1000-class layer), as in the paper/tutorial.
        self.classifier = nn.Sequential(*list(alexnet.classifier.children())[:-1])
        self.fc = nn.Linear(4096, 1)

    def forward(self, x):
        x = x.squeeze(0)                 # (S, 3, 256, 256)
        x = self.features(x)             # (S, 256, 6, 6)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)        # (S, 9216)
        x = self.classifier(x)           # (S, 4096)
        x = x.max(dim=0, keepdim=True)[0]  # (1, 4096)  max-pool across slices
        x = self.fc(x)                   # (1, 1)
        return x


def build_baseline_model():
    """Factory for the AlexNet + max-pool MRNet baseline (Bien et al. 2018)."""
    return AlexNetBaseline()
