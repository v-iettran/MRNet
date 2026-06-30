"""Supervised Contrastive Learning (Khosla et al., NeurIPS 2020).

Owner: Viet

Two-stage scheme:
  1. Pretrain the encoder with a supervised contrastive loss on exam-level
     embeddings (projection head on top of the backbone + slice pooling).
  2. Freeze the encoder and train a linear classifier head on top.

Run as a SEPARATE single-factor ablation (not combined with CBAM):
  * winning architecture vs winning architecture + SupCon
  * MedViT + SupCon regardless of the sweep winner

The encoder here is an ``MRNetModel`` (model_factory.py): we use its
``forward_features(exam) -> (1, feat_dim)`` to get one embedding per exam.

NOTE on batching: the DataLoader yields one exam at a time (variable #slices),
but SupCon needs several exams per batch to form positive/negative pairs. We
therefore accumulate ``supcon_batch`` exam embeddings, then compute the loss and
step. Lower ``supcon_batch`` if you hit GPU OOM.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection head mapping exam embeddings to the contrastive space."""

    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SupConLoss(nn.Module):
    """Supervised contrastive loss (single view per sample).

    ``forward`` takes L2-normalized embeddings ``(B, dim)`` and labels ``(B,)``
    and returns a scalar loss. Samples that have no same-label partner in the
    batch are ignored.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        batch_size = embeddings.shape[0]

        labels = labels.contiguous().view(-1, 1)
        same_label = torch.eq(labels, labels.T).float().to(device)   # (B, B)

        # cosine-similarity logits (embeddings are already normalized)
        logits = torch.matmul(embeddings, embeddings.T) / self.temperature
        # numerical stability
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # remove self-comparisons
        self_mask = torch.eye(batch_size, device=device)
        positive_mask = same_label * (1.0 - self_mask)

        exp_logits = torch.exp(logits) * (1.0 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        num_positives = positive_mask.sum(dim=1)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / num_positives.clamp(min=1)

        loss = -mean_log_prob_pos[num_positives > 0]
        if loss.numel() == 0:
            return torch.zeros((), device=device, requires_grad=True)
        return loss.mean()


def pretrain_encoder(model, train_loader, val_loader=None, *, epochs=10,
                     supcon_batch=8, temperature=0.07, lr=1e-4,
                     proj_hidden=512, proj_dim=128, device=None, verbose=True,
                     use_amp=True):
    """Stage 1: contrastively pretrain the encoder (an ``MRNetModel``).

    ``use_amp`` enables mixed precision on CUDA (T4 speedup + lower memory,
    which also helps avoid OOM with larger ``supcon_batch``).

    Returns ``(model, history)`` with the encoder's weights updated in place.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp_on = bool(use_amp) and str(device).startswith("cuda")
    model.to(device)
    model.train()

    projection = ProjectionHead(model.feat_dim, proj_hidden, proj_dim).to(device)
    criterion = SupConLoss(temperature)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(projection.parameters()), lr=lr
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)

    history = []
    for epoch in range(1, epochs + 1):
        embeds, labels = [], []
        running_loss, num_steps = 0.0, 0

        def _flush(embeds, labels):
            with torch.cuda.amp.autocast(enabled=amp_on):
                loss = criterion(torch.cat(embeds, dim=0), torch.cat(labels, dim=0))
            optimizer.zero_grad()
            if amp_on:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            return loss.item()

        for image, label in train_loader:
            image = image.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_on):
                z = projection(model.forward_features(image))   # (1, proj_dim)
                z = F.normalize(z, dim=1)
            embeds.append(z)
            labels.append(label.view(-1).to(device))

            if len(embeds) >= supcon_batch:
                running_loss += _flush(embeds, labels)
                num_steps += 1
                embeds, labels = [], []

        if len(embeds) >= 2:   # leftover partial batch (need >=2 for pairs)
            running_loss += _flush(embeds, labels)
            num_steps += 1

        record = {"epoch": epoch, "supcon_loss": running_loss / max(num_steps, 1)}
        history.append(record)
        if verbose:
            print(f"[SupCon] epoch {epoch}: loss {record['supcon_loss']:.4f}")

    return model, history


def train_linear_classifier(encoder, train_loader, val_loader=None, *,
                            epochs=15, lr=1e-3, criterion=None, device=None,
                            verbose=True, use_amp=True):
    """Stage 2: freeze the encoder and train a fresh linear head on top.

    Args:
        criterion: classification loss. Defaults to ``BCEWithLogitsLoss()``;
            pass ``BCEWithLogitsLoss(pos_weight=...)`` for imbalance handling.
        use_amp: enable mixed precision on CUDA (T4 speedup).

    Returns ``(encoder, history)``. The encoder body is frozen and kept in eval
    mode (so BatchNorm running stats don't drift); only the new head trains.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp_on = bool(use_amp) and str(device).startswith("cuda")
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    encoder.to(device)

    for param in encoder.parameters():
        param.requires_grad = False

    encoder.head = nn.Linear(encoder.feat_dim, 1).to(device)
    for param in encoder.head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(encoder.head.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    encoder.eval()   # freeze BN stats in the (frozen) body

    history = []
    for epoch in range(1, epochs + 1):
        running_loss, num_batches = 0.0, 0
        for image, label in train_loader:
            image = image.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_on):
                logits = encoder(image)                      # (1, 1)
                label_d = label.to(device, non_blocking=True).view_as(logits)
                loss = criterion(logits, label_d)

            optimizer.zero_grad()
            if amp_on:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        record = {"epoch": epoch, "train_loss": running_loss / max(num_batches, 1)}
        if val_loader is not None:
            # training_utils was refactored: the old `validate` is now
            # `validate_one_epoch(model, loader, pos_weight, device)`.
            from .training_utils import validate_one_epoch
            pw = torch.ones(1, device=device)
            val_loss, val_metrics = validate_one_epoch(encoder, val_loader, pw, device)
            record["val_loss"] = val_loss
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(record)
        if verbose:
            print(record)

    return encoder, history
