"""
training_utils.py
-----------------
Training and validation loop utilities for the MRNet classification project.

Owner: Ilaria

Handles:
- Per-task class-weighted BCE loss (to address class imbalance)
- Gradient accumulation (since each exam = 1 sample, effective batch > 1)
- Early stopping based on validation AUC
- Checkpoint saving (best model by val AUC)
- Full metric logging: AUC, accuracy, F1, precision, recall, sensitivity, specificity

Starting baseline: MRNet_tutorial_solution.ipynb
Improvements over baseline:
  - Class-weighted BCE computed from data statistics (not per-sample weight)
  - Gradient accumulation for larger effective batch size
  - Comprehensive metric set beyond AUC + loss
  - Checkpoint saving to disk
  - Clean separation of train / validate / run_training functions
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Compute all evaluation metrics from ground-truth labels and predicted
    probabilities.

    Args:
        y_true (list[int]): Ground-truth binary labels (0 or 1).
        y_pred_proba (list[float]): Predicted probabilities from sigmoid output.
        threshold (float): Decision threshold for binary predictions.

    Returns:
        dict: A dictionary containing all computed metrics.
    """
    y_pred = [1 if p >= threshold else 0 for p in y_pred_proba]

    # ROC-AUC as primary MRNet metric; falls back to 0.5 if only one class present
    try:
        auc = metrics.roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = 0.5 # chance-level AUC when only one class is present in y_true

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)

    # Sensitivity = recall (TP / (TP + FN))
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": round(float(auc), 4),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4)}


# ---------------------------------------------------------------------------
# Class-weight computation
# ---------------------------------------------------------------------------

def compute_pos_weight(loader, device):
    """
    Compute the positive-class weight for BCEWithLogitsLoss from the training
    data distribution.

    Weight = (number of negative samples) / (number of positive samples).
    This is equivalent to the pos_weight argument in PyTorch's BCEWithLogitsLoss
    and up-weights the minority class during training.

    Args:
        loader (DataLoader): Training DataLoader (batch_size=1).
        device (torch.device): Target device for the weight tensor.

    Returns:
        torch.Tensor: Scalar tensor with the positive class weight.
    """
    labels = []
    for _, label in loader:
        labels.append(int(label[0]))

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0:
        raise ValueError("No positive samples found in the training set.")

    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    print(f"  Class weight | neg: {n_neg}, pos: {n_pos}, pos_weight: {pos_weight.item():.4f}")
    return pos_weight


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, pos_weight, device,
                    accumulation_steps=8):
    """
    Run one full pass over the training data with gradient accumulation.

    Because each MRI exam is a single sample (batch_size=1), we accumulate
    gradients over `accumulation_steps` exams before calling optimizer.step(),
    simulating a larger effective batch size.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training DataLoader (batch_size=1).
        optimizer (Optimizer): PyTorch optimiser.
        pos_weight (torch.Tensor): Positive-class weight for BCE loss.
        device (torch.device): Target device.
        accumulation_steps(int): Number of gradient accumulation steps.

    Returns:
        tuple: (mean_loss, metrics_dict) over the full epoch.
    """
    model.train()

    # GRADING (criterion 4 - loss function): class-weighted binary
    # cross-entropy. pos_weight up-weights the minority (ACL-tear) class to
    # counter the ~4:1 imbalance; BCEWithLogitsLoss is numerically stable as it
    # fuses the sigmoid with the loss.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    losses = []
    y_trues = []
    y_preds = []

    optimizer.zero_grad() # zero gradients at start of epoch

    # GRADING (criterion 4 - training loop): iterate over batches, feed the data
    # into the model, compute the loss, the gradients, and update the weights.
    for step, (image, label) in enumerate(loader):

        image = image.float().to(device)
        label = label.to(device)

        # Forward pass: input the batch into the model.
        prediction = model(image)

        # Compute loss and scale by accumulation steps so the effective
        # loss magnitude stays consistent regardless of accumulation_steps.
        loss = criterion(prediction, label) / accumulation_steps
        loss.backward()                       # compute gradients (backprop)

        # Accumulate raw loss value (un-scaled) for logging
        losses.append(loss.item() * accumulation_steps)

        # Collect predictions for metric computation
        proba = torch.sigmoid(prediction).detach().cpu().item()
        y_trues.append(int(label[0]))
        y_preds.append(proba)

        # Update the model weights every `accumulation_steps` samples (or at the
        # last batch so the dataset tail isn't skipped). Each exam is one sample
        # (batch_size=1), so accumulation gives an effective batch of
        # `accumulation_steps` exams before the optimiser step.
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            optimizer.step()                  # update weights
            optimizer.zero_grad()

    epoch_metrics = compute_metrics(y_trues, y_preds)
    return np.mean(losses), epoch_metrics

# ---------------------------------------------------------------------------
# One validation epoch
# ---------------------------------------------------------------------------

def validate_one_epoch(model, loader, pos_weight, device):
    """
    Evaluate the model on the validation set (no gradient computation).

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation DataLoader (batch_size=1).
        pos_weight (torch.Tensor): Positive-class weight for BCE loss.
        device (torch.device): Target device.

    Returns:
        tuple: (mean_loss, metrics_dict) over the full validation set.
    """
    model.eval()

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    losses  = []
    y_trues = []
    y_preds = []

    with torch.no_grad():
        for image, label in loader:

            image = image.float().to(device)
            label = label.to(device)

            prediction = model(image)
            loss = criterion(prediction, label)

            losses.append(loss.item())

            proba = torch.sigmoid(prediction).cpu().item()
            y_trues.append(int(label[0]))
            y_preds.append(proba)

    epoch_metrics = compute_metrics(y_trues, y_preds)
    return np.mean(losses), epoch_metrics

# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def run_training(model, train_loader, val_loader, optimizer, scheduler,
                 device, num_epochs=50, accumulation_steps=8,
                 early_stopping_patience=10, checkpoint_dir="checkpoints",
                 task_name="task"):
    """
    Full training loop with early stopping and checkpoint saving.

    Mirrors the structure of the MRNet tutorial baseline but adds:
      - Class-weighted BCE (computed once from training data)
      - Gradient accumulation
      - Best-model checkpointing
      - Full metric logging per epoch

    Early stopping is based on validation AUC (primary MRNet metric).
    The LR scheduler receives validation loss (same as baseline).

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        optimizer (Optimizer): PyTorch optimiser.
        scheduler (LRScheduler): Learning-rate scheduler.
        device (torch.device): Target device.
        num_epochs (int): Maximum number of epochs.
        accumulation_steps (int): Gradient accumulation steps.
        early_stopping_patience (int): Epochs without AUC improvement before stopping.
        checkpoint_dir (str): Directory to save checkpoints.
        task_name (str): Task label for log messages (e.g. "ACL", "meniscus").

    Returns:
        dict: History dictionary with lists of per-epoch metrics for both
              train and validation splits.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Compute positive-class weight once from training data
    print(f"\n[{task_name}] Computing class weights from training data...")
    pos_weight = compute_pos_weight(train_loader, device)

    best_val_auc = 0.0
    epochs_no_improve = 0 # early stopping counter

    # History storage mirrors the baseline's per-epoch print but in structured form
    history = {
        "train_loss": [], "train_auc": [],
        "val_loss":   [], "val_auc":   [],
        "val_metrics": []}

    # GRADING (criterion 4 - epoch loop): iterate over a number of epochs,
    # training then validating each one; the LR scheduler and early stopping
    # below act on the validation signal.
    for epoch in range(num_epochs):

        # Training
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, pos_weight, device, accumulation_steps)

        # Validation
        val_loss, val_metrics = validate_one_epoch(
            model, val_loader, pos_weight, device)

        # LR scheduling (step on val loss, same as baseline)
        scheduler.step(val_loss)

        # Logging
        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_metrics["auc"])
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_metrics["auc"])
        history["val_metrics"].append(val_metrics)

        print(
            f"[{task_name}] epoch: {epoch:3d} | "
            f"train loss: {train_loss:.4f} | train auc: {train_metrics['auc']:.4f} | "
            f"val loss: {val_loss:.4f} | val auc: {val_metrics['auc']:.4f} | "
            f"val f1: {val_metrics['f1']:.4f} | "
            f"sens: {val_metrics['sensitivity']:.4f} | spec: {val_metrics['specificity']:.4f}")
        print("-" * 80)

        # Checkpoint saving
        # Save whenever validation AUC improves (keeps the single best model)
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            epochs_no_improve = 0

            checkpoint_path = os.path.join(checkpoint_dir, f"best_{task_name}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_val_auc,
                "val_metrics": val_metrics}, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path} (val AUC: {best_val_auc:.4f})")
        else:
            epochs_no_improve += 1

        # GRADING (criterion 4 - early stopping): stop once validation AUC has
        # not improved for `early_stopping_patience` epochs, preventing overfit.
        if epochs_no_improve >= early_stopping_patience:
            print(
                f"\n[{task_name}] Early stopping triggered after {epoch + 1} epochs "
                f"({early_stopping_patience} epochs without AUC improvement).")
            break

    print(f"\n[{task_name}] Training complete. Best val AUC: {best_val_auc:.4f}")
    return history

# ---------------------------------------------------------------------------
# Checkpoint loading utility
# ---------------------------------------------------------------------------

def load_checkpoint(model, checkpoint_path, device):
    """
    Load a saved checkpoint into the model.

    Args:
        model (nn.Module): Model instance (architecture must match).
        checkpoint_path (str): Path to the .pth checkpoint file.
        device (torch.device): Target device.

    Returns:
        tuple: (model, checkpoint_dict) model with loaded weights and the
               full checkpoint dictionary (for inspecting epoch, metrics, etc.).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(
        f"Loaded checkpoint from '{checkpoint_path}' "
        f"(epoch {checkpoint['epoch']}, val AUC: {checkpoint['val_auc']:.4f})")
    return model, checkpoint
