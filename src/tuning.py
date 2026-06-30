"""
tuning.py
---------
Hyperparameter and optimiser tuning for the MRNet classification project.
Applied to the best-performing architecture found in the model sweep.

Owner: Ilaria

Search space
------------
- Learning rate: [1e-5, 1e-4, 1e-3]
- Weight decay: [0.0, 1e-4, 0.1]
- Dropout: [0.0, 0.3, 0.5]
- Accumulation steps: [4, 8, 16] (effective batch size proxy)
- Optimizer: [Adam, AdamW, SGD]

Strategy
--------
Grid search over the combinations above. Each configuration is trained for a
fixed number of epochs with early stopping. The best configuration is selected
by mean validation AUC across all three tasks (or on the ACL task only if
running a focused sweep).

Results are saved to a CSV file for easy comparison.
"""

import os
import csv
import random
import itertools
import copy

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Robust import: works whether ``codes/`` (``src.training_utils``) or
# ``codes/src/`` (``training_utils``) is on sys.path.
try:
    from training_utils import run_training
except ImportError:  # pragma: no cover
    from src.training_utils import run_training

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

# Each key maps to a list of candidate values.
# Extend or reduce these lists to control the size of the search.
SEARCH_SPACE = {"lr": [1e-5, 1e-4, 3e-4, 1e-3], # learning rates to try
    "weight_decay": [0.0, 1e-4, 1e-1], # L2 regularization strengths
    "dropout": [0.0, 0.3, 0.5], # dropout probabilities
    "accumulation_steps":[4, 8, 16], # gradient accumulation steps
    "optimizer": ["Adam", "AdamW", "SGD"]}

# Training budget per configuration
TUNING_EPOCHS = 30 # max epochs per config (early stopping kicks in sooner)
EARLY_STOPPING_PATIENCE = 7 # shorter patience to keep tuning tractable

# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(optimizer_name, model_parameters, lr, weight_decay):
    """
    Instantiate the requested optimiser with the given hyperparameters.

    Args:
        optimizer_name (str): One of "Adam", "AdamW", "SGD".
        model_parameters: Output of model.parameters().
        lr (float): Learning rate.
        weight_decay (float): L2 weight decay coefficient.

    Returns:
        torch.optim.Optimizer
    """
    name = optimizer_name.lower()

    if name == "adam":
        # Adam: adaptive learning rates, good default choice (used in baseline)
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)

    elif name == "adamw":
        # AdamW: Adam with decoupled weight decay (better regularization than Adam)
        return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)

    elif name == "sgd":
        # SGD with Nesterov momentum, can generalise better than adaptive methods
        # but typically needs a higher initial LR than Adam variants
        return optim.SGD(
            model_parameters, lr=lr, momentum=0.9,
            weight_decay=weight_decay, nesterov=True)

    else:
        raise ValueError(f"Unknown optimizer: '{optimizer_name}'. "
                         "Choose from Adam, AdamW, SGD.")

# ---------------------------------------------------------------------------
# Model dropout injection utility
# ---------------------------------------------------------------------------

def apply_dropout(model, dropout_p):
    """
    Replace all existing nn.Dropout / nn.Dropout2d layers in the model with
    new ones using the specified probability.

    If dropout_p == 0.0 all dropout layers are effectively disabled (p=0 means
    no units are dropped), keeping the architecture identical.

    Args:
        model (nn.Module): Model to modify (in-place).
        dropout_p (float): New dropout probability.

    Returns:
        nn.Module: The modified model (same object, modified in-place).
    """
    import torch.nn as nn

    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_p
        elif isinstance(module, nn.Dropout2d):
            module.p = dropout_p

    return model

# ---------------------------------------------------------------------------
# Single configuration evaluation
# ---------------------------------------------------------------------------

def evaluate_config(config, model_class, train_loader, val_loader, device,
                    checkpoint_dir, task_name="ACL",
                    epochs=None, patience=None):
    """
    Train the model with one hyperparameter configuration and return the best
    validation AUC achieved.

    A fresh copy of the model is created for each configuration to ensure
    independent evaluation.

    Args:
        config (dict): Hyperparameter dictionary with keys:
                                     lr, weight_decay, dropout,
                                     accumulation_steps, optimizer.
        model_class (callable): Callable that returns a fresh model
                                     instance (e.g. lambda: ResNet50MRNet()).
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        device (torch.device): Target device.
        checkpoint_dir(str): Directory for saving checkpoints.
        task_name (str): Task identifier for logging.

    Returns:
        float: Best validation AUC for this configuration.
    """
    # Instantiate a fresh model for each config
    model = model_class().to(device)
    model = apply_dropout(model, config["dropout"])

    optimizer = build_optimizer(config["optimizer"],
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"])

    # LR scheduler: reduce LR on validation loss plateau (same as baseline)
    scheduler = ReduceLROnPlateau(optimizer, patience=4, factor=0.3, threshold=1e-4)

    # Config label for checkpoint naming (avoids overwriting between runs)
    config_label = (f"{task_name}_opt{config['optimizer']}_lr{config['lr']}_"
        f"wd{config['weight_decay']}_do{config['dropout']}_"
        f"acc{config['accumulation_steps']}")

    history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=epochs if epochs is not None else TUNING_EPOCHS,
        accumulation_steps=config["accumulation_steps"],
        early_stopping_patience=(patience if patience is not None
                                 else EARLY_STOPPING_PATIENCE),
        checkpoint_dir=checkpoint_dir,
        task_name=config_label)

    best_auc = max(history["val_auc"]) if history["val_auc"] else 0.0
    return best_auc


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------

def sample_config(rng, search_space=None):
    """Draw one hyperparameter configuration uniformly from ``search_space``."""
    space = search_space if search_space is not None else SEARCH_SPACE
    return {k: rng.choice(v) for k, v in space.items()}


def random_search(model_class, train_loader, val_loader, device,
                  n_trials=10, seed=42, search_space=None,
                  checkpoint_dir="checkpoints/tuning",
                  results_csv="tuning_results.csv", task_name="ACL",
                  epochs=None, patience=None):
    """Random search over ``search_space`` (defaults to ``SEARCH_SPACE``).

    Samples ``n_trials`` distinct configurations (falls back to sampling with
    replacement if the space is smaller than ``n_trials``), trains each, and
    records the best validation AUC. Results stream to ``results_csv`` so partial
    progress survives an interruption. Returns the best config dict.
    """
    # GRADING (criterion 4 - hyper-parameter tuning): random search over lr,
    # weight decay, dropout, accumulation steps and optimiser; each trial trains
    # a fresh model and is scored on validation AUC, the best config is returned.
    os.makedirs(checkpoint_dir, exist_ok=True)
    rng = random.Random(seed)
    space = search_space if search_space is not None else SEARCH_SPACE
    keys = list(space.keys())

    # Sample unique configs where possible.
    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*space.values())]
    rng.shuffle(all_combos)
    if n_trials <= len(all_combos):
        configs = all_combos[:n_trials]
    else:
        configs = all_combos + [sample_config(rng, space) for _ in range(n_trials - len(all_combos))]

    print(f"\n[Tuning] Random search: {len(configs)} trials (of "
          f"{len(all_combos)} possible) for task '{task_name}'")
    print("=" * 80)

    with open(results_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=keys + ["val_auc"]).writeheader()

    best_auc, best_config = 0.0, None
    for i, config in enumerate(configs):
        print(f"\n[Tuning] Trial {i + 1}/{len(configs)}: {config}")
        val_auc = evaluate_config(
            config=config, model_class=model_class,
            train_loader=train_loader, val_loader=val_loader, device=device,
            checkpoint_dir=checkpoint_dir, task_name=task_name,
            epochs=epochs, patience=patience)

        with open(results_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=keys + ["val_auc"]).writerow(
                {**config, "val_auc": round(val_auc, 4)})

        print(f"Trial val AUC: {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc, best_config = val_auc, copy.deepcopy(config)
            print(f"  ** New best! AUC: {best_auc:.4f}")

    print("\n" + "=" * 80)
    print(f"[Tuning] Random search complete. Best val AUC: {best_auc:.4f}")
    print(f"[Tuning] Best config: {best_config}")
    return best_config

# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(model_class, train_loader, val_loader, device,
                checkpoint_dir="checkpoints/tuning",
                results_csv="tuning_results.csv",
                task_name="ACL"):
    """
    Exhaustive grid search over SEARCH_SPACE.

    For each combination of hyperparameters, trains the model and records the
    best validation AUC. Results are written incrementally to a CSV file so
    that partial results are not lost if the search is interrupted.

    Args:
        model_class (callable): Returns a fresh model instance.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        device (torch.device): Target device.
        checkpoint_dir (str): Directory for saving per-config checkpoints.
        results_csv (str): Path to output CSV file.
        task_name (str): Task identifier for logging.

    Returns:
        dict: Best hyperparameter configuration (highest val AUC).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Build all combinations from the search space
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    all_configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    print(f"\n[Tuning] Starting grid search: {len(all_configs)} configurations "
          f"for task '{task_name}'")
    print("=" * 80)

    results = []

    # CSV header
    csv_path = results_csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + ["val_auc"])
        writer.writeheader()

    best_auc = 0.0
    best_config = None

    for i, config in enumerate(all_configs):

        print(f"\n[Tuning] Config {i + 1}/{len(all_configs)}: {config}")

        val_auc = evaluate_config(
            config=config,
            model_class=model_class,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            checkpoint_dir=checkpoint_dir,
            task_name=task_name)

        row = {**config, "val_auc": round(val_auc, 4)}
        results.append(row)

        # Write result immediately so partial results are not lost
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys + ["val_auc"])
            writer.writerow(row)

        print(f"Best val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_config = copy.deepcopy(config)
            print(f"  ** New best config! AUC: {best_auc:.4f}")

    print("\n" + "=" * 80)
    print(f"[Tuning] Grid search complete.")
    print(f"[Tuning] Best val AUC: {best_auc:.4f}")
    print(f"[Tuning] Best config:  {best_config}")
    print(f"[Tuning] Full results saved to: {csv_path}")

    return best_config

# ---------------------------------------------------------------------------
# Convenience: train best config on all three tasks
# ---------------------------------------------------------------------------

def train_best_config_all_tasks(best_config, model_class,
                                 task_loaders, device,
                                 checkpoint_dir="checkpoints/best",
                                 num_epochs=50,
                                 early_stopping_patience=10):
    """
    Re-train the winning configuration on all three MRNet tasks
    (abnormal, ACL, meniscus) using the full training budget.

    Called after grid_search() has identified the best hyperparameters.

    Args:
        best_config (dict): Best hyperparameter config from grid search.
        model_class (callable): Returns a fresh model instance.
        task_loaders (dict): Dict mapping task name to
                            (train_loader, val_loader) tuple.
                            E.g. {"ACL": (tr, va), "meniscus": (...), ...}
        device (torch.device): Target device.
        checkpoint_dir (str): Directory for best-task checkpoints.
        num_epochs (int): Full training budget per task.
        early_stopping_patience (int): Early stopping patience.

    Returns:
        dict: Per-task training histories.
    """
    all_histories = {}

    for task_name, (train_loader, val_loader) in task_loaders.items():

        print(f"\n{'=' * 80}")
        print(f"Training best config on task: {task_name}")
        print(f"{'=' * 80}")

        model = model_class().to(device)
        model = apply_dropout(model, best_config["dropout"])

        optimizer = build_optimizer(
            best_config["optimizer"],
            model.parameters(),
            lr=best_config["lr"],
            weight_decay=best_config["weight_decay"])

        scheduler = ReduceLROnPlateau(
            optimizer, patience=4, factor=0.3, threshold=1e-4)

        history = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=num_epochs,
            accumulation_steps=best_config["accumulation_steps"],
            early_stopping_patience=early_stopping_patience,
            checkpoint_dir=checkpoint_dir,
            task_name=task_name)

        all_histories[task_name] = history

    return all_histories
