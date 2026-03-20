"""
MAPS baseline training and evaluation.

Implements the MAPS (Machine learning for Analysis of Proteomics in Spatial biology)
MLP classifier from the Mahmood Lab for cell type classification in multiplexed imaging data.

Reference:
- Paper: Nature Communications 2023, DOI: 10.1038/s41467-023-44188-w
- Code: https://github.com/mahmoodlab/MAPS
"""

import os
import click
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from maps.model import MAPSModel

# Default data directory from environment
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data2"))

from deepcelltypes.config import TissueNetConfig, CELL_TYPE_HIERARCHY
from deepcelltypes.utils import (
    compute_baseline_metrics,
    save_baseline_predictions,
    extract_features_from_zarr,
)


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute class weights for WeightedRandomSampler.

    Args:
        y: (N,) labels

    Returns:
        sample_weights: (N,) weight for each sample
    """
    class_counts = np.bincount(y)
    # Sqrt-inverse frequency weighting (less aggressive than full inverse,
    # matches transformer baseline's WeightedRandomSampler approach)
    class_weights = 1.0 / (np.sqrt(class_counts) + 1e-8)
    sample_weights = class_weights[y]
    return sample_weights


def normalize_features(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize features.

    Args:
        X: (N, D) feature matrix
        mean: Optional pre-computed mean (for test set normalization)
        std: Optional pre-computed std (for test set normalization)

    Returns:
        X_norm: (N, D) normalized features
        mean: (D,) feature means
        std: (D,) feature stds
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)

    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)

    X_norm = (X - mean) / std
    return X_norm, mean, std


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train for one epoch.

    Args:
        model: MAPS model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits, _ = model(X_batch)  # Use logits for loss computation
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on dataset.

    Args:
        model: MAPS model
        X: (N, D) features (normalized)
        y: (N,) true labels
        device: Device to use
        batch_size: Batch size for inference

    Returns:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
    """
    model.eval()
    all_prob = []

    # Process in batches
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size].to(device)
        _, probs = model(X_batch)  # Use probs for evaluation
        all_prob.append(probs.cpu().numpy())

    y_prob = np.concatenate(all_prob, axis=0)
    y_pred = y_prob.argmax(axis=1)

    return y, y_pred, y_prob


@click.command()
@click.option("--model_name", type=str, default="maps_0")
@click.option("--device_num", type=str, default="cuda:0")
@click.option("--enable_wandb", type=bool, default=False)
@click.option(
    "--zarr_dir",
    type=str,
    default=str(DATA_DIR / "tissuenet-caitlin-labels.zarr"),
)
@click.option(
    "--skip_datasets",
    type=str,
    multiple=True,
    default=[],
    help="Dataset keys to skip",
)
@click.option(
    "--keep_datasets",
    type=str,
    multiple=True,
    default=[],
    help="Dataset keys to keep (exclusive with skip_datasets)",
)
@click.option(
    "--hidden_dim",
    type=int,
    default=512,
    help="Hidden layer dimension",
)
@click.option(
    "--dropout",
    type=float,
    default=0.25,
    help="Dropout rate (0.25 matches original experiment scripts)",
)
@click.option(
    "--learning_rate",
    type=float,
    default=0.001,
    help="Learning rate",
)
@click.option(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size for training",
)
@click.option(
    "--min_epochs",
    type=int,
    default=250,
    help="Minimum epochs before early stopping",
)
@click.option(
    "--max_epochs",
    type=int,
    default=500,
    help="Maximum epochs",
)
@click.option(
    "--patience",
    type=int,
    default=100,
    help="Early stopping patience (epochs without improvement)",
)
@click.option("--split_mode", type=click.Choice(["fov", "patch"]), default="fov",
              help="Split strategy: 'fov' (default, no spatial leakage) or 'patch' (cell-level random)")
@click.option("--split_file", type=str, default=None,
              help="Path to pre-computed FOV split JSON (overrides split_mode/seed for splitting)")
@click.option("--features_cache", type=str, default=None,
              help="Path to cache extracted features (.npz). Reuses cache if it exists.")
@click.option("--min_channels", type=int, default=3, help="Min non-DAPI channels per dataset (filters 2-channel datasets)")
def main(
    model_name: str,
    device_num: str,
    enable_wandb: bool,
    zarr_dir: str,
    skip_datasets: Tuple[str, ...],
    keep_datasets: Tuple[str, ...],
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    min_epochs: int,
    max_epochs: int,
    patience: int,
    split_mode: str,
    split_file: str,
    features_cache: str,
    min_channels: int,
):
    """Train MAPS baseline for cell type classification."""
    # Set device
    device = torch.device(device_num if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb if enabled
    if enable_wandb:
        import wandb
        wandb.login()
        run = wandb.init(
            project="deepcelltypes-temp-train",
            dir="wandb_tmp",
            job_type="train",
            name=f"{model_name}_maps",
            config={
                "model_type": "maps",
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "min_epochs": min_epochs,
                "max_epochs": max_epochs,
                "patience": patience,
                "split_mode": split_mode,
            },
        )

    # Load config
    dct_config = TissueNetConfig(zarr_dir)
    num_classes = dct_config.NUM_CELLTYPES
    input_dim = dct_config.NUM_MARKERS  # 271 features (globally aligned mean intensity per marker)

    print(f"Loading data from {zarr_dir}")
    print(f"Number of cell types: {num_classes}")
    print(f"Input features: {input_dim} (mean intensity per channel)")

    # Convert to lists (click returns tuples)
    skip_datasets = list(skip_datasets) if skip_datasets else None
    keep_datasets = list(keep_datasets) if keep_datasets else None

    if split_file is None:
        raise click.UsageError("--split_file is required. Generate one with: python -m scripts.generate_splits")

    # Extract features directly from zarr (fast path, no DataLoader overhead)
    print("\nExtracting features from zarr...")
    data = extract_features_from_zarr(
        zarr_dir=zarr_dir,
        dct_config=dct_config,
        split_file=split_file,
        skip_datasets=skip_datasets,
        keep_datasets=keep_datasets,
        cache_path=features_cache,
        min_channels=min_channels,
    )

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_val"], data["y_val"]
    test_dataset_names = data["val_dataset_names"]
    test_fov_names = data["val_fov_names"]
    test_cell_indices = data["val_cell_indices"]

    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # Remap labels to contiguous 0-indexed (required for CrossEntropyLoss).
    # Train labels must be contiguous [0..n_train-1]; test-only labels appended after.
    train_unique = np.sort(np.unique(y_train))
    label_to_compact = {orig: i for i, orig in enumerate(train_unique)}
    next_idx = len(train_unique)
    for label in np.sort(np.unique(y_test)):
        if label not in label_to_compact:
            label_to_compact[label] = next_idx
            next_idx += 1
    compact_to_label = {i: orig for orig, i in label_to_compact.items()}
    n_classes_compact = next_idx
    compact_ct2idx = {
        name: label_to_compact[idx]
        for name, idx in dct_config.ct2idx.items()
        if idx in label_to_compact
    }
    y_train = np.array([label_to_compact[y] for y in y_train])
    y_test = np.array([label_to_compact[y] for y in y_test])
    print(f"Unique cell types in data: {n_classes_compact} (of {num_classes} total)")

    # Z-score normalization (compute stats from training set only)
    print("\nNormalizing features (Z-score)...")
    X_train_norm, train_mean, train_std = normalize_features(X_train)
    X_test_norm, _, _ = normalize_features(X_test, mean=train_mean, std=train_std)

    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train_norm.astype(np.float32))
    y_train_tensor = torch.from_numpy(y_train.astype(np.int64))
    X_test_tensor = torch.from_numpy(X_test_norm.astype(np.float32))

    # Compute class weights for WeightedRandomSampler
    print("Computing class weights for balanced sampling...")
    sample_weights = compute_class_weights(y_train)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Create training dataloader with balanced sampling
    # drop_last=True matches original MAPS implementation
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=0,  # Features already in memory
    )

    # Create model
    model = MAPSModel(
        input_dim=input_dim,
        num_classes=n_classes_compact,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    print(f"\nModel architecture:")
    print(f"  Input: {input_dim} -> Hidden: {hidden_dim} -> Output: {n_classes_compact}")
    print(f"  Dropout: {dropout}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=learning_rate * 0.01)

    # Training loop with early stopping
    print(f"\nTraining MAPS model (min {min_epochs}, max {max_epochs} epochs, patience {patience})...")
    print(f"  LR schedule: cosine annealing {learning_rate} -> {learning_rate * 0.01}")
    best_val_loss = float("inf")
    best_macro_acc = 0.0
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(max_epochs):
        # Train
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        scheduler.step()

        # Evaluate
        y_true, y_pred, y_prob = evaluate(model, X_test_tensor, y_test, device)
        metrics = compute_baseline_metrics(
            y_true, y_pred, y_prob, n_classes_compact,
            hierarchy=CELL_TYPE_HIERARCHY, ct2idx=compact_ct2idx,
        )

        # Compute validation loss
        model.eval()
        with torch.no_grad():
            X_test_dev = X_test_tensor.to(device)
            y_test_dev = torch.from_numpy(y_test.astype(np.int64)).to(device)
            val_logits, _ = model(X_test_dev)
            val_loss = criterion(val_logits, y_test_dev).item()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:4d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Macro Acc={metrics['macro_accuracy']:.4f}, Weighted Acc={metrics['weighted_accuracy']:.4f}")

        # Log to wandb
        if enable_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "test/macro_accuracy": metrics["macro_accuracy"],
                "test/weighted_accuracy": metrics["weighted_accuracy"],
            })

        # Early stopping check — select on macro accuracy (consistent with transformer/CellSighter)
        macro_acc = metrics["macro_accuracy"]
        if macro_acc > best_macro_acc:
            best_macro_acc = macro_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            # Save best model
            model_path = Path(f"models/maps_{model_name}.pth")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_mean": train_mean,
                "train_std": train_std,
                "macro_accuracy": best_macro_acc,
                "val_loss": best_val_loss,
            }, model_path)
        else:
            epochs_without_improvement += 1

        # Check early stopping (only after min_epochs)
        if epoch >= min_epochs - 1 and epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            print(f"Best epoch: {best_epoch}, Best Macro Acc: {best_macro_acc:.4f}, Best Val Loss: {best_val_loss:.4f}")
            break

    # Load best model for final evaluation
    print(f"\nLoading best model from epoch {best_epoch}...")
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final evaluation
    print("\nFinal evaluation on test set...")
    y_true_compact, y_pred_compact, y_prob_compact = evaluate(model, X_test_tensor, y_test, device)
    metrics = compute_baseline_metrics(
        y_true_compact, y_pred_compact, y_prob_compact, n_classes_compact,
        hierarchy=CELL_TYPE_HIERARCHY, ct2idx=compact_ct2idx,
    )

    print(f"\nFinal Test Results:")
    print(f"  Macro Accuracy: {metrics['macro_accuracy']:.4f}")
    print(f"  Weighted Accuracy: {metrics['weighted_accuracy']:.4f}")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")

    # Log final metrics to wandb
    if enable_wandb:
        wandb.log({
            "final/macro_accuracy": metrics["macro_accuracy"],
            "final/weighted_accuracy": metrics["weighted_accuracy"],
            "final/best_epoch": best_epoch,
            "final/best_val_loss": best_val_loss,
        })

    # Map probabilities to ct2idx-sorted columns for saving
    # save_baseline_predictions expects y_prob with len(ct2idx) columns,
    # one per cell type sorted by ct2idx value
    y_true_orig = np.array([compact_to_label[y] for y in y_true_compact])
    sorted_ct_values = sorted(dct_config.ct2idx.values())
    ct_value_to_col = {v: i for i, v in enumerate(sorted_ct_values)}
    n_model_classes = y_prob_compact.shape[1]
    y_prob_orig = np.zeros((len(y_true_compact), len(dct_config.ct2idx)), dtype=np.float32)
    for compact_idx, orig_idx in compact_to_label.items():
        if compact_idx < n_model_classes and orig_idx in ct_value_to_col:
            y_prob_orig[:, ct_value_to_col[orig_idx]] = y_prob_compact[:, compact_idx]

    # Save predictions
    output_path = Path(f"output/{model_name}_maps_prediction.csv")
    save_baseline_predictions(
        y_true_orig,
        y_prob_orig,
        test_cell_indices,
        test_dataset_names,
        test_fov_names,
        dct_config.ct2idx,
        output_path,
    )

    # Save normalization stats for inference
    stats_path = Path(f"models/maps_{model_name}_stats.npz")
    np.savez(stats_path, mean=train_mean, std=train_std)
    print(f"Normalization stats saved to {stats_path}")

    if enable_wandb:
        run.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()
