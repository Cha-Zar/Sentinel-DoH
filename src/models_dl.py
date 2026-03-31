"""PyTorch 1D-CNN model, training loop, and prediction helpers."""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import CNN_PARAMS, OUTPUT_DIR
from src.preprocessing import SplitData

logger = logging.getLogger("sentinel_doh")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentinelCNN1D(nn.Module):
    """Simple 1D-CNN for tabular feature sequences."""

    def __init__(self, n_features: int):
        super().__init__()
        # First conv block.
        self.conv1 = nn.Conv1d(1, CNN_PARAMS["conv1_filters"], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(CNN_PARAMS["conv1_filters"])
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)

        # Second conv block with global average pooling.
        self.conv2 = nn.Conv1d(CNN_PARAMS["conv1_filters"], CNN_PARAMS["conv2_filters"], kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classification head.
        self.fc1 = nn.Linear(CNN_PARAMS["conv2_filters"], CNN_PARAMS["dense_units"])
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(CNN_PARAMS["dropout_rate"])
        self.fc2 = nn.Linear(CNN_PARAMS["dense_units"], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_features)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        self._conv2_out = x  # Cached for Grad-CAM.

        x = self.gap(x)  # (batch, 128, 1)
        x = x.squeeze(-1)  # (batch, 128)

        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, 1) raw logit
        return x


def _compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weight for BCEWithLogitsLoss pos_weight."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info("CNN class weights: %s", cw)
    # pos_weight for BCEWithLogitsLoss.
    pos_weight = cw.get(1, 1.0) / cw.get(0, 1.0)
    return torch.tensor([pos_weight], dtype=torch.float32)


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Create a PyTorch DataLoader from numpy arrays."""
    # Conv1D input shape is (batch, channels, length).
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, n = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        n += X_batch.size(0)
    return total_loss / n


@torch.no_grad()
def _eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        n += X_batch.size(0)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        all_probs.extend(probs)
        all_labels.extend(y_batch.cpu().numpy().ravel())
    avg_loss = total_loss / n
    all_probs_arr = np.array(all_probs)
    all_labels_arr = np.array(all_labels)
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels_arr, all_probs_arr)
    except ValueError:
        auc = 0.0
    return avg_loss, auc, all_probs_arr


@torch.no_grad()
def _predict(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    """Return P(malicious) for each sample."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    probs = []
    for (batch,) in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
    return np.concatenate(probs)


def build_cnn(data: SplitData) -> Tuple[SentinelCNN1D, Dict]:
    """
    Build, train the 1D-CNN and return the model + results dict.

    Returns
    -------
    model : SentinelCNN1D
    results : dict
        y_prob_val, y_pred_val, y_prob_test, y_pred_test, history
    """
    n_features = data.X_train.shape[1]
    logger.info("CNN input: %d features, device=%s", n_features, DEVICE)

    model = SentinelCNN1D(n_features).to(DEVICE)
    logger.info("Model architecture:\n%s", model)

    # Weighted BCE for class imbalance.
    pos_weight = _compute_pos_weight(data.y_train).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_PARAMS["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6,
    )

    # Data loaders.
    bs = CNN_PARAMS["batch_size"]
    train_loader = _make_loader(data.X_train, data.y_train, bs, shuffle=True)
    val_loader = _make_loader(data.X_val, data.y_val, bs, shuffle=False)

    # Training loop with early stopping.
    epochs = CNN_PARAMS["epochs"]
    patience = CNN_PARAMS["patience"]
    best_auc = -1.0
    wait = 0
    best_state = None
    history: Dict[str, List[float]] = {
        "loss": [], "val_loss": [], "auc_roc": [], "val_auc_roc": [],
    }

    logger.info("Training 1D-CNN (epochs=%d, batch=%d) …", epochs, bs)

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_auc, _ = _eval_epoch(model, val_loader, criterion)

        # Track train AUC for the learning curves.
        _, train_auc, _ = _eval_epoch(model, train_loader, criterion)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["auc_roc"].append(train_auc)
        history["val_auc_roc"].append(val_auc)

        scheduler.step(val_auc)

        logger.info(
            "  Epoch %3d/%d — loss=%.4f  val_loss=%.4f  val_auc=%.4f  lr=%.2e",
            epoch, epochs, train_loss, val_loss, val_auc,
            optimizer.param_groups[0]["lr"],
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("  Early stopping at epoch %d (best val_auc=%.4f)", epoch, best_auc)
                break

    # Restore best checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final predictions.
    y_prob_val = _predict(model, data.X_val, bs)
    y_prob_test = _predict(model, data.X_test, bs)
    y_pred_val = (y_prob_val >= 0.5).astype(int)
    y_pred_test = (y_prob_test >= 0.5).astype(int)

    # Save model checkpoint.
    model_path = OUTPUT_DIR / "cnn_model.pt"
    torch.save(model.state_dict(), str(model_path))
    logger.info("CNN model saved → %s", model_path)

    results = {
        "y_prob_val": y_prob_val,
        "y_pred_val": y_pred_val,
        "y_prob_test": y_prob_test,
        "y_pred_test": y_pred_test,
        "history": history,
    }
    return model, results
