"""Metrics, plots, and reporting utilities for model evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")  # Headless backend for CLI and Docker runs.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import OUTPUT_DIR

logger = logging.getLogger("sentinel_doh")

# Keep figures consistent across runs.
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

TARGET_NAMES = ["Benign (0)", "Malicious (1)"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> Dict:
    """Compute a standard binary-classification metric bundle."""
    metrics = {
        "model": model_name,
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_class1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_class1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_class1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "auc_pr": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
    }

    logger.info(
        "\n%s  |  F1-macro=%.4f  AUC-ROC=%.4f  AUC-PR=%.4f\n%s",
        model_name,
        metrics["f1_macro"],
        metrics["auc_roc"],
        metrics["auc_pr"],
        classification_report(y_true, y_pred, target_names=TARGET_NAMES, zero_division=0),
    )
    return metrics


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred_ml: np.ndarray,
    y_pred_dl: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """Plot side-by-side confusion matrices for XGBoost and CNN."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y_pred, title in zip(
        axes,
        [y_pred_ml, y_pred_dl],
        ["XGBoost (ML)", "1D-CNN (DL)"],
    ):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=TARGET_NAMES,
            yticklabels=TARGET_NAMES,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.suptitle("Confusion Matrices - Test Set", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = save_path or OUTPUT_DIR / "confusion_matrices.png"
    plt.savefig(save_path)
    plt.close()
    logger.info("Confusion matrices saved to %s", save_path)


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob_ml: np.ndarray,
    y_prob_dl: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """Plot ROC and precision-recall curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC curves
    ax = axes[0]
    for y_prob, label, color in [
        (y_prob_ml, "XGBoost", "#2196F3"),
        (y_prob_dl, "1D-CNN", "#FF5722"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.4f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    # Precision-recall curves
    ax = axes[1]
    for y_prob, label, color in [
        (y_prob_ml, "XGBoost", "#2196F3"),
        (y_prob_dl, "1D-CNN", "#FF5722"),
    ]:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f"{label} (AP={ap:.4f})", color=color, lw=2)

    # Baseline is just positive class prevalence.
    pos_rate = y_true.mean()
    ax.axhline(
        y=pos_rate,
        color="grey",
        linestyle="--",
        lw=1,
        alpha=0.5,
        label=f"Baseline ({pos_rate:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")

    plt.suptitle("ROC and PR Curves - Test Set", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = save_path or OUTPUT_DIR / "roc_pr_curves.png"
    plt.savefig(save_path)
    plt.close()
    logger.info("ROC/PR curves saved to %s", save_path)


def plot_confidence_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> None:
    """Plot score histograms split by ground-truth class."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for cls, label, color in [(0, "Benign", "#4CAF50"), (1, "Malicious", "#F44336")]:
        mask = y_true == cls
        ax.hist(y_prob[mask], bins=50, alpha=0.6, label=label, color=color, density=True)
    ax.set_xlabel("Predicted Probability (malicious)")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name} - Confidence Score Distribution")
    ax.legend()
    plt.tight_layout()
    save_path = save_path or OUTPUT_DIR / f"confidence_dist_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path)
    plt.close()


def analyse_high_confidence_failures(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    threshold: float = 0.95,
) -> Dict:
    """Count high-confidence false positives and false negatives."""
    y_pred = (y_prob >= 0.5).astype(int)
    fp_mask = (y_pred == 1) & (y_true == 0)
    hc_fp_mask = fp_mask & (y_prob > threshold)

    fn_mask = (y_pred == 0) & (y_true == 1)
    hc_fn_mask = fn_mask & (y_prob < (1 - threshold))

    stats = {
        "model": model_name,
        "total_fp": int(fp_mask.sum()),
        "high_confidence_fp": int(hc_fp_mask.sum()),
        "total_fn": int(fn_mask.sum()),
        "high_confidence_fn": int(hc_fn_mask.sum()),
        "threshold": threshold,
    }
    logger.info(
        "%s confidence analysis: HC-FP=%d/%d, HC-FN=%d/%d (threshold=%.2f)",
        model_name,
        stats["high_confidence_fp"],
        stats["total_fp"],
        stats["high_confidence_fn"],
        stats["total_fn"],
        threshold,
    )
    return stats


def plot_training_history(
    history: Dict,
    save_path: Optional[Path] = None,
) -> None:
    """Plot CNN loss and AUC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(history["loss"], label="Train", lw=2)
    ax.plot(history["val_loss"], label="Val", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("Training Loss")
    ax.legend()

    ax = axes[1]
    if "auc_roc" in history:
        ax.plot(history["auc_roc"], label="Train AUC-ROC", lw=2)
        ax.plot(history["val_auc_roc"], label="Val AUC-ROC", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Training AUC-ROC")
    ax.legend()

    plt.suptitle("1D-CNN Training History", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = save_path or OUTPUT_DIR / "cnn_training_history.png"
    plt.savefig(save_path)
    plt.close()
    logger.info("Training history plot saved to %s", save_path)


def save_metrics_json(
    metrics_list: list[Dict],
    save_path: Optional[Path] = None,
) -> None:
    """Write all metrics to JSON."""
    save_path = save_path or OUTPUT_DIR / "metrics.json"
    with open(save_path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    logger.info("Metrics JSON saved to %s", save_path)
