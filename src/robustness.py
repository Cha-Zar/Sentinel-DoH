"""Robustness checks: temporal drift and simple adversarial perturbations."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import (
    FLOW_FEATURES,
    JITTER_SIGMAS,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    PACKET_LENGTH_FEATURES,
    PACKET_TIME_FEATURES,
    PADDING_DELTAS,
    RESPONSE_TIME_FEATURES,
)
from src.evaluation import compute_metrics

logger = logging.getLogger("sentinel_doh")


def _feature_indices(feature_names: List[str], target_features: List[str]) -> List[int]:
    """Return column indices of *target_features* within *feature_names*."""
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    return [name_to_idx[f] for f in target_features if f in name_to_idx]


def apply_jitter(
    X: np.ndarray,
    feature_names: List[str],
    sigma_frac: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add Gaussian noise to IAT (packet-time + response-time) features.

    Parameters
    ----------
    sigma_frac : float
        Fraction of feature standard deviation used as noise σ.
        E.g. 0.05 → 5 % jitter.
    """
    rng = rng or np.random.default_rng(42)
    X_adv = X.copy()
    time_features = PACKET_TIME_FEATURES + RESPONSE_TIME_FEATURES
    indices = _feature_indices(feature_names, time_features)

    for idx in indices:
        col = X_adv[:, idx]
        noise_std = np.std(col) * sigma_frac
        noise = rng.normal(0, noise_std, size=col.shape)
        X_adv[:, idx] = col + noise

    return X_adv


def apply_padding(
    X: np.ndarray,
    feature_names: List[str],
    delta_frac: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add uniform noise to packet-size features.

    Parameters
    ----------
    delta_frac : float
        Maximum perturbation fraction (e.g. 0.10 → ±10 %).
    """
    rng = rng or np.random.default_rng(42)
    X_adv = X.copy()
    size_features = PACKET_LENGTH_FEATURES + FLOW_FEATURES  # Flow bytes act like size features too.
    indices = _feature_indices(feature_names, size_features)

    for idx in indices:
        col = X_adv[:, idx]
        perturbation = rng.uniform(-delta_frac, delta_frac, size=col.shape) * np.abs(col)
        X_adv[:, idx] = col + perturbation

    return X_adv


def adversarial_sweep(
    model_name: str,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> Dict:
    """
    Run the full adversarial test battery on a model.

    Parameters
    ----------
    predict_fn : callable
        A function ``X → y_prob`` returning probability of class 1.

    Returns
    -------
    dict  with keys: "clean", "jitter_{σ}", "padding_{δ}"
    """
    results: Dict[str, Dict] = {}

    # Baseline on unmodified test data.
    y_prob_clean = predict_fn(X_test)
    y_pred_clean = (y_prob_clean >= 0.5).astype(int)
    metrics_clean = compute_metrics(y_test, y_pred_clean, y_prob_clean, f"{model_name} [clean]")
    results["clean"] = metrics_clean

    # Time-feature jitter sweep.
    for sigma in JITTER_SIGMAS:
        X_jit = apply_jitter(X_test, feature_names, sigma)
        y_prob = predict_fn(X_jit)
        y_pred = (y_prob >= 0.5).astype(int)
        label = f"{model_name} [jitter σ={sigma:.0%}]"
        results[f"jitter_{sigma}"] = compute_metrics(y_test, y_pred, y_prob, label)

    # Packet-size padding sweep.
    for delta in PADDING_DELTAS:
        X_pad = apply_padding(X_test, feature_names, delta)
        y_prob = predict_fn(X_pad)
        y_pred = (y_prob >= 0.5).astype(int)
        label = f"{model_name} [padding +/-{delta:.0%}]"
        results[f"padding_{delta}"] = compute_metrics(y_test, y_pred, y_prob, label)

    return results


def compare_temporal_drift(
    f1_random: float,
    f1_chrono: float,
    model_name: str,
) -> Dict:
    """Report F1 degradation from random split to chronological split."""
    drift = f1_random - f1_chrono
    pct = (drift / f1_random * 100) if f1_random > 0 else 0.0
    result = {
        "model": model_name,
        "f1_random_split": f1_random,
        "f1_chrono_split": f1_chrono,
        "f1_degradation": drift,
        "degradation_pct": pct,
    }
    logger.info(
        "Temporal drift [%s]: F1 random=%.4f -> chrono=%.4f  (delta=%.4f, -%.1f%%)",
        model_name, f1_random, f1_chrono, drift, pct,
    )
    return result


def plot_robustness_summary(
    adv_results_ml: Dict,
    adv_results_dl: Dict,
    save_path: Optional[Path] = None,
) -> None:
    """Bar chart comparing F1-macro under each adversarial condition."""
    conditions = sorted(set(adv_results_ml.keys()) | set(adv_results_dl.keys()))

    f1_ml = [adv_results_ml.get(c, {}).get("f1_macro", 0) for c in conditions]
    f1_dl = [adv_results_dl.get(c, {}).get("f1_macro", 0) for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, f1_ml, width, label="XGBoost", color="#2196F3", alpha=0.85)
    ax.bar(x + width / 2, f1_dl, width, label="1D-CNN", color="#FF5722", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel("F1-score (macro)")
    ax.set_title("Robustness — F1 Under Adversarial Perturbations")
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_path = save_path or OUTPUT_DIR / "robustness_summary.png"
    plt.savefig(save_path)
    plt.close()
    logger.info("Robustness summary plot saved → %s", save_path)
