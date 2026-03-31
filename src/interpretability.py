"""Model interpretability helpers (SHAP for XGBoost, Grad-CAM for CNN)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import OUTPUT_DIR, SHAP_MAX_DISPLAY, SHAP_SAMPLE_SIZE

logger = logging.getLogger("sentinel_doh")


def explain_xgboost(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    save_dir: Optional[Path] = None,
) -> None:
    """
    Generate SHAP explanations for the XGBoost model.

    Produces a SHAP summary plot, bar plot, and one force plot sample.
    """
    import shap

    save_dir = save_dir or OUTPUT_DIR
    logger.info("Computing SHAP values (TreeExplainer)...")

    explainer = shap.TreeExplainer(model)

    # Subsample test rows to keep runtime manageable.
    n_samples = min(SHAP_SAMPLE_SIZE, len(X_test))
    idx = np.random.default_rng(42).choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[idx]

    shap_values = explainer.shap_values(X_sample)

    # Summary beeswarm plot.
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=SHAP_MAX_DISPLAY,
        show=False,
    )
    path = save_dir / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved to %s", path)

    # Mean absolute SHAP values.
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=SHAP_MAX_DISPLAY,
        plot_type="bar",
        show=False,
    )
    path = save_dir / "shap_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP bar plot saved to %s", path)

    # Force plot for one high-confidence malicious sample.
    try:
        # Find a sample predicted as malicious with high confidence
        y_prob = model.predict_proba(X_sample)[:, 1]
        mal_idx = np.where(y_prob > 0.8)[0]
        if len(mal_idx) > 0:
            single = mal_idx[0]
        else:
            single = 0

        force_html = shap.force_plot(
            explainer.expected_value,
            shap_values[single],
            X_sample[single],
            feature_names=feature_names,
            matplotlib=False,
        )
        path = save_dir / "shap_force_plot.html"
        shap.save_html(str(path), force_html)
        logger.info("SHAP force plot saved to %s", path)
    except Exception as e:
        logger.warning("Could not generate SHAP force plot: %s", e)


def grad_cam_1d(
    model,
    X_sample: np.ndarray,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a PyTorch 1D-CNN.

    Parameters
    ----------
    model : SentinelCNN1D (PyTorch nn.Module)
    X_sample : np.ndarray  shape (1, n_features)

    Returns
    -------
    heatmap : np.ndarray  shape (seq_len,)
    """
    import torch

    model.eval()
    x = torch.tensor(X_sample, dtype=torch.float32).unsqueeze(1)  # (1, 1, n_features)
    x.requires_grad_(False)

    # Capture conv2 activations and gradients.
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["conv2"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["conv2"] = grad_out[0].detach()

    handle_fwd = model.conv2.register_forward_hook(fwd_hook)
    handle_bwd = model.conv2.register_full_backward_hook(bwd_hook)

    # Forward pass.
    x_input = x.clone().requires_grad_(True)
    logits = model(x_input)
    prob = torch.sigmoid(logits)

    # Backprop through positive class probability.
    model.zero_grad()
    prob.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    # Grad-CAM weights channels by mean gradient.
    conv_out = activations["conv2"][0]  # (filters, seq_len)
    grads_val = gradients["conv2"][0]  # (filters, seq_len)
    weights = grads_val.mean(dim=-1)  # (filters,)

    # Weighted combination across channels.
    heatmap = (weights.unsqueeze(-1) * conv_out).sum(dim=0).numpy()  # (seq_len,)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def explain_cnn(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    save_dir: Optional[Path] = None,
    n_samples: int = 5,
) -> None:
    """
    Generate Grad-CAM 1D explanations for the CNN model.

    Save Grad-CAM overlays showing which features drove each prediction.
    """
    save_dir = save_dir or OUTPUT_DIR

    # Score test samples and pick malicious/high-score examples.
    from src.models_dl import _predict, CNN_PARAMS

    y_prob = _predict(model, X_test, batch_size=CNN_PARAMS["batch_size"])
    mal_indices = np.where(y_prob > 0.8)[0]
    if len(mal_indices) == 0:
        mal_indices = np.argsort(y_prob)[-n_samples:]
    else:
        mal_indices = mal_indices[:n_samples]

    logger.info("Generating Grad-CAM for %d samples...", len(mal_indices))

    fig, axes = plt.subplots(len(mal_indices), 1, figsize=(14, 3 * len(mal_indices)))
    if len(mal_indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, mal_indices):
        sample = X_test[idx : idx + 1]
        heatmap = grad_cam_1d(model, sample)

        # Interpolate heatmap back to feature length after pooling.
        n_features = len(feature_names)
        if len(heatmap) != n_features:
            from scipy.interpolate import interp1d

            x_old = np.linspace(0, 1, len(heatmap))
            x_new = np.linspace(0, 1, n_features)
            interp_fn = interp1d(x_old, heatmap, kind="linear")
            heatmap = interp_fn(x_new)

        # Plot feature values colored by CAM intensity.
        colors = plt.cm.YlOrRd(heatmap)
        ax.bar(range(n_features), X_test[idx], color=colors, alpha=0.8)
        ax.set_xticks(range(n_features))
        ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
        ax.set_title(f"Sample {idx}  |  P(malicious)={y_prob[idx]:.3f}")
        ax.set_ylabel("Feature value (scaled)")

    plt.suptitle("Grad-CAM 1D — Feature Contribution Heatmap", fontsize=14, y=1.01)
    plt.tight_layout()
    path = save_dir / "gradcam_1d.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Grad-CAM 1D plot saved to %s", path)
