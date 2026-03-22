#!/usr/bin/env python3
"""
SENTINEL-DoH — Main Pipeline Orchestrator
===========================================
Behavioural Classification of DNS-over-HTTPS Traffic
via Hybrid ML/DL Spatio-Temporal Analysis.

Usage
-----
    python main.py                    # Full pipeline (random split)
    python main.py --chrono           # Include temporal-drift analysis
    python main.py --skip-dl          # Only train XGBoost (faster)
    python main.py --data-dir ./data  # Custom data directory
    python main.py --help

Pipeline Steps
--------------
1. Data loading & preprocessing  (CIRA-CIC-DoHBrw-2020)
2. XGBoost training & evaluation
3. 1D-CNN training & evaluation
4. Comparative evaluation (confusion matrices, ROC/PR, metrics JSON)
5. Robustness tests (adversarial jitter + padding)
6. Interpretability (SHAP + Grad-CAM 1D)
7. (Optional) Temporal drift analysis with chronological split

All outputs are saved to the ``outputs/`` directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ── Project imports ──────────────────────────────────────────────────────────
from src.utils import setup_logging, set_seeds
from src.config import OUTPUT_DIR, RANDOM_STATE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SENTINEL-DoH — DoH Traffic Classification Pipeline",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to directory containing CIRA-CIC-DoHBrw-2020 CSV files (default: ./data)",
    )
    p.add_argument(
        "--chrono",
        action="store_true",
        help="Run temporal drift analysis with chronological split",
    )
    p.add_argument(
        "--skip-dl",
        action="store_true",
        help="Skip 1D-CNN training (ML-only mode)",
    )
    p.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip adversarial robustness tests",
    )
    p.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP explainability (faster)",
    )
    p.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging("INFO")
    set_seeds(RANDOM_STATE)

    t0 = time.time()
    logger.info("=" * 70)
    logger.info("  SENTINEL-DoH  —  Pipeline Start")
    logger.info("=" * 70)

    all_metrics: list[dict] = []

    # ================================================================
    # 1. PREPROCESSING
    # ================================================================
    logger.info("\n▸ STEP 1 — Data Loading & Preprocessing")
    from src.preprocessing import run_preprocessing

    data = run_preprocessing(
        data_dir=args.data_dir,
        chronological=False,
        apply_smote=not args.no_smote,
    )
    logger.info(
        "  Train: %d  |  Val: %d  |  Test: %d  |  Features: %d",
        len(data.y_train), len(data.y_val), len(data.y_test),
        len(data.feature_names),
    )

    # ================================================================
    # 2. XGBOOST (ML BASELINE)
    # ================================================================
    logger.info("\n▸ STEP 2 — XGBoost Training")
    from src.models_ml import build_xgboost
    from src.evaluation import compute_metrics, analyse_high_confidence_failures

    xgb_model, xgb_results = build_xgboost(data)
    metrics_xgb = compute_metrics(
        data.y_test, xgb_results["y_pred_test"], xgb_results["y_prob_test"],
        model_name="XGBoost",
    )
    all_metrics.append(metrics_xgb)
    hcf_xgb = analyse_high_confidence_failures(
        data.y_test, xgb_results["y_prob_test"], model_name="XGBoost",
    )

    # ================================================================
    # 3. 1D-CNN (DEEP LEARNING)
    # ================================================================
    cnn_model = None
    cnn_results = None
    if not args.skip_dl:
        logger.info("\n▸ STEP 3 — 1D-CNN Training")
        from src.models_dl import build_cnn

        cnn_model, cnn_results = build_cnn(data)
        metrics_cnn = compute_metrics(
            data.y_test, cnn_results["y_pred_test"], cnn_results["y_prob_test"],
            model_name="1D-CNN",
        )
        all_metrics.append(metrics_cnn)
        hcf_cnn = analyse_high_confidence_failures(
            data.y_test, cnn_results["y_prob_test"], model_name="1D-CNN",
        )

        # Training history plot
        from src.evaluation import plot_training_history
        plot_training_history(cnn_results["history"])
    else:
        logger.info("\n▸ STEP 3 — Skipped (--skip-dl)")

    # ================================================================
    # 4. COMPARATIVE EVALUATION
    # ================================================================
    logger.info("\n▸ STEP 4 — Comparative Evaluation")
    from src.evaluation import (
        plot_confusion_matrices,
        plot_roc_pr_curves,
        plot_confidence_distribution,
        save_metrics_json,
    )

    if cnn_results is not None:
        plot_confusion_matrices(
            data.y_test,
            xgb_results["y_pred_test"],
            cnn_results["y_pred_test"],
        )
        plot_roc_pr_curves(
            data.y_test,
            xgb_results["y_prob_test"],
            cnn_results["y_prob_test"],
        )
    else:
        # Single-model mode — plot XGBoost confusion matrix only
        from sklearn.metrics import confusion_matrix as cm_fn
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = cm_fn(data.y_test, xgb_results["y_pred_test"])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("XGBoost — Confusion Matrix")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "confusion_matrix_xgboost.png")
        plt.close()

    plot_confidence_distribution(
        data.y_test, xgb_results["y_prob_test"], model_name="XGBoost",
    )
    if cnn_results is not None:
        plot_confidence_distribution(
            data.y_test, cnn_results["y_prob_test"], model_name="1D-CNN",
        )

    save_metrics_json(all_metrics)

    # ================================================================
    # 5. ROBUSTNESS / ADVERSARIAL TESTS
    # ================================================================
    if not args.skip_robustness:
        logger.info("\n▸ STEP 5 — Robustness & Adversarial Tests")
        from src.robustness import adversarial_sweep, plot_robustness_summary

        # XGBoost predict function
        def xgb_predict(X: "np.ndarray") -> "np.ndarray":
            return xgb_model.predict_proba(X)[:, 1]

        adv_xgb = adversarial_sweep(
            "XGBoost", xgb_predict,
            data.X_test, data.y_test, data.feature_names,
        )

        adv_cnn = {}
        if cnn_model is not None:
            from src.models_dl import _predict as cnn_predict_fn

            def cnn_predict(X: "np.ndarray") -> "np.ndarray":
                return cnn_predict_fn(cnn_model, X, batch_size=256)

            adv_cnn = adversarial_sweep(
                "1D-CNN", cnn_predict,
                data.X_test, data.y_test, data.feature_names,
            )

        if adv_cnn:
            plot_robustness_summary(adv_xgb, adv_cnn)

        # Save adversarial results
        adv_path = OUTPUT_DIR / "adversarial_results.json"
        with open(adv_path, "w") as f:
            json.dump({"xgboost": adv_xgb, "cnn": adv_cnn}, f, indent=2)
        logger.info("Adversarial results saved → %s", adv_path)
    else:
        logger.info("\n▸ STEP 5 — Skipped (--skip-robustness)")

    # ================================================================
    # 6. INTERPRETABILITY (SHAP + GRAD-CAM)
    # ================================================================
    logger.info("\n▸ STEP 6 — Interpretability")
    from src.interpretability import explain_xgboost, explain_cnn

    if not args.skip_shap:
        explain_xgboost(xgb_model, data.X_test, data.feature_names)
    else:
        logger.info("  SHAP skipped (--skip-shap)")

    if cnn_model is not None:
        explain_cnn(cnn_model, data.X_test, data.feature_names)

    # ================================================================
    # 7. TEMPORAL DRIFT ANALYSIS (Optional)
    # ================================================================
    if args.chrono:
        logger.info("\n▸ STEP 7 — Temporal Drift Analysis")
        from src.preprocessing import run_preprocessing as prep
        from src.robustness import compare_temporal_drift
        from sklearn.metrics import f1_score

        # Re-run with chronological split
        data_chrono = prep(
            data_dir=args.data_dir,
            chronological=True,
            apply_smote=not args.no_smote,
        )

        # Retrain XGBoost on chrono split
        from src.models_ml import build_xgboost as build_xgb_fn
        _, xgb_chrono_res = build_xgb_fn(data_chrono)
        f1_xgb_chrono = float(
            f1_score(data_chrono.y_test, xgb_chrono_res["y_pred_test"], average="macro")
        )
        drift_xgb = compare_temporal_drift(
            metrics_xgb["f1_macro"], f1_xgb_chrono, "XGBoost"
        )

        drift_results = [drift_xgb]

        if not args.skip_dl:
            from src.models_dl import build_cnn as build_cnn_fn
            _, cnn_chrono_res = build_cnn_fn(data_chrono)
            f1_cnn_chrono = float(
                f1_score(data_chrono.y_test, cnn_chrono_res["y_pred_test"], average="macro")
            )
            drift_cnn = compare_temporal_drift(
                metrics_cnn["f1_macro"], f1_cnn_chrono, "1D-CNN"
            )
            drift_results.append(drift_cnn)

        drift_path = OUTPUT_DIR / "temporal_drift.json"
        with open(drift_path, "w") as f:
            json.dump(drift_results, f, indent=2)
        logger.info("Temporal drift results saved → %s", drift_path)
    else:
        logger.info("\n▸ STEP 7 — Skipped (use --chrono to enable)")

    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("  SENTINEL-DoH  —  Pipeline Complete  (%.1f s)", elapsed)
    logger.info("=" * 70)
    logger.info("Outputs directory: %s", OUTPUT_DIR)
    logger.info("Files generated:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        if f.is_file():
            logger.info("  • %s", f.name)


if __name__ == "__main__":
    main()
