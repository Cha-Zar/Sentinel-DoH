"""
SENTINEL-DoH — XGBoost Baseline (ML Model)
============================================
Section III.1 of the specification.

* Statistical feature vector as input.
* ``scale_pos_weight`` computed from training-set class ratio.
* SMOTE applied upstream (preprocessing module).
* Returns trained model + predictions for evaluation.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from xgboost import XGBClassifier

from src.config import XGB_PARAMS, OUTPUT_DIR
from src.preprocessing import SplitData

logger = logging.getLogger("sentinel_doh")


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    """Ratio of negative to positive samples — XGBoost's native imbalance handler."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        logger.warning("No positive samples — scale_pos_weight set to 1.0")
        return 1.0
    weight = n_neg / n_pos
    logger.info("scale_pos_weight = %.2f  (neg=%d, pos=%d)", weight, n_neg, n_pos)
    return float(weight)


def build_xgboost(data: SplitData) -> Tuple[XGBClassifier, Dict]:
    """
    Train an XGBoost classifier on the preprocessed data.

    Returns
    -------
    model : XGBClassifier
        Fitted model.
    results : dict
        Contains probability predictions on val/test and training history.
    """
    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = _compute_scale_pos_weight(data.y_train)

    model = XGBClassifier(**params)

    logger.info("Training XGBoost (%d estimators, depth=%d) …",
                params["n_estimators"], params["max_depth"])

    model.fit(
        data.X_train,
        data.y_train,
        eval_set=[(data.X_val, data.y_val)],
        verbose=False,
    )

    # ── Predictions ──────────────────────────────────────────────────────
    y_prob_val = model.predict_proba(data.X_val)[:, 1]
    y_prob_test = model.predict_proba(data.X_test)[:, 1]
    y_pred_val = (y_prob_val >= 0.5).astype(int)
    y_pred_test = (y_prob_test >= 0.5).astype(int)

    # ── Feature importance (gain-based) ──────────────────────────────────
    importance = dict(
        zip(data.feature_names, model.feature_importances_)
    )
    top_5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("Top-5 features (gain): %s", top_5)

    # ── Save model ───────────────────────────────────────────────────────
    model_path = OUTPUT_DIR / "xgboost_model.json"
    model.save_model(str(model_path))
    logger.info("XGBoost model saved → %s", model_path)

    results = {
        "y_prob_val": y_prob_val,
        "y_pred_val": y_pred_val,
        "y_prob_test": y_prob_test,
        "y_pred_test": y_pred_test,
        "feature_importance": importance,
    }
    return model, results
