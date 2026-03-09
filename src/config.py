"""
SENTINEL-DoH — Central Configuration
=====================================
All hyper-parameters, paths, feature lists, and constants live here so that
every other module stays free of magic numbers.
"""

from pathlib import Path
import os

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Dataset — CIRA-CIC-DoHBrw-2020 feature schema
# =============================================================================
# Metadata columns (dropped before training)
META_COLS = [
    "SourceIP",
    "DestinationIP",
    "SourcePort",
    "DestinationPort",
]

# Timestamp column (kept for temporal-split, dropped before training)
TIMESTAMP_COL = "TimeStamp"

# Label column produced by DoHLyzer (Layer 1 — DoH vs Non-DoH)
DOH_LABEL_COL = "DoH"

# Label column for Layer 2 classification (Benign vs Malicious)
# This may be named "Label" in the pre-processed CSV; we normalise below.
LABEL_COL = "Label"

# Flow-level features
FLOW_FEATURES = [
    "Duration",
    "FlowBytesSent",
    "FlowSentRate",
    "FlowBytesReceived",
    "FlowReceivedRate",
]

# Packet-length statistical features (spatial dimension)
PACKET_LENGTH_FEATURES = [
    "PacketLengthVariance",
    "PacketLengthStandardDeviation",
    "PacketLengthMean",
    "PacketLengthMedian",
    "PacketLengthMode",
    "PacketLengthSkewFromMedian",
    "PacketLengthSkewFromMode",
    "PacketLengthCoefficientofVariation",
]

# Packet-time statistical features (temporal dimension — IAT)
PACKET_TIME_FEATURES = [
    "PacketTimeVariance",
    "PacketTimeStandardDeviation",
    "PacketTimeMean",
    "PacketTimeMedian",
    "PacketTimeMode",
    "PacketTimeSkewFromMedian",
    "PacketTimeSkewFromMode",
    "PacketTimeCoefficientofVariation",
]

# Response-time statistical features
RESPONSE_TIME_FEATURES = [
    "ResponseTimeTimeVariance",
    "ResponseTimeTimeStandardDeviation",
    "ResponseTimeTimeMean",
    "ResponseTimeTimeMedian",
    "ResponseTimeTimeMode",
    "ResponseTimeTimeSkewFromMedian",
    "ResponseTimeTimeSkewFromMode",
    "ResponseTimeTimeCoefficientofVariation",
]

# All numeric features used for training (order matters for CNN reshaping)
NUMERIC_FEATURES = (
    FLOW_FEATURES
    + PACKET_LENGTH_FEATURES
    + PACKET_TIME_FEATURES
    + RESPONSE_TIME_FEATURES
)

# Feature groups for CNN channel-based reshaping
# Each group becomes one "channel" in the Conv1D input
FEATURE_GROUPS = {
    "flow": FLOW_FEATURES,
    "packet_length": PACKET_LENGTH_FEATURES,
    "packet_time": PACKET_TIME_FEATURES,
    "response_time": RESPONSE_TIME_FEATURES,
}

# =============================================================================
# Splitting Strategy
# =============================================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
RANDOM_STATE = 42

# =============================================================================
# XGBoost Hyper-parameters (Section III.1)
# =============================================================================
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    # scale_pos_weight is computed dynamically in models_ml.py
}

# =============================================================================
# 1D-CNN Hyper-parameters (Section III.2)
# =============================================================================
CNN_PARAMS = {
    "conv1_filters": 64,
    "conv2_filters": 128,
    "dense_units": 64,
    "dropout_rate": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 50,
    "patience": 7,  # early-stopping patience
}

# =============================================================================
# SMOTE Parameters
# =============================================================================
SMOTE_PARAMS = {
    "sampling_strategy": "auto",  # resample minority to match majority
    "k_neighbors": 5,
    "random_state": RANDOM_STATE,
}

# =============================================================================
# Robustness — Adversarial Perturbation Budgets (Section V.2)
# =============================================================================
JITTER_SIGMAS = [0.05, 0.10, 0.20]       # 5%, 10%, 20% Gaussian noise on IAT
PADDING_DELTAS = [0.10]                    # ±10% on packet sizes

# =============================================================================
# Misc
# =============================================================================
# Number of SHAP samples for TreeExplainer beeswarm / force plots
SHAP_MAX_DISPLAY = 20
SHAP_SAMPLE_SIZE = 500

# Dataset download URL (informational — user must download manually)
DATASET_URL = "https://www.unb.ca/cic/datasets/dohbrw-2020.html"
