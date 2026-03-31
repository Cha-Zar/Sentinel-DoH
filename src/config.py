"""Project-wide constants and default hyperparameters."""

from __future__ import annotations

from pathlib import Path


# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Reproducibility / splitting
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20


# Dataset columns
LABEL_COL = "Label"
TIMESTAMP_COL = "TimeStamp"

META_COLS = [
	"SourceIP",
	"DestinationIP",
	"SourcePort",
	"DestinationPort",
]

FLOW_FEATURES = [
	"Duration",
	"FlowBytesSent",
	"FlowSentRate",
	"FlowBytesReceived",
	"FlowReceivedRate",
]

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

NUMERIC_FEATURES = (
	FLOW_FEATURES
	+ PACKET_LENGTH_FEATURES
	+ PACKET_TIME_FEATURES
	+ RESPONSE_TIME_FEATURES
)


# Class imbalance handling
SMOTE_PARAMS = {
	"sampling_strategy": "auto",
	"k_neighbors": 5,
	"random_state": RANDOM_STATE,
}


# XGBoost defaults
XGB_PARAMS = {
	"n_estimators": 400,
	"max_depth": 6,
	"learning_rate": 0.05,
	"subsample": 0.9,
	"colsample_bytree": 0.9,
	"objective": "binary:logistic",
	"eval_metric": "logloss",
	"random_state": RANDOM_STATE,
	"tree_method": "hist",
	"n_jobs": -1,
}


# CNN defaults
CNN_PARAMS = {
	"conv1_filters": 64,
	"conv2_filters": 128,
	"dense_units": 64,
	"dropout_rate": 0.30,
	"learning_rate": 1e-3,
	"batch_size": 256,
	"epochs": 30,
	"patience": 6,
}


# Robustness sweep settings
JITTER_SIGMAS = [0.05, 0.10, 0.20]
PADDING_DELTAS = [0.10]


# Interpretability settings
SHAP_SAMPLE_SIZE = 2000
SHAP_MAX_DISPLAY = 20

