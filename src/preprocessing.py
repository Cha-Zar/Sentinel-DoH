"""Data loading and preprocessing helpers for model training."""

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_DIR,
    LABEL_COL,
    META_COLS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    SMOTE_PARAMS,
    TEST_RATIO,
    TIMESTAMP_COL,
    TRAIN_RATIO,
    VAL_RATIO,
)

logger = logging.getLogger("sentinel_doh")


@dataclass
class SplitData:
    """Holds train / validation / test arrays after all preprocessing."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler
    timestamps_test: Optional[np.ndarray] = None  # Keep test timestamps for drift checks.


def _find_csv_files(data_dir: Path) -> list[Path]:
    """Recursively find all CSV files under *data_dir*.

    Prefers Layer-2 files (``l2-*.csv``) when the CIRA-CIC-DoHBrw-2020
    archive is extracted as-is.  If none are found, falls back to every CSV.
    """
    patterns = ["**/*.csv", "**/*.CSV"]
    found: list[Path] = []
    for pat in patterns:
        found.extend(data_dir.glob(pat))
    if not found:
        raise FileNotFoundError(
            f"No CSV files found under {data_dir}. "
            "Please download the CIRA-CIC-DoHBrw-2020 dataset and place the "
            "CSV files (e.g. l2-*.csv or per-tool CSVs) inside the data/ directory."
        )

    # Prefer L2 files when both L1 and L2 are present.
    l2_files = [f for f in found if f.stem.lower().startswith("l2")]
    if l2_files:
        logger.info(
            "Detected Layer-2 files — using only %d L2 CSV(s) (ignoring L1).",
            len(l2_files),
        )
        return sorted(set(l2_files))

    return sorted(set(found))


def _infer_label(filepath: Path, df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has a ``Label`` column with values 0 (Benign) / 1 (Malicious).

    The dataset ships in several layouts:
    1. A ``Label`` column already present (values: Benign / Malicious or 0/1).
    2. No ``Label`` column, but the file/folder name hints at the class
       (e.g. ``benign/Chrome.csv`` vs ``malicious/dns2tcp.csv``).
    """
    # Case 1: Label column already exists.
    if LABEL_COL in df.columns:
        if df[LABEL_COL].dtype == object or pd.api.types.is_string_dtype(df[LABEL_COL]):
            # Normalize common string forms to 0/1.
            df[LABEL_COL] = df[LABEL_COL].str.strip().str.lower().map(
                {"benign": 0, "malicious": 1, "0": 0, "1": 1}
            )
            unmapped = df[LABEL_COL].isna().sum()
            if unmapped:
                logger.warning("Dropped %d rows with unmapped Label values", unmapped)
                df = df.dropna(subset=[LABEL_COL])
        df[LABEL_COL] = df[LABEL_COL].astype(int)
        return df

    # Case 2: infer labels from path and filename.
    path_lower = str(filepath).lower()
    if "malicious" in path_lower or any(
        tool in path_lower for tool in ("dns2tcp", "dnscat", "iodine")
    ):
        df[LABEL_COL] = 1
    elif "benign" in path_lower or any(
        browser in path_lower for browser in ("chrome", "firefox")
    ):
        df[LABEL_COL] = 0
    else:
        raise ValueError(
            f"Cannot determine label for {filepath}. "
            "Please ensure CSV files contain a 'Label' column or are stored "
            "in 'benign/' / 'malicious/' subdirectories."
        )
    return df


def load_dataset(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and concatenate all CSV files from *data_dir*, returning a single
    DataFrame with a normalised ``Label`` column.
    """
    data_dir = data_dir or DATA_DIR
    csv_files = _find_csv_files(data_dir)
    logger.info("Found %d CSV file(s) in %s", len(csv_files), data_dir)

    frames: list[pd.DataFrame] = []
    for fp in csv_files:
        df = pd.read_csv(fp, low_memory=False)
        df = _infer_label(fp, df)
        # Keep track of the source tool/browser for bias analysis
        df["_source_file"] = fp.stem
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Combined dataset: %d samples  |  Label distribution:\n%s",
        len(combined),
        combined[LABEL_COL].value_counts().to_string(),
    )
    return combined


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-numeric metadata, handle NaN / Inf, and validate features."""
    # Drop metadata IP/port columns (not useful for classification)
    cols_to_drop = [c for c in META_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Some releases have minor column-name differences.
    available = [f for f in NUMERIC_FEATURES if f in df.columns]
    missing = set(NUMERIC_FEATURES) - set(available)
    if missing:
        logger.warning("Missing features (will be zero-filled): %s", missing)
        for col in missing:
            df[col] = 0.0

    # Replace inf and fill gaps with per-column medians.
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in NUMERIC_FEATURES:
        if df[col].isna().any():
            median_val = df[col].median()
            # Entirely empty columns default to 0.
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

    # Final safety pass before train/val/test split.
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    remaining_nan = df[NUMERIC_FEATURES].isna().any(axis=1).sum()
    if remaining_nan:
        logger.warning("Dropping %d rows with residual NaN values", remaining_nan)
        df = df.dropna(subset=NUMERIC_FEATURES)

    logger.info("Cleaning complete — %d samples, %d features retained", len(df), len(NUMERIC_FEATURES))
    return df


def split_stratified(
    df: pd.DataFrame,
    chronological: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / val / test with stratification.

    If *chronological* is True, the split respects temporal order (for drift
    analysis): the earliest 70 % → train, next 10 % → val, last 20 % → test.
    """
    if chronological and TIMESTAMP_COL in df.columns:
        logger.info("Using CHRONOLOGICAL split (temporal drift mode)")
        df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        n = len(df)
        tr_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        train_df = df.iloc[:tr_end]
        val_df = df.iloc[tr_end:val_end]
        test_df = df.iloc[val_end:]
    else:
        logger.info("Using STRATIFIED random split (70/10/20)")
        train_val_df, test_df = train_test_split(
            df, test_size=TEST_RATIO, stratify=df[LABEL_COL], random_state=RANDOM_STATE
        )
        relative_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val,
            stratify=train_val_df[LABEL_COL],
            random_state=RANDOM_STATE,
        )

    for name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = part[LABEL_COL].value_counts().to_dict()
        logger.info("  %s  — %d samples  |  %s", name, len(part), dist)

    return train_df, val_df, test_df


def prepare_arrays(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    apply_smote: bool = True,
) -> SplitData:
    """
    Convert DataFrames to numpy arrays, apply StandardScaler, and optionally
    apply SMOTE on the training split only.
    """
    feature_cols = [c for c in NUMERIC_FEATURES if c in train_df.columns]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[LABEL_COL].values.astype(np.int32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[LABEL_COL].values.astype(np.int32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[LABEL_COL].values.astype(np.int32)

    # Keep test timestamps when available.
    timestamps_test = None
    if TIMESTAMP_COL in test_df.columns:
        timestamps_test = test_df[TIMESTAMP_COL].values

    # Fit scaler on train only, then apply to val/test.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Apply SMOTE only on the training split.
    if apply_smote:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(**SMOTE_PARAMS)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(
            "SMOTE applied — Train set after resampling: %d samples  |  "
            "class 0: %d, class 1: %d",
            len(y_train),
            (y_train == 0).sum(),
            (y_train == 1).sum(),
        )

    return SplitData(
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.int32),
        X_val=X_val.astype(np.float32),
        y_val=y_val.astype(np.int32),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.int32),
        feature_names=feature_cols,
        scaler=scaler,
        timestamps_test=timestamps_test,
    )


def run_preprocessing(
    data_dir: Optional[Path] = None,
    chronological: bool = False,
    apply_smote: bool = True,
) -> SplitData:
    """End-to-end: load → clean → split → scale → SMOTE."""
    df = load_dataset(data_dir)
    df = clean(df)
    train_df, val_df, test_df = split_stratified(df, chronological=chronological)
    return prepare_arrays(train_df, val_df, test_df, apply_smote=apply_smote)
