"""Metrics and threshold selection shared by train.py and explain.py."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, SEGMENTS, TEST_SIZE, processed_path


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "AUC": roc_auc_score(y_true, y_proba),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
    }


def find_threshold(y_true, y_proba) -> float:
    """Threshold with the largest geometric mean of TPR and 1 - FPR on the ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    gmeans = np.sqrt(tpr * (1 - fpr))
    return float(thresholds[np.argmax(gmeans)])


def scale_pos_weight(y) -> float:
    y = np.asarray(y)
    return float((y == 0).sum() / (y == 1).sum())


def load_split(segment: str):
    """Read the processed Parquet file of a segment and return a stratified train/test split."""
    df = pd.read_parquet(processed_path(segment)).drop(columns=["id"], errors="ignore")
    if "auto_payment" in df.columns:
        df["auto_payment"] = df["auto_payment"].astype(np.int32)
    X = df.drop(columns=["churn"])
    y = df["churn"]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def segment_names():
    return list(SEGMENTS)
