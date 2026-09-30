"""Per-segment settings for the churn models."""

import os

DATA_DIR = os.environ.get("TT_DATA", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Columns dropped, filled and treated as actionable in each service segment.
# Thresholds are the values found on the out-of-fold predictions by train.py.
SEGMENTS = {
    "broadband": {
        "file": "Broadband.parquet",
        "drop": ["avg_call_duration", "call_drops", "roaming_usage"],
        "fill_mode": ["auto_payment"],
        "fill_median": ["monthly_charge", "data_usage"],
        "actionable": ["apps_count", "satisfaction_score", "data_usage", "monthly_charge", "customer_support_calls"],
        "threshold": 0.4901,
    },
    "postpaid": {
        "file": "Postpaid.parquet",
        "drop": ["avg_top_up_count"],
        "fill_mode": ["auto_payment"],
        "fill_median": ["avg_call_duration", "data_usage", "monthly_charge"],
        "actionable": [
            "apps_count",
            "satisfaction_score",
            "data_usage",
            "monthly_charge",
            "customer_support_calls",
            "avg_call_duration",
        ],
        "threshold": 0.5428,
    },
    "prepaid": {
        "file": "Prepaid.parquet",
        "drop": ["auto_payment", "overdue_payments"],
        "fill_mode": [],
        "fill_median": ["avg_call_duration", "data_usage", "monthly_charge"],
        "actionable": [
            "apps_count",
            "satisfaction_score",
            "data_usage",
            "monthly_charge",
            "customer_support_calls",
            "avg_call_duration",
        ],
        "threshold": 0.5414,
    },
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42,
    "n_estimators": 300,
    "learning_rate": 0.02,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "gamma": 0.3,
    "reg_alpha": 0.3,
    "reg_lambda": 1.5,
}

TEST_SIZE = 0.15
RANDOM_STATE = 42


def processed_path(segment: str) -> str:
    return os.path.join(DATA_DIR, "processed", SEGMENTS[segment]["file"])


def results_dir(segment: str) -> str:
    path = os.path.join(RESULTS_DIR, segment)
    os.makedirs(path, exist_ok=True)
    return path
