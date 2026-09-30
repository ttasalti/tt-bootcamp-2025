"""Train the churn model of one segment and write its evaluation to results/<segment>/.

    python churn/train.py --segment postpaid

Five-fold stratified cross-validation on the training part, a decision threshold chosen on
the out-of-fold predictions, then a final model on the full training part evaluated on the
15% hold-out.
"""

import argparse
import os

import matplotlib
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from common import evaluate, find_threshold, load_split, scale_pos_weight, segment_names
from config import RANDOM_STATE, XGB_PARAMS, results_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def make_model(y_train):
    return xgb.XGBClassifier(scale_pos_weight=scale_pos_weight(y_train), **XGB_PARAMS)


def cross_validate(X_train, y_train, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X_train))
    fold_metrics = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        model = make_model(y_train.iloc[train_idx])
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        proba = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        oof[val_idx] = proba
        fold_metrics.append(evaluate(y_train.iloc[val_idx], (proba >= 0.5).astype(int), proba))
    mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    return oof, mean_metrics


def plot_feature_importance(model, path, top_n=9):
    gains = model.get_booster().get_score(importance_type="gain")
    total = sum(gains.values())
    top = sorted(gains.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names = [name for name, _ in top][::-1]
    shares = [100 * value / total for _, value in top][::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(names, shares, color="navy")
    plt.xlabel("Importance (%)")
    plt.title(f"XGBoost feature importance (top {top_n} by share of gain)")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.set_axis_off()
    table = ax.table(
        cellText=[[str(cm[0, 0]), str(cm[0, 1])], [str(cm[1, 0]), str(cm[1, 1])]],
        rowLabels=["Actual No Churn", "Actual Churn"],
        colLabels=["Pred No Churn", "Pred Churn"],
        loc="center",
        cellLoc="center",
    )
    table.set_fontsize(14)
    table.scale(1, 2)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_threshold_comparison(y_true, proba, threshold, path):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(proba[y_true == 0], fill=True, color="blue", alpha=0.3, label="No churn")
    sns.kdeplot(proba[y_true == 1], fill=True, color="red", alpha=0.3, label="Churn")
    plt.axvline(x=0.5, color="black", linestyle="--", label="Threshold 0.5")
    plt.axvline(x=threshold, color="darkred", linestyle="-", label=f"Chosen threshold {threshold:.3f}")
    plt.legend()
    plt.title("Predicted probability on the test set")
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc(y_true, proba, path):
    fpr, tpr, _ = roc_curve(y_true, proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="blue", lw=2)
    plt.fill_between(fpr, tpr, color="lightblue", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.text(
        0.5, 0.5, f"AUC = {roc_auc_score(y_true, proba):.3f}", fontsize=12, bbox=dict(facecolor="white", alpha=0.8)
    )
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main(segment: str):
    out = results_dir(segment)
    X_train, X_test, y_train, y_test = load_split(segment)

    oof, cv_metrics = cross_validate(X_train, y_train)
    threshold = find_threshold(np.asarray(y_train), oof)
    print("5-fold validation:", {k: round(v, 4) for k, v in cv_metrics.items()})
    print(f"Threshold from out-of-fold predictions: {threshold:.4f}")

    model = make_model(y_train)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    test_metrics = evaluate(y_test, pred, proba)
    print("Test:", {k: round(v, 4) for k, v in test_metrics.items()})

    plot_feature_importance(model, os.path.join(out, "feature_importance.png"))
    plot_confusion_matrix(y_test, pred, os.path.join(out, "confusion_matrix.png"))
    plot_threshold_comparison(np.asarray(y_test), proba, threshold, os.path.join(out, "threshold_comparison.png"))
    plot_roc(y_test, proba, os.path.join(out, "roc_curve.png"))
    with open(os.path.join(out, "test_metrics.txt"), "w") as fh:
        fh.write("Test Metrics\n")
        for name, value in test_metrics.items():
            fh.write(f"{name}: {value:.4f}\n")
        fh.write(f"Threshold: {threshold:.4f}\n")
    print(f"Results written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", choices=segment_names(), required=True)
    main(parser.parse_args().segment)
