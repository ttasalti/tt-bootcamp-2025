"""Counterfactual explanations for the customers just above the churn threshold.

    python churn/explain.py --segment postpaid

Refits the segment model, takes the 20 test customers with the lowest predicted probability
that still exceeds the threshold, and asks DiCE for three counterfactuals each over the
actionable features, weighting every feature by the inverse of its range.
"""

import argparse
import os

import dice_ml
import numpy as np
import pandas as pd
import xgboost as xgb
from dice_ml import Dice

from common import evaluate, load_split, scale_pos_weight, segment_names
from config import SEGMENTS, XGB_PARAMS, results_dir

N_CUSTOMERS = 20
N_COUNTERFACTUALS = 3


def main(segment: str):
    settings = SEGMENTS[segment]
    out = results_dir(segment)
    X_train, X_test, y_train, y_test = load_split(segment)

    model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight(y_train), **XGB_PARAMS)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= settings["threshold"]).astype(int)
    print("Test:", {k: round(v, 4) for k, v in evaluate(y_test, pred, proba).items()})

    flagged = np.where(pred == 1)[0]
    if len(flagged) < N_CUSTOMERS:
        raise SystemExit(f"Only {len(flagged)} customers above the threshold.")
    selected = flagged[np.argsort(proba[flagged])[:N_CUSTOMERS]]
    originals = X_test.iloc[selected]

    actionable = settings["actionable"]
    ranges = {c: (X_train[c].min(), X_train[c].max()) for c in actionable}
    weights = {c: 1 / (hi - lo) if hi != lo else 1 for c, (lo, hi) in ranges.items()}
    permitted = {c: [lo, hi] for c, (lo, hi) in ranges.items()}

    data = dice_ml.Data(
        dataframe=pd.concat([X_train, y_train], axis=1), continuous_features=actionable, outcome_name="churn"
    )
    explainer = Dice(data, dice_ml.Model(model=model, backend="sklearn", model_type="classifier"))

    rows = []
    for i in range(len(originals)):
        instance = originals.iloc[[i]]
        result = explainer.generate_counterfactuals(
            instance,
            total_CFs=N_COUNTERFACTUALS,
            desired_class=0,
            features_to_vary=actionable,
            proximity_weight=weights,
            permitted_range=permitted,
        )
        rows.append(instance.assign(Type="Actual"))
        rows.append(result.cf_examples_list[0].final_cfs_df.assign(Type="Counterfactual"))
        print(f"{i + 1}/{N_CUSTOMERS} done")

    path = os.path.join(out, "counterfactuals.csv")
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    print(f"Counterfactuals written to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", choices=segment_names(), required=True)
    main(parser.parse_args().segment)
