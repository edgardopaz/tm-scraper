"""
Predict end-of-season market value from that season's FBref statistics.

    python models/gradient_boosting.py

The model learns log(market value). The 2025/26 season is held out.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from median_baseline import (
    DATA_PATH,
    HOLDOUT_SEASON,
    TARGET,
    load_modeling_data,
    regression_metrics,
    split_holdout,
)

MODEL_PATH = Path(__file__).resolve().parent / "gradient_boosting.json"
PREDICTIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "reports"
    / "value_model"
    / "gradient_boosting_predictions.csv"
)

NUMERIC_FEATURES = [
    "age_",
    "Playing Time_MP",
    "Playing Time_Starts",
    "Playing Time_Min",
    "Playing Time_90s",
    "Performance_Gls",
    "Performance_Ast",
    "Performance_G+A",
    "Performance_G-PK",
    "Performance_PK",
    "Performance_PKatt",
    "Performance_CrdY",
    "Performance_CrdR",
    "Per90_Gls",
    "Per90_Ast",
    "Per90_G+A",
    "n_clubs",
]
CATEGORICAL_FEATURES = ["pos_primary", "league_code"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def design_matrix(frame):
    """Select model inputs and mark position and league as categories."""
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Modeling data is missing features: {missing}")

    features = frame[FEATURES].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("Unknown").astype("category")
    return features


def fit_gradient_boosting(features, values, **kwargs):
    """Fit a tree ensemble to the log of market value."""
    parameters = {
        "categorical_features": "from_dtype",
        "random_state": 0,
    }
    parameters.update(kwargs)
    model = HistGradientBoostingRegressor(**parameters)
    model.fit(features, np.log(np.asarray(values, dtype=float)))
    return model


def predict_market_value(model, features):
    """Convert log predictions back to positive euro values."""
    return np.exp(model.predict(features))


def evaluate_gradient_boosting(frame, holdout_season=HOLDOUT_SEASON, **kwargs):
    train, test = split_holdout(frame, holdout_season)
    train_features = design_matrix(train)
    test_features = design_matrix(test)
    model = fit_gradient_boosting(train_features, train[TARGET], **kwargs)
    predicted = predict_market_value(model, test_features)
    metrics = regression_metrics(test[TARGET], predicted)
    metrics.update(
        {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "holdout_season": holdout_season,
            "train_seasons": sorted(train["season"].unique()),
            "dropped_nonpositive": int(frame.attrs.get("dropped_nonpositive", 0)),
            "features": FEATURES,
        }
    )
    predictions = test[
        ["fbref_player", "season", "league_code", "fbref_team", TARGET]
    ].copy()
    predictions["predicted_value_eur"] = predicted
    return model, metrics, predictions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--holdout-season", default=HOLDOUT_SEASON)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Modeling data not found: {args.data}")

    _, metrics, predictions = evaluate_gradient_boosting(
        load_modeling_data(args.data),
        holdout_season=args.holdout_season,
    )
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    predictions.to_csv(args.predictions, index=False)

    print(f"Training seasons: {', '.join(metrics['train_seasons'])}")
    print(f"Holdout season: {metrics['holdout_season']} ({metrics['test_rows']:,} players)")
    print(f"MAE: €{metrics['mae_eur']:,.0f}")
    print(
        "Median absolute percentage error: "
        f"{metrics['median_absolute_percentage_error']:.1%}"
    )
    print(f"Dropped non-positive values: {metrics['dropped_nonpositive']:,}")
    print(f"Log R²: {metrics['log_r2']:.3f}")
    print(f"Wrote {args.model}")
    print(f"Wrote {args.predictions}")


if __name__ == "__main__":
    main()
