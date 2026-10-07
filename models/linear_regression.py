"""
Predict end-of-season market value with a linear model.

    python models/linear_regression.py

The model learns log(market value). The 2025/26 season is held out.
Numeric inputs are scaled, so each coefficient is the change in log value
for a one-standard-deviation increase. Position is compared with midfielders
and league is compared with the Premier League.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from median_baseline import (
    DATA_PATH,
    HOLDOUT_SEASON,
    TARGET,
    load_modeling_data,
    regression_metrics,
    split_holdout,
)

MODEL_PATH = Path(__file__).resolve().parent / "linear_regression.json"
PREDICTIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "reports"
    / "value_model"
    / "linear_regression_predictions.csv"
)

# Rates and volume stay separate. Totals such as goals and goal contributions
# repeat those two inputs, so they are left out.
NUMERIC_FEATURES = [
    "age_from_25",
    "age_from_25_sq",
    "Playing Time_90s",
    "Per90_Gls",
    "Per90_Ast",
    "Performance_CrdY",
    "Performance_CrdR",
    "n_clubs",
]
POSITION_COLUMN = "pos_primary"
LEAGUE_COLUMN = "league_code"
POSITION_LEVELS = ["DF", "FW", "GK", "MF"]
LEAGUE_LEVELS = ["ES1", "FR1", "GB1", "IT1", "L1"]
POSITION_REFERENCE = "MF"
LEAGUE_REFERENCE = "GB1"
SOURCE_COLUMNS = [
    "age_",
    "Playing Time_90s",
    "Per90_Gls",
    "Per90_Ast",
    "Performance_CrdY",
    "Performance_CrdR",
    "n_clubs",
    POSITION_COLUMN,
    LEAGUE_COLUMN,
]


def design_matrix(frame):
    """Build the linear-model inputs from a season row."""
    missing = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Modeling data is missing features: {missing}")

    age = pd.to_numeric(frame["age_"], errors="coerce")
    features = pd.DataFrame(index=frame.index)
    features["age_from_25"] = (age - 25).fillna(0)
    features["age_from_25_sq"] = features["age_from_25"] ** 2
    features["Playing Time_90s"] = pd.to_numeric(frame["Playing Time_90s"], errors="coerce").fillna(0)
    features["Per90_Gls"] = pd.to_numeric(frame["Per90_Gls"], errors="coerce").fillna(0)
    features["Per90_Ast"] = pd.to_numeric(frame["Per90_Ast"], errors="coerce").fillna(0)
    features["Performance_CrdY"] = pd.to_numeric(frame["Performance_CrdY"], errors="coerce").fillna(0)
    features["Performance_CrdR"] = pd.to_numeric(frame["Performance_CrdR"], errors="coerce").fillna(0)
    features["n_clubs"] = pd.to_numeric(frame["n_clubs"], errors="coerce").fillna(1)
    features[POSITION_COLUMN] = frame[POSITION_COLUMN].fillna("Unknown").astype(str)
    features[LEAGUE_COLUMN] = frame[LEAGUE_COLUMN].fillna("Unknown").astype(str)
    return features


def _encoder(levels, reference):
    return OneHotEncoder(
        categories=[levels],
        drop=[reference],
        handle_unknown="ignore",
        sparse_output=False,
    )


def fit_linear_regression(features, values):
    """Fit log market value on scaled stats and position/league indicators."""
    prepare = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("position", _encoder(POSITION_LEVELS, POSITION_REFERENCE), [POSITION_COLUMN]),
            ("league", _encoder(LEAGUE_LEVELS, LEAGUE_REFERENCE), [LEAGUE_COLUMN]),
        ]
    )
    model = Pipeline(
        [
            ("prepare", prepare),
            ("linear", LinearRegression()),
        ]
    )
    model.fit(features, np.log(np.asarray(values, dtype=float)))
    return model


def predict_market_value(model, features):
    """Convert log predictions back to positive euro values."""
    return np.exp(model.predict(features))


def coefficient_table(model):
    """Describe each weight as a multiplier on market value."""
    linear = model.named_steps["linear"]
    names = model.named_steps["prepare"].get_feature_names_out()
    rows = []
    for name, coefficient in zip(names, linear.coef_):
        rows.append(
            {
                "feature": _readable_feature(name),
                "log_coefficient": float(coefficient),
                "value_multiplier": float(np.exp(coefficient)),
            }
        )
    rows.append(
        {
            "feature": "intercept",
            "log_coefficient": float(linear.intercept_),
            "value_multiplier": float(np.exp(linear.intercept_)),
        }
    )
    return rows


def _readable_feature(name):
    if name.startswith("numeric__"):
        return name.removeprefix("numeric__")
    if name.startswith("position__"):
        return f"position {name.removeprefix('position__pos_primary_')} vs {POSITION_REFERENCE}"
    if name.startswith("league__"):
        return f"league {name.removeprefix('league__league_code_')} vs {LEAGUE_REFERENCE}"
    return name


def evaluate_linear_regression(frame, holdout_season=HOLDOUT_SEASON):
    train, test = split_holdout(frame, holdout_season)
    train_features = design_matrix(train)
    test_features = design_matrix(test)
    model = fit_linear_regression(train_features, train[TARGET])
    predicted = predict_market_value(model, test_features)
    metrics = regression_metrics(test[TARGET], predicted)
    metrics.update(
        {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "holdout_season": holdout_season,
            "train_seasons": sorted(train["season"].unique()),
            "dropped_nonpositive": int(frame.attrs.get("dropped_nonpositive", 0)),
            "numeric_features": NUMERIC_FEATURES,
            "position_reference": POSITION_REFERENCE,
            "league_reference": LEAGUE_REFERENCE,
            "coefficients": coefficient_table(model),
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

    _, metrics, predictions = evaluate_linear_regression(
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
    print("Coefficients (numeric inputs are per standard deviation):")
    for row in metrics["coefficients"]:
        if row["feature"] == "intercept":
            print(
                "  intercept: "
                f"€{row['value_multiplier']:,.0f} for an average "
                f"{POSITION_REFERENCE} in {LEAGUE_REFERENCE}"
            )
            continue
        print(
            f"  {row['feature']}: {row['log_coefficient']:+.3f} "
            f"({row['value_multiplier']:.2f}x value)"
        )
    print(f"Wrote {args.model}")
    print(f"Wrote {args.predictions}")


if __name__ == "__main__":
    main()
