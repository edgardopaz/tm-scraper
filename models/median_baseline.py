"""
Predict every player-season with the training-set median market value.

    python models/median_baseline.py

The 2025/26 season is held out. Earlier seasons supply the single median.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "modeling_players.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "median_baseline.json"
PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "reports" / "value_model" / "median_baseline_predictions.csv"
)
HOLDOUT_SEASON = "2526"
TARGET = "market_value_eur"


def season_label(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def load_modeling_data(path):
    frame = pd.read_csv(path)
    if TARGET not in frame.columns or "season" not in frame.columns:
        raise ValueError(f"{path} must contain season and {TARGET}.")
    frame = frame.copy()
    frame["season"] = frame["season"].map(season_label)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise")
    if frame[TARGET].isna().any():
        raise ValueError("Market values contain missing numbers.")
    positive = frame.loc[frame[TARGET] > 0].copy()
    dropped = len(frame) - len(positive)
    if positive.empty:
        raise ValueError("No positive market values are available.")
    positive.attrs["dropped_nonpositive"] = dropped
    return positive


def split_holdout(frame, holdout_season=HOLDOUT_SEASON):
    train = frame.loc[frame["season"] != holdout_season].copy()
    test = frame.loc[frame["season"] == holdout_season].copy()
    if train.empty or test.empty:
        raise ValueError(f"Both training rows and {holdout_season} rows are required.")
    return train, test


def fit_median_baseline(values):
    """Fit a constant predictor equal to the median training value."""
    model = DummyRegressor(strategy="median")
    model.fit(np.zeros((len(values), 1)), np.asarray(values, dtype=float))
    return model


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    percentage_error = np.abs(predicted - actual) / actual
    log_r2 = (
        float(r2_score(np.log(actual), np.log(predicted)))
        if len(actual) > 1
        else None
    )
    return {
        "mae_eur": float(mean_absolute_error(actual, predicted)),
        "median_absolute_percentage_error": float(np.median(percentage_error)),
        "log_r2": log_r2,
    }


def evaluate_median_baseline(frame, holdout_season=HOLDOUT_SEASON):
    train, test = split_holdout(frame, holdout_season)
    model = fit_median_baseline(train[TARGET])
    predicted = model.predict(np.zeros((len(test), 1)))
    metrics = regression_metrics(test[TARGET], predicted)
    metrics.update(
        {
            "median_value_eur": float(model.constant_[0][0]),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "holdout_season": holdout_season,
            "train_seasons": sorted(train["season"].unique()),
            "dropped_nonpositive": int(frame.attrs.get("dropped_nonpositive", 0)),
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

    _, metrics, predictions = evaluate_median_baseline(
        load_modeling_data(args.data),
        holdout_season=args.holdout_season,
    )
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    predictions.to_csv(args.predictions, index=False)

    print(f"Training seasons: {', '.join(metrics['train_seasons'])}")
    print(f"Holdout season: {metrics['holdout_season']} ({metrics['test_rows']:,} players)")
    print(f"Median value: €{metrics['median_value_eur']:,.0f}")
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
