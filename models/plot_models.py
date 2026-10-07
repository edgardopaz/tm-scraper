"""
Plot the three holdout models on the same scale.

    python models/plot_models.py

Reads the saved predictions and metric files. The 2025/26 season is the
holdout already used to score each model.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "value_model"
OUTPUT_PATH = REPORT_DIR / "model_comparison.png"

MODELS = [
    ("Median baseline", "median_baseline"),
    ("Linear regression", "linear_regression"),
    ("Gradient boosting", "gradient_boosting"),
]


def load_model(slug):
    metrics = json.loads((MODEL_DIR / f"{slug}.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(REPORT_DIR / f"{slug}_predictions.csv")
    return metrics, predictions


def plot_models(models, output_path):
    """Draw predicted-versus-actual and log-error panels for each model."""
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharey="row", layout="constrained")
    log_errors = []
    for _, predictions in models:
        actual = predictions["market_value_eur"].to_numpy(dtype=float)
        predicted = predictions["predicted_value_eur"].to_numpy(dtype=float)
        log_errors.append(np.log(predicted) - np.log(actual))

    lowest = min(frame["market_value_eur"].min() for _, frame in models)
    highest = max(
        max(frame["market_value_eur"].max(), frame["predicted_value_eur"].max())
        for _, frame in models
    )
    error_limit = float(np.quantile(np.abs(np.concatenate(log_errors)), 0.99))
    bins = np.linspace(-error_limit, error_limit, 41)

    for column, ((name, _), (metrics, predictions)) in enumerate(zip(MODELS, models)):
        actual = predictions["market_value_eur"].to_numpy(dtype=float)
        predicted = predictions["predicted_value_eur"].to_numpy(dtype=float)
        scatter = axes[0, column]
        scatter.scatter(actual, predicted, s=8, alpha=0.35, color="#3d5a80", linewidths=0)
        scatter.plot([lowest, highest], [lowest, highest], color="#1a1a1a", linewidth=1)
        scatter.set_xscale("log")
        scatter.set_yscale("log")
        scatter.set_xlim(lowest, highest)
        scatter.set_ylim(lowest, highest)
        scatter.set_title(
            f"{name}\n"
            f"MAE €{metrics['mae_eur'] / 1e6:.1f} million\n"
            f"median error {metrics['median_absolute_percentage_error']:.1%}\n"
            f"log R² {metrics['log_r2']:.3f}",
            fontsize=10,
        )
        scatter.set_xlabel("Actual market value (euros)")
        if column == 0:
            scatter.set_ylabel("Predicted market value (euros)")

        histogram = axes[1, column]
        histogram.hist(log_errors[column], bins=bins, color="#3d5a80")
        histogram.axvline(0, color="#1a1a1a", linewidth=1)
        histogram.set_xlim(-error_limit, error_limit)
        histogram.set_xlabel("Log error (log predicted − log actual)")
        if column == 0:
            histogram.set_ylabel("Players")

    holdout = models[0][0]["holdout_season"]
    players = models[0][0]["test_rows"]
    figure.suptitle(f"Holdout season {holdout}, {players:,} players")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def main():
    models = []
    for name, slug in MODELS:
        metrics_path = MODEL_DIR / f"{slug}.json"
        predictions_path = REPORT_DIR / f"{slug}_predictions.csv"
        if not metrics_path.exists() or not predictions_path.exists():
            raise FileNotFoundError(
                f"Run models/{slug}.py before plotting. Missing {slug} results."
            )
        models.append(load_model(slug))

    output_path = plot_models(models, OUTPUT_PATH)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
