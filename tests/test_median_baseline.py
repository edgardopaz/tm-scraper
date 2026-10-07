import numpy as np
import pandas as pd

from median_baseline import evaluate_median_baseline, fit_median_baseline


def test_median_baseline_predicts_the_training_median():
    frame = pd.DataFrame(
        {
            "fbref_player": ["A", "B", "C", "D"],
            "season": ["2122", "2223", "2324", "2526"],
            "league_code": ["GB1", "GB1", "GB1", "GB1"],
            "fbref_team": ["Arsenal", "Arsenal", "Arsenal", "Arsenal"],
            "market_value_eur": [1_000_000, 3_000_000, 5_000_000, 9_000_000],
        }
    )

    model, metrics, predictions = evaluate_median_baseline(frame)

    assert model.constant_[0][0] == 3_000_000
    assert metrics["median_value_eur"] == 3_000_000
    assert metrics["train_rows"] == 3
    assert metrics["test_rows"] == 1
    assert predictions["predicted_value_eur"].tolist() == [3_000_000]
    assert metrics["mae_eur"] == 6_000_000


def test_fit_median_baseline_uses_the_middle_training_value():
    model = fit_median_baseline([4, 1, 2])

    assert model.predict(np.zeros((2, 1))).tolist() == [2, 2]
