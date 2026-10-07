import pandas as pd

from linear_regression import evaluate_linear_regression


def _row(player, season, goals_per_90):
    return {
        "fbref_player": player,
        "season": season,
        "league_code": "GB1",
        "fbref_team": "Arsenal",
        "pos_primary": "MF",
        "age_": 25,
        "Playing Time_90s": 30,
        "Per90_Gls": goals_per_90,
        "Per90_Ast": 0.1,
        "Performance_CrdY": 2,
        "Performance_CrdR": 0,
        "n_clubs": 1,
        "market_value_eur": 1_000_000 * (2**goals_per_90),
    }


def test_linear_regression_raises_value_with_goals_per_90():
    rows = [_row(f"Train {goals}-{repeat}", "2223", goals) for goals in range(4) for repeat in range(6)]
    rows.extend(_row(f"Test {goals}", "2526", goals) for goals in (0, 3))

    model, metrics, predictions = evaluate_linear_regression(pd.DataFrame(rows))
    predicted = predictions.set_index("fbref_player")["predicted_value_eur"]
    goals_coefficient = next(
        row["log_coefficient"]
        for row in metrics["coefficients"]
        if row["feature"] == "Per90_Gls"
    )

    assert predicted["Test 3"] > predicted["Test 0"]
    assert goals_coefficient > 0
    assert (predictions["predicted_value_eur"] > 0).all()
    assert model.named_steps["linear"].intercept_ is not None
