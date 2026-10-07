import pandas as pd

from gradient_boosting import evaluate_gradient_boosting


def test_gradient_boosting_ranks_higher_scoring_players_higher():
    rows = []
    for goals in range(6):
        for repeat in range(8):
            rows.append(
                {
                    "fbref_player": f"Train {goals}-{repeat}",
                    "season": "2223",
                    "league_code": "GB1",
                    "fbref_team": "Arsenal",
                    "pos_primary": "FW",
                    "age_": 24,
                    "Playing Time_MP": 30,
                    "Playing Time_Starts": 30,
                    "Playing Time_Min": 2700,
                    "Playing Time_90s": 30,
                    "Performance_Gls": goals,
                    "Performance_Ast": 0,
                    "Performance_G+A": goals,
                    "Performance_G-PK": goals,
                    "Performance_PK": 0,
                    "Performance_PKatt": 0,
                    "Performance_CrdY": 0,
                    "Performance_CrdR": 0,
                    "Per90_Gls": goals / 30,
                    "Per90_Ast": 0,
                    "Per90_G+A": goals / 30,
                    "n_clubs": 1,
                    "market_value_eur": 1_000_000 * (2**goals),
                }
            )
    for goals in (0, 5):
        rows.append(
            {
                "fbref_player": f"Test {goals}",
                "season": "2526",
                "league_code": "GB1",
                "fbref_team": "Arsenal",
                "pos_primary": "FW",
                "age_": 24,
                "Playing Time_MP": 30,
                "Playing Time_Starts": 30,
                "Playing Time_Min": 2700,
                "Playing Time_90s": 30,
                "Performance_Gls": goals,
                "Performance_Ast": 0,
                "Performance_G+A": goals,
                "Performance_G-PK": goals,
                "Performance_PK": 0,
                "Performance_PKatt": 0,
                "Performance_CrdY": 0,
                "Performance_CrdR": 0,
                "Per90_Gls": goals / 30,
                "Per90_Ast": 0,
                "Per90_G+A": goals / 30,
                "n_clubs": 1,
                "market_value_eur": 1_000_000 * (2**goals),
            }
        )

    _, _, predictions = evaluate_gradient_boosting(
        pd.DataFrame(rows),
        min_samples_leaf=1,
        max_iter=40,
    )
    predicted = predictions.set_index("fbref_player")["predicted_value_eur"]

    assert predicted["Test 5"] > predicted["Test 0"]
    assert (predictions["predicted_value_eur"] > 0).all()
