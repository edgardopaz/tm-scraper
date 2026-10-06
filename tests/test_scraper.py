import pandas as pd
import pytest

from scraper import collapse_transfers, fix_league, flatten


def test_flatten():
    columns = pd.MultiIndex.from_tuples(
        [
            ("Unnamed: 0_level_0", "player"),
            ("Playing Time", "MP"),
            ("Performance", "Gls"),
        ]
    )
    raw = pd.DataFrame([["Alex", 20, 8]], columns=columns)

    result = flatten(raw)

    assert list(result.columns) == ["player", "Playing Time_MP", "Performance_Gls"]
    assert result.iloc[0].tolist() == ["Alex", 20, 8]
    assert isinstance(raw.columns, pd.MultiIndex)


def test_fix_league():
    teams = [f"Bundesliga Team {number}" for number in range(1, 19)]
    raw = pd.DataFrame(
        {
            "team": teams + ["Premier League Team"],
            "league": [None] * 18 + ["ENG-Premier League"],
        }
    )

    result = fix_league(raw)

    assert (result.loc[:17, "league"] == "GER-Bundesliga").all()
    assert result.loc[18, "league"] == "ENG-Premier League"
    assert raw["league"].isna().sum() == 18

    unexpected = pd.DataFrame({"team": ["Unknown Team"], "league": [None]})
    with pytest.raises(ValueError, match="Expected 0 or 18 unlabelled teams"):
        fix_league(unexpected)


def test_collapse_transfers():
    raw = pd.DataFrame(
        [
            {
                "player": "Alex",
                "born_": 1998,
                "season": "2526",
                "team": "Primary FC",
                "league": "ENG-Premier League",
                "nation_": "ENG",
                "pos_": "FW,MF",
                "age_": 27,
                "Playing Time_MP": 10,
                "Playing Time_Min": 900,
                "Playing Time_90s": 10,
                "Performance_Gls": 5,
                "Performance_Ast": 1,
                "Performance_G+A": 6,
            },
            {
                "player": "Alex",
                "born_": 1998,
                "season": "2526",
                "team": "Former FC",
                "league": "ENG-Premier League",
                "nation_": "ENG",
                "pos_": "MF,FW",
                "age_": 27,
                "Playing Time_MP": 5,
                "Playing Time_Min": 450,
                "Playing Time_90s": 5,
                "Performance_Gls": 2,
                "Performance_Ast": 2,
                "Performance_G+A": 4,
            },
        ]
    )

    result = collapse_transfers(raw)

    assert len(result) == 1
    player = result.iloc[0]
    assert player["team"] == "Primary FC"
    assert player["Playing Time_MP"] == 15
    assert player["Playing Time_Min"] == 1350
    assert player["Performance_Gls"] == 7
    assert player["Performance_Ast"] == 3
    assert player["Performance_G+A"] == 10
    assert player["Per90_Gls"] == pytest.approx(0.467)
    assert player["Per90_Ast"] == pytest.approx(0.2)
    assert player["Per90_G+A"] == pytest.approx(0.667)
    assert player["pos_primary"] == "FW"
    assert player["n_clubs"] == 2