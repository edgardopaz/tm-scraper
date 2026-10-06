import pandas as pd
import pytest
import soccerdata as sd

from scraper import LEAGUE, SEASONS, flatten


@pytest.mark.network
def test_fbref_returns_player_season_stats(tmp_path):
    """Confirm that soccerdata can download and parse live FBref player data."""
    fbref = sd.FBref(
        leagues=LEAGUE,
        seasons=SEASONS[-1],
        no_cache=True,
        no_store=True,
        data_dir=tmp_path / "soccerdata",
        headless=True,
    )

    stats = flatten(
        fbref.read_player_season_stats(stat_type="standard")
    ).reset_index()

    assert isinstance(stats, pd.DataFrame)
    assert not stats.empty
    assert {"player", "team", "league", "Playing Time_MP"} <= set(stats.columns)
