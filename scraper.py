"""
soccerdata / FBref setup check.

Run this first. It pulls one season of Big 5 player stats and tells you
exactly what shape the data is in before you build anything on top of it.

    pip install soccerdata pandas
    python fbref_setup.py

First run is slow - soccerdata rate-limits itself to be polite to FBref,
and each stat type is a separate page. Subsequent runs read from the local
cache (~/soccerdata/data/FBref) and are instant.
"""

import pandas as pd
import soccerdata as sd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

SEASON = "2526"          # 2023-24. Also accepts 2023 or "2023-2024".
LEAGUE = "Big 5 European Leagues Combined"   # one request covers all five


def flatten(df):
    """FBref returns two header rows. Collapse them into single names."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            b if (not a or a.startswith("Unnamed")) else f"{a}_{b}"
            for a, b in df.columns
        ]
    return df


def main():
    print("Available leagues:")
    for lg in sd.FBref.available_leagues():
        print("  ", lg)

    fbref = sd.FBref(leagues=LEAGUE, seasons=SEASON)

    # ---- 1. standard stats: minutes, goals, assists, age, position -------
    print(f"\nPulling standard stats for {SEASON} ...")
    std = fbref.read_player_season_stats(stat_type="standard")

    print("\nIndex levels:", std.index.names)
    print("Shape:", std.shape)
    print("\nRaw column MultiIndex (first 15):")
    for c in list(std.columns)[:15]:
        print("  ", c)

    flat = flatten(std).reset_index()
    flat.loc[flat["league"].isna(), "league"] = "GER-Bundesliga"
    print("\nFlattened columns:")
    print(list(flat.columns))

    print("\nFirst 5 rows, key fields:")
    keep = [c for c in flat.columns
            if c in ("league", "season", "team", "player", "nation", "pos", "age")
            or "90s" in c or "Min" in c or c.endswith("_Gls") or c.endswith("_Ast")]
    print(flat[keep].head())

    # ---- 2. playing time: the minutes detail --------------------------
    print("\nPulling playing_time ...")
    pt = flatten(fbref.read_player_season_stats(stat_type="playing_time")).reset_index()
    print("Shape:", pt.shape)
    print(list(pt.columns))

    # ---- 3. sanity checks --------------------------------------------
    print("\n--- sanity ---")
    print("Players:", flat["player"].nunique())
    print("Teams:  ", flat["team"].nunique(), "(expect ~98 across Big 5)")
    print("Duplicate player+team rows:",
          flat.duplicated(["player", "team"]).sum())

    mins = [c for c in flat.columns if c == "Min" or c.endswith("_Min")]
    if mins:
        print(f"Minutes column '{mins[0]}' range:",
              flat[mins[0]].min(), "-", flat[mins[0]].max())

    if "pos" in flat:
        print("\nPositions:")
        print(flat["pos"].value_counts().head(10))

    flat.to_csv("fbref_standard_sample.csv", index=False)
    print("\nWrote fbref_standard_sample.csv")


if __name__ == "__main__":
    main()