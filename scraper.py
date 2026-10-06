"""
Pull several seasons of Big 5 player stats from FBref and produce one clean
player-season table.

    python scraper.py

What it does:
  1. Pulls each season separately, saving raw output as it goes. If a season
     fails or you interrupt the run, re-running skips what's already saved.
  2. Patches the missing Bundesliga league label (a soccerdata mapping gap on
     the Big 5 combined page).
  3. Collapses mid-season transfers into one row per player-season.

Output: fbref_players.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soccerdata as sd

SEASONS = ["2122", "2223", "2324", "2425", "2526"]
LEAGUE = "Big 5 European Leagues Combined"
RAW_DIR = Path("data/raw")
OUT = Path("data/fbref_players.csv")

# Longest Big 5 season is 38 games; allow slack for players crossing leagues.
MAX_MATCHES = 42
MAX_MINUTES = 3800

# Counting stats: safe to add across a player's two club spells.
SUM_COLS = [
    "Playing Time_MP", "Playing Time_Starts", "Playing Time_Min",
    "Playing Time_90s", "Performance_Gls", "Performance_Ast",
    "Performance_G+A", "Performance_G-PK", "Performance_PK",
    "Performance_PKatt", "Performance_CrdY", "Performance_CrdR",
]

# Attributes: take from whichever spell had the most minutes.
FIRST_COLS = ["team", "league", "nation_", "pos_", "age_", "born_"]


def flatten(df):
    """FBref tables have two header rows. Collapse to single column names."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            b if (not a or a.startswith("Unnamed")) else f"{a}_{b}"
            for a, b in df.columns
        ]
    return df


def fix_league(df):
    """Bundesliga rows come back unlabelled. Patch, but verify the shape first."""
    n_teams = df.loc[df["league"].isna(), "team"].nunique()
    if n_teams not in (0, 18):
        raise ValueError(
            f"Expected 0 or 18 unlabelled teams (Bundesliga), got {n_teams}. "
            "The mapping failure changed shape - inspect before patching."
        )
    df = df.copy()
    df.loc[df["league"].isna(), "league"] = "GER-Bundesliga"
    assert df["league"].notna().all()
    return df


def pull_season(season):
    """Fetch one season, or load it from disk if already pulled."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"fbref_{season}.csv"

    if path.exists():
        print(f"{season}: cached")
        return pd.read_csv(path)

    print(f"{season}: pulling (slow - browser + 7s rate limit) ...")
    fbref = sd.FBref(leagues=LEAGUE, seasons=season, headless=False)
    df = flatten(fbref.read_player_season_stats(stat_type="standard")).reset_index()
    df = fix_league(df)
    df.to_csv(path, index=False)
    print(f"{season}: {len(df):,} rows saved")
    return df


def collapse_transfers(df):
    """
    One row per player-season.

    A player who moves in January appears twice - once per club. Summing his
    counting stats and keeping the club where he played most minutes turns him
    back into a regular starter instead of two players.

    Grouped on player + birth year so two players sharing a name stay separate.
    """
    df = df.sort_values("Playing Time_Min", ascending=False).copy()

    # --- guard against same-name, same-birth-year players -----------------
    # Two different Portuguese players named Vitinha, both born 2000, play in
    # different leagues in the same season. Name + birth year merges them into
    # one impossible player with 64 appearances. A real mid-season transfer
    # cannot exceed the season length, so use that to tell the cases apart.
    totals = df.groupby(["player", "born_", "season"], dropna=False)[
        "Playing Time_MP"
    ].transform("sum")
    df["_collision"] = totals > MAX_MATCHES

    if df["_collision"].any():
        clashes = df.loc[df["_collision"], ["player", "born_", "season"]].drop_duplicates()
        print(f"  {len(clashes)} name collision(s) kept separate:")
        for _, r in clashes.iterrows():
            print(f"    {r['player']} (born {r['born_']:.0f}, {r['season']})")

    # Collided players are keyed by club too, so they stay as separate rows.
    df["_key"] = np.where(df["_collision"], df["team"], "")

    agg = {c: "sum" for c in SUM_COLS if c in df.columns}
    agg.update({c: "first" for c in FIRST_COLS if c in df.columns})
    # rows are sorted by minutes desc, so "first" = the main club spell

    out = df.groupby(
        ["player", "born_", "season", "_key"], as_index=False, dropna=False
    ).agg(agg)

    # Per-90 rates can't be summed - recompute from the combined totals.
    # np.nan, not pd.NA: the latter makes the column object dtype and breaks round().
    nineties = out["Playing Time_90s"].replace(0, np.nan)
    for src, dst in [
        ("Performance_Gls", "Per90_Gls"),
        ("Performance_Ast", "Per90_Ast"),
        ("Performance_G+A", "Per90_G+A"),
    ]:
        out[dst] = (out[src] / nineties).round(3)

    # Primary position: FBref uses "MF,FW" for dual-role players.
    out["pos_primary"] = out["pos_"].str.split(",").str[0]

    # How many clubs the player turned out for. Merged on the keys rather than
    # assigned positionally - groupby order is not guaranteed to line up.
    n_clubs = (
        df.groupby(["player", "born_", "season", "_key"], dropna=False)["team"]
        .nunique()
        .reset_index(name="n_clubs")
    )
    out = out.merge(n_clubs, on=["player", "born_", "season", "_key"], how="left")

    # Nothing downstream should see the disambiguation key.
    out = out.drop(columns=["_key"])

    # Fail loudly rather than let a fabricated player into training data.
    bad_mp = out[out["Playing Time_MP"] > MAX_MATCHES]
    bad_min = out[out["Playing Time_Min"] > MAX_MINUTES]
    if len(bad_mp) or len(bad_min):
        print(bad_mp[["player", "born_", "season", "Playing Time_MP"]].to_string())
        raise ValueError("Impossible totals after aggregation - unhandled collision.")

    return out


def main():
    frames = [pull_season(s) for s in SEASONS]
    raw = pd.concat(frames, ignore_index=True)
    print(f"\nCombined raw: {len(raw):,} rows")

    clean = collapse_transfers(raw)
    print(f"After collapsing transfers: {len(clean):,} rows")

    print("\n--- checks ---")
    print(clean.groupby("season").size().rename("rows"))
    print("\nnull leagues:", clean["league"].isna().sum())
    print("dup player+season:",
          clean.duplicated(["player", "born_", "season"]).sum())
    print("minutes range:",
          clean["Playing Time_Min"].min(), "-", clean["Playing Time_Min"].max())
    print("\npositions:")
    print(clean["pos_primary"].value_counts())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()