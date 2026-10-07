"""
Select one end-of-season Transfermarkt valuation per player-season-league.

    python feature/transfermarkt-values/build_season_values.py

For each row, the target is the latest valuation available on or before that
league's final matchday. Valuations older than 365 days are excluded.
"""

import argparse
from pathlib import Path

import pandas as pd

from transfermarkt import DATA_DIR

DEFAULT_PLAYERS = DATA_DIR / "transfermarkt_players.csv"
DEFAULT_HISTORY = DATA_DIR / "transfermarkt_value_history.csv"
DEFAULT_OUTPUT = DATA_DIR / "transfermarkt_season_values.csv"

KEY_COLUMNS = ["transfermarkt_player_id", "season", "league_code"]

SEASON_CUTOFFS = {
    ("GB1", "2122"): "2022-05-22",
    ("ES1", "2122"): "2022-05-22",
    ("L1", "2122"): "2022-05-14",
    ("IT1", "2122"): "2022-05-22",
    ("FR1", "2122"): "2022-05-21",
    ("GB1", "2223"): "2023-05-28",
    ("ES1", "2223"): "2023-06-04",
    ("L1", "2223"): "2023-05-27",
    ("IT1", "2223"): "2023-06-04",
    ("FR1", "2223"): "2023-06-03",
    ("GB1", "2324"): "2024-05-19",
    ("ES1", "2324"): "2024-05-26",
    ("L1", "2324"): "2024-05-18",
    ("IT1", "2324"): "2024-05-26",
    ("FR1", "2324"): "2024-05-19",
    ("GB1", "2425"): "2025-05-25",
    ("ES1", "2425"): "2025-05-25",
    ("L1", "2425"): "2025-05-17",
    ("IT1", "2425"): "2025-05-25",
    ("FR1", "2425"): "2025-05-17",
    ("GB1", "2526"): "2026-05-24",
    ("ES1", "2526"): "2026-05-24",
    ("L1", "2526"): "2026-05-16",
    ("IT1", "2526"): "2026-05-24",
    ("FR1", "2526"): "2026-05-17",
}

OUTPUT_COLUMNS = [
    "transfermarkt_player_id",
    "player_name",
    "season",
    "league_code",
    "cutoff_date",
    "valuation_date",
    "valuation_age_days",
    "market_value_eur",
    "valuation_club",
    "age_at_valuation",
]


def load_player_seasons(path):
    """Create one identity row per Transfermarkt player-season-league."""
    roster = pd.read_csv(
        path,
        dtype={
            "transfermarkt_player_id": "int64",
            "season": "string",
            "league_code": "string",
        },
    )
    required = set(KEY_COLUMNS + ["player_name"])
    missing = required.difference(roster.columns)
    if missing:
        raise ValueError(f"Player roster is missing fields: {sorted(missing)}")

    player_seasons = (
        roster.sort_values(KEY_COLUMNS + ["player_name"])
        .drop_duplicates(KEY_COLUMNS, keep="last")
        [KEY_COLUMNS + ["player_name"]]
        .reset_index(drop=True)
    )

    cutoff_table = pd.DataFrame(
        [
            {
                "league_code": league_code,
                "season": season,
                "cutoff_date": cutoff,
            }
            for (league_code, season), cutoff in SEASON_CUTOFFS.items()
        ]
    )
    cutoff_table["league_code"] = cutoff_table["league_code"].astype("string")
    cutoff_table["season"] = cutoff_table["season"].astype("string")
    cutoff_table["cutoff_date"] = pd.to_datetime(
        cutoff_table["cutoff_date"], errors="raise"
    )

    player_seasons = player_seasons.merge(
        cutoff_table,
        on=["league_code", "season"],
        how="left",
        validate="many_to_one",
    )
    missing_cutoffs = player_seasons.loc[
        player_seasons["cutoff_date"].isna(), ["league_code", "season"]
    ].drop_duplicates()
    if not missing_cutoffs.empty:
        raise ValueError(
            "Missing season cutoffs:\n"
            f"{missing_cutoffs.to_string(index=False)}"
        )

    return player_seasons


def load_valuation_history(path):
    """Read valuation events and parse their dates and numeric values."""
    history = pd.read_csv(
        path,
        dtype={"transfermarkt_player_id": "int64"},
    )
    required = {
        "transfermarkt_player_id",
        "valuation_date",
        "market_value_eur",
        "club",
        "age",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Valuation history is missing fields: {sorted(missing)}")

    history["valuation_date"] = pd.to_datetime(
        history["valuation_date"], errors="raise"
    )
    history["market_value_eur"] = pd.to_numeric(
        history["market_value_eur"], errors="raise"
    )
    return history.rename(
        columns={
            "club": "valuation_club",
            "age": "age_at_valuation",
        }
    )


def build_season_values(player_seasons, history, max_age_days=365):
    """Choose the latest non-stale valuation at each league-season cutoff."""
    history_fields = [
        "transfermarkt_player_id",
        "valuation_date",
        "market_value_eur",
        "valuation_club",
        "age_at_valuation",
    ]
    candidates = player_seasons.merge(
        history[history_fields],
        on="transfermarkt_player_id",
        how="left",
        validate="many_to_many",
    )
    candidates = candidates.loc[
        candidates["valuation_date"].notna()
        & (candidates["valuation_date"] <= candidates["cutoff_date"])
    ]

    selected = (
        candidates.sort_values(KEY_COLUMNS + ["valuation_date"])
        .drop_duplicates(KEY_COLUMNS, keep="last")
        .reset_index(drop=True)
    )
    selected["valuation_age_days"] = (
        selected["cutoff_date"] - selected["valuation_date"]
    ).dt.days

    valid = selected.loc[
        selected["valuation_age_days"].between(0, max_age_days, inclusive="both")
    ].copy()
    valid["valuation_age_days"] = valid["valuation_age_days"].astype("int64")
    valid["market_value_eur"] = valid["market_value_eur"].astype("int64")
    valid["age_at_valuation"] = pd.to_numeric(
        valid["age_at_valuation"], errors="coerce"
    ).astype("Int64")
    valid = valid[OUTPUT_COLUMNS].sort_values(
        ["season", "league_code", "player_name"]
    )

    if valid.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate player-season-league targets remain.")
    if (valid["valuation_date"] > valid["cutoff_date"]).any():
        raise ValueError("A selected valuation occurs after its season cutoff.")

    return valid.reset_index(drop=True), selected


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--players",
        type=Path,
        default=DEFAULT_PLAYERS,
        help=f"Roster CSV (default: {DEFAULT_PLAYERS}).",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY,
        help=f"Valuation history CSV (default: {DEFAULT_HISTORY}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Season values CSV (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=365,
        help="Exclude selected valuations older than this many days.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_age_days < 0:
        raise ValueError("--max-age-days cannot be negative.")
    if not args.players.exists():
        raise FileNotFoundError(f"Roster not found: {args.players}")
    if not args.history.exists():
        raise FileNotFoundError(f"Valuation history not found: {args.history}")

    player_seasons = load_player_seasons(args.players)
    history = load_valuation_history(args.history)
    values, selected = build_season_values(
        player_seasons,
        history,
        max_age_days=args.max_age_days,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    values.to_csv(args.output, index=False, date_format="%Y-%m-%d")

    no_prior_value = len(player_seasons) - len(selected)
    stale_values = len(selected) - len(values)
    print(f"Unique player-season-league rows: {len(player_seasons):,}")
    print(f"Excluded without a prior valuation: {no_prior_value:,}")
    print(f"Excluded older than {args.max_age_days} days: {stale_values:,}")
    print(f"Wrote {len(values):,} season values to {args.output}")


if __name__ == "__main__":
    main()
