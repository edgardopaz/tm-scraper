"""
Join accepted player matches to FBref stats and Transfermarkt season values.

    python feature/player-matching/build_modeling_data.py
"""

import argparse
from pathlib import Path

import pandas as pd

from matching import DATA_DIR, prepare_fbref

DEFAULT_FBREF = DATA_DIR / "fbref_players.csv"
DEFAULT_CROSSWALK = DATA_DIR / "player_crosswalk.csv"
DEFAULT_VALUES = DATA_DIR / "transfermarkt_season_values.csv"
DEFAULT_OUTPUT = DATA_DIR / "modeling_players.csv"

VALUE_COLUMNS = [
    "transfermarkt_player_id",
    "season",
    "league_code",
    "cutoff_date",
    "valuation_date",
    "valuation_age_days",
    "market_value_eur",
    "valuation_club",
    "age_at_valuation",
]


def build_modeling_data(fbref, crosswalk, values):
    """Attach Transfermarkt IDs and end-of-season values to FBref rows."""
    if crosswalk.empty:
        raise ValueError("Player crosswalk is empty. Run match_players.py first.")

    required = {
        "fbref_player",
        "birth_year",
        "season",
        "league_code",
        "fbref_team",
        "transfermarkt_player_id",
        "match_method",
    }
    missing = required.difference(crosswalk.columns)
    if missing:
        raise ValueError(f"Crosswalk is missing fields: {sorted(missing)}")

    prepared = prepare_fbref(fbref)
    matched = prepared.merge(
        crosswalk[
            [
                "fbref_row_id",
                "transfermarkt_player_id",
                "transfermarkt_player_name",
                "tm_club",
                "match_method",
            ]
        ],
        on="fbref_row_id",
        how="inner",
        validate="one_to_one",
    )

    value_fields = [column for column in VALUE_COLUMNS if column in values.columns]
    values = values[value_fields].copy()
    values["season"] = pd.to_numeric(values["season"], errors="raise").astype("int64").astype("string")
    values["league_code"] = values["league_code"].astype("string")
    values["transfermarkt_player_id"] = pd.to_numeric(
        values["transfermarkt_player_id"], errors="raise"
    ).astype("int64")

    modeling = matched.merge(
        values,
        on=["transfermarkt_player_id", "season", "league_code"],
        how="left",
        validate="many_to_one",
    )
    with_values = modeling.loc[modeling["market_value_eur"].notna()].copy()
    missing_values = len(modeling) - len(with_values)
    return with_values.reset_index(drop=True), missing_values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fbref", type=Path, default=DEFAULT_FBREF)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--values", type=Path, default=DEFAULT_VALUES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    for path, label in [
        (args.fbref, "FBref table"),
        (args.crosswalk, "player crosswalk"),
        (args.values, "season values"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    modeling, missing_values = build_modeling_data(
        pd.read_csv(args.fbref),
        pd.read_csv(args.crosswalk),
        pd.read_csv(args.values),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    modeling.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(
        f"Wrote {len(modeling):,} modeling rows to {args.output} "
        f"({missing_values:,} accepted matches had no season value)"
    )


if __name__ == "__main__":
    main()
