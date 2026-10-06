"""
Collect Transfermarkt player identities from Big Five league squad pages.

The default run covers the same five seasons as the FBref scraper:

    python feature/transfermarkt-values/collect_rosters.py

Use filters for a smaller run:

    python feature/transfermarkt-values/collect_rosters.py \
        --league L1 --season 2223
"""

import argparse
from pathlib import Path

import pandas as pd

from transfermarkt import (
    DATA_DIR,
    RAW_DIR,
    ROSTER_COLUMNS,
    TransfermarktClient,
    competition_url,
    parse_competition_clubs,
    parse_squad,
    squad_url,
)

LEAGUES = {
    "GB1": "premier-league",
    "ES1": "laliga",
    "IT1": "serie-a",
    "L1": "bundesliga",
    "FR1": "ligue-1",
}
SEASONS = ["2122", "2223", "2324", "2425", "2526"]
DEFAULT_OUTPUT = DATA_DIR / "transfermarkt_players.csv"


def season_start_year(season):
    """Convert the project's 2223 season label to Transfermarkt's 2022."""
    if len(season) != 4 or not season.isdigit():
        raise ValueError(f"Invalid season label: {season}")

    start = int(season[:2])
    end = int(season[2:])
    if end != (start + 1) % 100:
        raise ValueError(f"Season label is not consecutive: {season}")

    return 2000 + start


def collect_rosters(
    client,
    league_codes,
    seasons,
    refresh=False,
    max_clubs=None,
):
    """Collect roster rows for the requested leagues and seasons."""
    frames = []

    for season in seasons:
        start_year = season_start_year(season)
        for league_code in league_codes:
            league_slug = LEAGUES[league_code]
            cache_dir = RAW_DIR / "rosters" / season / league_code
            competition_path = cache_dir / "competition.html"

            competition_page = client.get_html(
                competition_url(league_slug, league_code, start_year),
                cache_path=competition_path,
                refresh=refresh,
            )
            clubs = parse_competition_clubs(competition_page)
            if len(clubs) < 15:
                raise ValueError(
                    f"Expected at least 15 clubs for {league_code} {season}, "
                    f"found {len(clubs)}."
                )

            if max_clubs is not None:
                clubs = clubs[:max_clubs]

            print(f"{season} {league_code}: collecting {len(clubs)} clubs")
            for number, club in enumerate(clubs, start=1):
                club_id = club["transfermarkt_club_id"]
                club_name = club["club_name"]
                squad_path = cache_dir / f"{club_id}.html"

                squad_page = client.get_html(
                    squad_url(club_id, start_year),
                    cache_path=squad_path,
                    refresh=refresh,
                )
                squad = parse_squad(
                    squad_page,
                    club_id=club_id,
                    club_name=club_name,
                    league_code=league_code,
                    season=season,
                )
                frames.append(squad)
                print(
                    f"  {number}/{len(clubs)} {club_name}: "
                    f"{len(squad)} players"
                )

    if not frames:
        return pd.DataFrame(columns=ROSTER_COLUMNS)

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(
            [
                "transfermarkt_player_id",
                "transfermarkt_club_id",
                "league_code",
                "season",
            ]
        )
        .sort_values(["season", "league_code", "club_name", "player_name"])
        .reset_index(drop=True)
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league",
        dest="leagues",
        action="append",
        choices=LEAGUES,
        help="League code to collect. Repeat for multiple leagues; default is all.",
    )
    parser.add_argument(
        "--season",
        dest="seasons",
        action="append",
        choices=SEASONS,
        help="Season label to collect. Repeat for multiple seasons; default is all.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload pages even when cached copies exist.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Minimum seconds between live requests (default: 2).",
    )
    parser.add_argument(
        "--max-clubs",
        type=int,
        help="Limit clubs per league-season for a small validation run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Roster CSV path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.delay < 0:
        raise ValueError("--delay cannot be negative.")
    if args.max_clubs is not None and args.max_clubs < 1:
        raise ValueError("--max-clubs must be at least 1.")

    client = TransfermarktClient(min_interval=args.delay)
    roster = collect_rosters(
        client,
        league_codes=args.leagues or list(LEAGUES),
        seasons=args.seasons or SEASONS,
        refresh=args.refresh,
        max_clubs=args.max_clubs,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    roster.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    unique_players = roster["transfermarkt_player_id"].nunique()
    print(
        f"Wrote {len(roster):,} roster rows for "
        f"{unique_players:,} unique players to {args.output}"
    )


if __name__ == "__main__":
    main()
