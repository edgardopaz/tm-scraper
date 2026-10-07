"""
Collect complete Transfermarkt valuation histories for discovered players.

    python feature/transfermarkt-values/collect_history.py

Each raw response is cached by player ID, so interrupted runs can be restarted
without downloading completed players again.
"""

import argparse
from pathlib import Path

import pandas as pd

from transfermarkt import DATA_DIR, TransfermarktClient, load_history, normalize_history

DEFAULT_INPUT = DATA_DIR / "transfermarkt_players.csv"
DEFAULT_OUTPUT = DATA_DIR / "transfermarkt_value_history.csv"
DEFAULT_FAILURES = DATA_DIR / "transfermarkt_history_failures.csv"

HISTORY_COLUMNS = [
    "transfermarkt_player_id",
    "player_name",
    "valuation_date",
    "market_value_eur",
    "club",
    "age",
]
FAILURE_COLUMNS = ["transfermarkt_player_id", "player_name", "error"]


def unique_players(roster):
    """Return one current display name for each discovered player ID."""
    required = {"transfermarkt_player_id", "player_name", "season"}
    missing = required.difference(roster.columns)
    if missing:
        raise ValueError(f"Roster is missing required fields: {sorted(missing)}")

    players = (
        roster.sort_values("season")
        .drop_duplicates("transfermarkt_player_id", keep="last")
        [["transfermarkt_player_id", "player_name"]]
        .copy()
    )
    players["transfermarkt_player_id"] = pd.to_numeric(
        players["transfermarkt_player_id"], errors="raise"
    ).astype("int64")
    return players.sort_values("transfermarkt_player_id").reset_index(drop=True)


def combine_histories(frames):
    """Combine normalized histories into a stable, deduplicated table."""
    if not frames:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["transfermarkt_player_id", "valuation_date"], keep="last")
        .sort_values(["transfermarkt_player_id", "valuation_date"])
        .reset_index(drop=True)
    )


def write_checkpoint(frames, failures, output_path, failures_path, partial=False):
    """Write current successful histories and failure details."""
    history = combine_histories(frames)
    destination = (
        output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
        if partial
        else output_path
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(destination, index=False, date_format="%Y-%m-%d")

    failure_table = pd.DataFrame(failures, columns=FAILURE_COLUMNS)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failure_table.to_csv(failures_path, index=False)
    return history, destination


def collect_histories(
    client,
    players,
    output_path,
    failures_path,
    refresh=False,
    checkpoint_every=250,
):
    """Fetch, normalize, and periodically checkpoint player histories."""
    frames = []
    failures = []
    total = len(players)

    for completed, player in enumerate(players.itertuples(index=False), start=1):
        player_id = int(player.transfermarkt_player_id)
        player_name = player.player_name

        try:
            payload = load_history(client, player_id, refresh=refresh)
            history = normalize_history(
                payload,
                player_id=player_id,
                player_name=player_name,
            )
            frames.append(history)
        except (RuntimeError, TypeError, ValueError) as exc:
            failures.append(
                {
                    "transfermarkt_player_id": player_id,
                    "player_name": player_name,
                    "error": str(exc),
                }
            )
            print(f"  failed {player_id} {player_name}: {exc}")

        if completed % checkpoint_every == 0:
            history, destination = write_checkpoint(
                frames,
                failures,
                output_path,
                failures_path,
                partial=True,
            )
            print(
                f"Progress {completed:,}/{total:,}: "
                f"{history['transfermarkt_player_id'].nunique():,} players, "
                f"{len(failures):,} failures; checkpoint {destination}"
            )

    history, destination = write_checkpoint(
        frames,
        failures,
        output_path,
        failures_path,
        partial=False,
    )
    partial_path = output_path.with_name(
        f"{output_path.stem}.partial{output_path.suffix}"
    )
    if partial_path.exists():
        partial_path.unlink()

    return history, failures, destination


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Player roster CSV (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Combined history CSV (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=DEFAULT_FAILURES,
        help=f"Failure report CSV (default: {DEFAULT_FAILURES}).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload histories even when cached responses exist.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Minimum seconds between live requests (default: 2).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help="Write a partial combined CSV after this many players.",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        help="Limit players for a small validation run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.delay < 0:
        raise ValueError("--delay cannot be negative.")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1.")
    if args.max_players is not None and args.max_players < 1:
        raise ValueError("--max-players must be at least 1.")
    if not args.input.exists():
        raise FileNotFoundError(
            f"Roster not found: {args.input}. Run collect_rosters.py first."
        )

    roster = pd.read_csv(args.input)
    players = unique_players(roster)
    if args.max_players is not None:
        players = players.iloc[: args.max_players].copy()

    print(
        f"Collecting valuation histories for {len(players):,} unique players "
        f"with a {args.delay:g}s request interval"
    )
    client = TransfermarktClient(min_interval=args.delay)
    history, failures, destination = collect_histories(
        client,
        players,
        output_path=args.output,
        failures_path=args.failures,
        refresh=args.refresh,
        checkpoint_every=args.checkpoint_every,
    )

    successful_players = history["transfermarkt_player_id"].nunique()
    print(
        f"Wrote {len(history):,} valuation events for "
        f"{successful_players:,} players to {destination}"
    )
    print(f"Recorded {len(failures):,} failures in {args.failures}")


if __name__ == "__main__":
    main()
