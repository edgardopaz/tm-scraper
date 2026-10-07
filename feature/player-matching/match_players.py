"""
Match FBref player-seasons to Transfermarkt identities.

    python feature/player-matching/match_players.py

Unique exact names are accepted automatically. Everything else is written to a
ranked review file. Existing decisions in data/manual/player_matches.csv are
applied first and are never overwritten.
"""

import argparse
from pathlib import Path

import pandas as pd

from matching import (
    DATA_DIR,
    MANUAL_COLUMNS,
    MANUAL_PATH,
    REPORT_DIR,
    load_manual_matches,
    match_identities,
)

DEFAULT_FBREF = DATA_DIR / "fbref_players.csv"
DEFAULT_TRANSFERMARKT = DATA_DIR / "transfermarkt_players.csv"
DEFAULT_CROSSWALK = DATA_DIR / "player_crosswalk.csv"


def write_outputs(result, crosswalk_path, report_dir, manual_path):
    report_dir.mkdir(parents=True, exist_ok=True)
    crosswalk_path.parent.mkdir(parents=True, exist_ok=True)
    manual_path.parent.mkdir(parents=True, exist_ok=True)

    result["crosswalk"].to_csv(crosswalk_path, index=False)
    result["exact"].to_csv(report_dir / "matched_exact.csv", index=False)
    result["review"].to_csv(report_dir / "manual_review.csv", index=False)
    result["unmatched"].to_csv(report_dir / "unmatched.csv", index=False)
    result["summary"].to_csv(report_dir / "match_summary.csv", index=False)

    if not manual_path.exists():
        pd.DataFrame(columns=MANUAL_COLUMNS).to_csv(manual_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fbref", type=Path, default=DEFAULT_FBREF)
    parser.add_argument("--transfermarkt", type=Path, default=DEFAULT_TRANSFERMARKT)
    parser.add_argument("--manual", type=Path, default=MANUAL_PATH)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of ranked Transfermarkt candidates to keep for review.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1.")
    if not args.fbref.exists():
        raise FileNotFoundError(f"FBref table not found: {args.fbref}")
    if not args.transfermarkt.exists():
        raise FileNotFoundError(
            f"Transfermarkt roster not found: {args.transfermarkt}"
        )

    result = match_identities(
        pd.read_csv(args.fbref),
        pd.read_csv(args.transfermarkt),
        manual=load_manual_matches(args.manual),
        top_n=args.top_n,
    )
    write_outputs(result, args.crosswalk, args.report_dir, args.manual)

    summary = dict(zip(result["summary"]["metric"], result["summary"]["value"]))
    print(f"FBref rows: {int(summary['fbref_rows']):,}")
    print(f"Exact matches: {int(summary['exact_matches']):,}")
    print(f"Score matches: {int(summary['score_matches']):,}")
    print(f"Manual matches: {int(summary['manual_matches']):,}")
    print(f"Needs review: {int(summary['review_rows']):,}")
    print(f"Unmatched: {int(summary['unmatched_rows']):,}")
    print(f"Accepted coverage: {summary['accepted_coverage']:.1%}")
    print(f"Wrote crosswalk to {args.crosswalk}")
    print(f"Wrote review reports to {args.report_dir}")
    print(
        "Add accepted IDs to "
        f"{args.manual} and rerun to keep those decisions."
    )


if __name__ == "__main__":
    main()
