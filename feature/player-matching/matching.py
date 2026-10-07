"""
Shared helpers for matching FBref player-seasons to Transfermarkt identities.

Automatic matches require a unique exact name after formatting normalization,
plus the same birth year, season, and league. Similarity scores are only used
to rank candidates for manual review.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = DATA_DIR / "reports" / "player_matching"
MANUAL_PATH = DATA_DIR / "manual" / "player_matches.csv"

LEAGUE_MAP = {
    "ENG-Premier League": "GB1",
    "ESP-La Liga": "ES1",
    "ITA-Serie A": "IT1",
    "GER-Bundesliga": "L1",
    "FRA-Ligue 1": "FR1",
}

FBREF_KEY = [
    "fbref_player",
    "birth_year",
    "season",
    "league_code",
    "fbref_team",
]
MANUAL_COLUMNS = FBREF_KEY + ["transfermarkt_player_id", "status", "notes"]
SCORE_MATCH_THRESHOLD = 0.7
CROSSWALK_COLUMNS = FBREF_KEY + [
    "fbref_row_id",
    "transfermarkt_player_id",
    "transfermarkt_player_name",
    "tm_club",
    "match_method",
]

_PUNCT_TO_SPACE = str.maketrans(
    {
        "-": " ",
        "–": " ",
        "—": " ",
        "'": "",
        "’": "",
        "`": "",
        "´": "",
        ".": " ",
        ",": " ",
        "&": " ",
        "/": " ",
    }
)


def normalize_name(value):
    """Lowercase a name and standardize apostrophes, hyphens, and spacing."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).translate(_PUNCT_TO_SPACE).lower()
    return re.sub(r"\s+", " ", text).strip()


def fold_name(value):
    """Normalize formatting and strip accents for ranking only."""
    text = normalize_name(value)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def name_similarity(left, right):
    """Return the better of raw and accent-folded name similarity."""
    raw_left, raw_right = normalize_name(left), normalize_name(right)
    if not raw_left or not raw_right:
        return 0.0
    raw = SequenceMatcher(None, raw_left, raw_right).ratio()
    folded = SequenceMatcher(None, fold_name(left), fold_name(right)).ratio()
    return max(raw, folded)


def club_similarity(left, right):
    """Score club names with equality, containment, then token overlap."""
    left_key, right_key = fold_name(left), fold_name(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    if left_key in right_key or right_key in left_key:
        return 0.9
    left_tokens, right_tokens = set(left_key.split()), set(right_key.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def combined_score(fbref_name, tm_name, fbref_club, tm_club):
    return round(
        0.8 * name_similarity(fbref_name, tm_name)
        + 0.2 * club_similarity(fbref_club, tm_club),
        4,
    )


def parse_birth_year(series):
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def parse_season(series):
    return pd.to_numeric(series, errors="raise").astype("int64").astype("string")


def map_league(series):
    mapped = series.map(LEAGUE_MAP)
    unknown = sorted(series[mapped.isna()].dropna().unique())
    if unknown:
        raise ValueError(f"Unmapped FBref league(s): {unknown}")
    return mapped.astype("string")


def fbref_row_id(row):
    return "|".join(
        [
            str(row["fbref_player"]),
            str(row["birth_year"]),
            str(row["season"]),
            str(row["league_code"]),
            str(row["fbref_team"]),
        ]
    )


def prepare_fbref(df):
    """Standardize FBref player-seasons for matching."""
    required = {"player", "team", "league", "season", "born_"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"FBref table is missing fields: {sorted(missing)}")

    out = df.copy()
    out["fbref_player"] = out["player"].astype("string")
    out["fbref_team"] = out["team"].astype("string")
    out["season"] = parse_season(out["season"])
    out["league_code"] = map_league(out["league"])
    out["birth_year"] = parse_birth_year(out["born_"])
    out["name_key"] = out["fbref_player"].map(normalize_name)
    out["fbref_row_id"] = out.apply(fbref_row_id, axis=1)
    return out


def prepare_transfermarkt(df):
    """Collapse Transfermarkt roster rows to one identity per player-season-league."""
    required = {
        "transfermarkt_player_id",
        "player_name",
        "birth_year",
        "club_name",
        "league_code",
        "season",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Transfermarkt roster is missing fields: {sorted(missing)}")

    out = df.copy()
    out["transfermarkt_player_id"] = pd.to_numeric(
        out["transfermarkt_player_id"], errors="raise"
    ).astype("int64")
    out["player_name"] = out["player_name"].astype("string")
    out["season"] = parse_season(out["season"])
    out["league_code"] = out["league_code"].astype("string")
    out["birth_year"] = parse_birth_year(out["birth_year"])
    out["name_key"] = out["player_name"].map(normalize_name)
    aggregations = {
        "player_name": "last",
        "name_key": "last",
        "birth_year": "first",
        "club_name": lambda values: " | ".join(sorted(set(map(str, values.dropna())))),
    }
    if "position" in out.columns:
        aggregations["position"] = "last"

    grouped = (
        out.groupby(["transfermarkt_player_id", "season", "league_code"], as_index=False)
        .agg(aggregations)
        .rename(columns={"club_name": "tm_club", "position": "tm_position"})
    )
    return grouped


def load_manual_matches(path=MANUAL_PATH):
    """Load reviewed matches. Missing files are treated as empty."""
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=MANUAL_COLUMNS)

    manual = pd.read_csv(path)
    if manual.empty:
        return pd.DataFrame(columns=MANUAL_COLUMNS)

    required = {"fbref_player", "birth_year", "season", "league_code", "transfermarkt_player_id"}
    missing = required.difference(manual.columns)
    if missing:
        raise ValueError(f"Manual match file is missing fields: {sorted(missing)}")

    manual = manual.copy()
    manual["fbref_player"] = manual["fbref_player"].astype("string")
    manual["season"] = parse_season(manual["season"])
    manual["league_code"] = manual["league_code"].astype("string")
    manual["birth_year"] = parse_birth_year(manual["birth_year"])
    if "fbref_team" not in manual.columns:
        manual["fbref_team"] = pd.Series(pd.NA, index=manual.index, dtype="string")
    else:
        manual["fbref_team"] = manual["fbref_team"].astype("string")
    if "notes" not in manual.columns:
        manual["notes"] = pd.Series(pd.NA, index=manual.index, dtype="string")
    manual["transfermarkt_player_id"] = pd.to_numeric(
        manual["transfermarkt_player_id"], errors="coerce"
    ).astype("Int64")
    if "status" not in manual.columns:
        manual["status"] = pd.Series(pd.NA, index=manual.index, dtype="string")
    manual["status"] = manual["status"].astype("string")
    missing_status = manual["status"].isna() | (manual["status"] == "")
    manual.loc[missing_status & manual["transfermarkt_player_id"].isna(), "status"] = "unmatched"
    manual.loc[missing_status & manual["transfermarkt_player_id"].notna(), "status"] = "match"
    return manual[MANUAL_COLUMNS]


def _candidate_record(fb_row, tm_row, rank):
    score = combined_score(
        fb_row.fbref_player,
        tm_row.player_name,
        fb_row.fbref_team,
        tm_row.tm_club,
    )
    prefix = f"candidate_{rank}"
    return {
        f"{prefix}_id": int(tm_row.transfermarkt_player_id),
        f"{prefix}_name": tm_row.player_name,
        f"{prefix}_club": tm_row.tm_club,
        f"{prefix}_score": score,
    }


def _blank_candidates(top_n):
    blank = {}
    for rank in range(1, top_n + 1):
        blank[f"candidate_{rank}_id"] = pd.NA
        blank[f"candidate_{rank}_name"] = pd.NA
        blank[f"candidate_{rank}_club"] = pd.NA
        blank[f"candidate_{rank}_score"] = pd.NA
    return blank


def _review_row(fb_row, candidates, reason, top_n):
    ranked = sorted(
        candidates,
        key=lambda tm_row: (
            combined_score(
                fb_row.fbref_player,
                tm_row.player_name,
                fb_row.fbref_team,
                tm_row.tm_club,
            ),
            tm_row.player_name,
        ),
        reverse=True,
    )
    row = {
        "fbref_row_id": fb_row.fbref_row_id,
        "fbref_player": fb_row.fbref_player,
        "birth_year": fb_row.birth_year,
        "season": fb_row.season,
        "league_code": fb_row.league_code,
        "fbref_team": fb_row.fbref_team,
        "review_reason": reason,
    }
    row.update(_blank_candidates(top_n))
    for rank, tm_row in enumerate(ranked[:top_n], start=1):
        row.update(_candidate_record(fb_row, tm_row, rank))
    return row


def _crosswalk_row(fb_row, tm_row, method):
    return {
        "fbref_player": fb_row.fbref_player,
        "birth_year": fb_row.birth_year,
        "season": fb_row.season,
        "league_code": fb_row.league_code,
        "fbref_team": fb_row.fbref_team,
        "fbref_row_id": fb_row.fbref_row_id,
        "transfermarkt_player_id": int(tm_row.transfermarkt_player_id),
        "transfermarkt_player_name": tm_row.player_name,
        "tm_club": tm_row.tm_club,
        "match_method": method,
    }


def _block_key(birth_year, season, league_code):
    year = int(birth_year) if pd.notna(birth_year) else None
    return (year, str(season), str(league_code))


def _manual_status(decision):
    status = getattr(decision, "status", None)
    if status is not None and not pd.isna(status) and str(status).strip():
        return str(status).strip()
    if pd.isna(decision.transfermarkt_player_id):
        return "unmatched"
    return "match"


def _apply_manual(fbref, transfermarkt, manual):
    """Return matched IDs, unmatched IDs, review holds, and crosswalk rows."""
    if manual.empty:
        return set(), set(), set(), []

    tm_by_id = {}
    for row in transfermarkt.itertuples(index=False):
        player_id = int(row.transfermarkt_player_id)
        tm_by_id[(player_id, str(row.season), str(row.league_code))] = row
        tm_by_id[player_id] = row
    matched_ids = set()
    unmatched_ids = set()
    held_ids = set()
    crosswalk_rows = []

    for decision in manual.itertuples(index=False):
        mask = (
            (fbref["fbref_player"] == decision.fbref_player)
            & (fbref["season"] == decision.season)
            & (fbref["league_code"] == decision.league_code)
        )
        if pd.notna(decision.birth_year):
            mask &= fbref["birth_year"] == decision.birth_year
        if pd.notna(decision.fbref_team):
            mask &= fbref["fbref_team"] == decision.fbref_team

        targets = fbref.loc[mask]
        if targets.empty:
            continue

        status = _manual_status(decision)
        if status == "review":
            held_ids.update(targets["fbref_row_id"])
            continue

        if status == "unmatched" or pd.isna(decision.transfermarkt_player_id):
            unmatched_ids.update(targets["fbref_row_id"])
            continue

        player_id = int(decision.transfermarkt_player_id)
        tm_row = tm_by_id.get(
            (player_id, str(decision.season), str(decision.league_code)),
            tm_by_id.get(player_id),
        )
        if tm_row is None:
            unmatched_ids.update(targets["fbref_row_id"])
            continue

        for fb_row in targets.itertuples(index=False):
            matched_ids.add(fb_row.fbref_row_id)
            crosswalk_rows.append(_crosswalk_row(fb_row, tm_row, "manual"))

    return matched_ids, unmatched_ids, held_ids, crosswalk_rows


def match_identities(fbref, transfermarkt, manual=None, top_n=3):
    """
    Match FBref rows to Transfermarkt identities.

    Returns exact matches, review rows, unmatched rows, and the accepted crosswalk.
    """
    fbref = prepare_fbref(fbref)
    transfermarkt = prepare_transfermarkt(transfermarkt)
    manual = load_manual_matches(None) if manual is None else manual
    if not isinstance(manual, pd.DataFrame):
        manual = load_manual_matches(manual)

    resolved, manual_unmatched, held_ids, crosswalk_rows = _apply_manual(
        fbref, transfermarkt, manual
    )
    pending = fbref.loc[~fbref["fbref_row_id"].isin(resolved | manual_unmatched)].copy()

    exact_keys = ["name_key", "birth_year", "season", "league_code"]
    exact_tm = transfermarkt.dropna(subset=["birth_year"]).drop_duplicates(
        exact_keys + ["transfermarkt_player_id"]
    )
    merged = pending.merge(
        exact_tm,
        on=exact_keys,
        how="left",
        suffixes=("", "_tm"),
    )

    id_counts = (
        merged.dropna(subset=["transfermarkt_player_id"])
        .groupby("fbref_row_id")["transfermarkt_player_id"]
        .nunique()
    )
    unique_ids = set(id_counts[id_counts == 1].index)
    ambiguous_ids = set(id_counts[id_counts > 1].index)

    exact_rows = []
    review_rows = []
    unmatched_rows = []

    unique_merged = merged.loc[
        merged["fbref_row_id"].isin(unique_ids)
        & merged["transfermarkt_player_id"].notna()
    ].drop_duplicates("fbref_row_id")
    fb_by_id = {row.fbref_row_id: row for row in pending.itertuples(index=False)}
    for row in unique_merged.itertuples(index=False):
        fb_row = fb_by_id[row.fbref_row_id]
        tm_row = row
        exact_rows.append(fb_row.fbref_row_id)
        crosswalk_rows.append(_crosswalk_row(fb_row, tm_row, "exact"))

    blocks = {}
    for row_key, group in transfermarkt.groupby(
        ["birth_year", "season", "league_code"], dropna=False
    ):
        blocks[_block_key(*row_key)] = group

    name_blocks = {}
    for row_key, group in transfermarkt.groupby(
        ["name_key", "season", "league_code"], dropna=False
    ):
        name_blocks[(row_key[0], str(row_key[1]), str(row_key[2]))] = group

    remaining = pending.loc[~pending["fbref_row_id"].isin(unique_ids)]
    for fb_row in remaining.itertuples(index=False):
        if fb_row.fbref_row_id in ambiguous_ids:
            candidates = merged.loc[
                merged["fbref_row_id"] == fb_row.fbref_row_id,
                ["transfermarkt_player_id", "player_name", "tm_club"],
            ].drop_duplicates("transfermarkt_player_id")
            review_rows.append(
                _review_row(
                    fb_row,
                    candidates.itertuples(index=False),
                    "multiple_exact_ids",
                    top_n,
                )
            )
            continue

        if pd.isna(fb_row.birth_year):
            candidates = name_blocks.get(
                (fb_row.name_key, str(fb_row.season), str(fb_row.league_code)),
                pd.DataFrame(columns=transfermarkt.columns),
            )
            if candidates.empty:
                unmatched_rows.append(
                    {
                        "fbref_row_id": fb_row.fbref_row_id,
                        "fbref_player": fb_row.fbref_player,
                        "birth_year": fb_row.birth_year,
                        "season": fb_row.season,
                        "league_code": fb_row.league_code,
                        "fbref_team": fb_row.fbref_team,
                        "unmatched_reason": "missing_birth_year",
                    }
                )
            else:
                review_rows.append(
                    _review_row(
                        fb_row,
                        candidates.itertuples(index=False),
                        "missing_birth_year",
                        top_n,
                    )
                )
            continue

        candidates = blocks.get(
            _block_key(fb_row.birth_year, fb_row.season, fb_row.league_code),
            pd.DataFrame(columns=transfermarkt.columns),
        )
        if candidates.empty:
            unmatched_rows.append(
                {
                    "fbref_row_id": fb_row.fbref_row_id,
                    "fbref_player": fb_row.fbref_player,
                    "birth_year": fb_row.birth_year,
                    "season": fb_row.season,
                    "league_code": fb_row.league_code,
                    "fbref_team": fb_row.fbref_team,
                    "unmatched_reason": "no_candidates",
                }
            )
            continue

        ranked_candidates = list(candidates.itertuples(index=False))
        review_row = _review_row(fb_row, ranked_candidates, "name_not_exact", top_n)
        top_score = review_row.get("candidate_1_score")
        if (
            fb_row.fbref_row_id not in held_ids
            and pd.notna(top_score)
            and top_score >= SCORE_MATCH_THRESHOLD
        ):
            top_id = int(review_row["candidate_1_id"])
            top_candidate = next(
                candidate
                for candidate in ranked_candidates
                if int(candidate.transfermarkt_player_id) == top_id
            )
            crosswalk_rows.append(_crosswalk_row(fb_row, top_candidate, "score"))
            continue

        review_rows.append(review_row)

    for fbref_id in sorted(manual_unmatched):
        fb_row = fbref.loc[fbref["fbref_row_id"] == fbref_id].iloc[0]
        unmatched_rows.append(
            {
                "fbref_row_id": fb_row.fbref_row_id,
                "fbref_player": fb_row.fbref_player,
                "birth_year": fb_row.birth_year,
                "season": fb_row.season,
                "league_code": fb_row.league_code,
                "fbref_team": fb_row.fbref_team,
                "unmatched_reason": "manual_unmatched",
            }
        )

    crosswalk = pd.DataFrame(crosswalk_rows, columns=CROSSWALK_COLUMNS)
    exact = fbref.loc[fbref["fbref_row_id"].isin(exact_rows), FBREF_KEY + ["fbref_row_id"]]
    if not crosswalk.empty:
        exact = exact.merge(
            crosswalk.loc[crosswalk["match_method"] == "exact"],
            on=FBREF_KEY + ["fbref_row_id"],
            how="left",
        )
    exact = exact.reset_index(drop=True)

    review = pd.DataFrame(review_rows)
    unmatched = pd.DataFrame(unmatched_rows)
    if not crosswalk.empty:
        crosswalk = (
            crosswalk.drop_duplicates("fbref_row_id", keep="last")
            .sort_values(["season", "league_code", "fbref_player"])
            .reset_index(drop=True)
        )

    summary = pd.DataFrame(
        [
            {"metric": "fbref_rows", "value": len(fbref)},
            {"metric": "exact_matches", "value": int((crosswalk["match_method"] == "exact").sum()) if not crosswalk.empty else 0},
            {"metric": "score_matches", "value": int((crosswalk["match_method"] == "score").sum()) if not crosswalk.empty else 0},
            {"metric": "manual_matches", "value": int((crosswalk["match_method"] == "manual").sum()) if not crosswalk.empty else 0},
            {"metric": "review_rows", "value": len(review)},
            {"metric": "unmatched_rows", "value": len(unmatched)},
            {
                "metric": "accepted_coverage",
                "value": round(len(crosswalk) / len(fbref), 4) if len(fbref) else 0,
            },
        ]
    )
    return {
        "fbref": fbref,
        "exact": exact,
        "review": review,
        "unmatched": unmatched,
        "crosswalk": crosswalk,
        "summary": summary,
    }
