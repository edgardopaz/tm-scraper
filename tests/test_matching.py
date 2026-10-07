import pandas as pd

from matching import match_identities, normalize_name


def test_normalize_name_standardizes_formatting():
    assert normalize_name("  Jean-Clair Todibo ") == "jean clair todibo"
    assert normalize_name("O'Brien") == "obrien"
    assert normalize_name("Aarón") == "aarón"


def test_unique_exact_name_is_accepted():
    fbref = pd.DataFrame(
        [
            {
                "player": "Jude Bellingham",
                "team": "Dortmund",
                "league": "GER-Bundesliga",
                "season": "2223",
                "born_": 2003,
            }
        ]
    )
    transfermarkt = pd.DataFrame(
        [
            {
                "transfermarkt_player_id": 581678,
                "player_name": "Jude Bellingham",
                "birth_year": 2003,
                "club_name": "Borussia Dortmund",
                "league_code": "L1",
                "season": "2223",
                "position": "Central Midfield",
            }
        ]
    )

    result = match_identities(fbref, transfermarkt)

    assert len(result["exact"]) == 1
    assert result["review"].empty
    assert result["unmatched"].empty
    assert result["crosswalk"].iloc[0]["transfermarkt_player_id"] == 581678
    assert result["crosswalk"].iloc[0]["match_method"] == "exact"


def test_high_score_name_difference_is_accepted():
    fbref = pd.DataFrame(
        [
            {
                "player": "Aaron Escandell",
                "team": "Granada",
                "league": "ESP-La Liga",
                "season": "2122",
                "born_": 1995,
            }
        ]
    )
    transfermarkt = pd.DataFrame(
        [
            {
                "transfermarkt_player_id": 284430,
                "player_name": "Aarón Escandell",
                "birth_year": 1995,
                "club_name": "Granada CF",
                "league_code": "ES1",
                "season": "2122",
                "position": "Goalkeeper",
            },
            {
                "transfermarkt_player_id": 1,
                "player_name": "Unrelated Player",
                "birth_year": 1995,
                "club_name": "Valencia CF",
                "league_code": "ES1",
                "season": "2122",
                "position": "Midfield",
            },
        ]
    )

    result = match_identities(fbref, transfermarkt)

    assert result["exact"].empty
    assert result["review"].empty
    assert result["crosswalk"].iloc[0]["match_method"] == "score"
    assert result["crosswalk"].iloc[0]["transfermarkt_player_id"] == 284430


def test_low_score_candidate_stays_in_review():
    fbref = pd.DataFrame(
        [
            {
                "player": "Alex",
                "team": "Arsenal",
                "league": "ENG-Premier League",
                "season": "2223",
                "born_": 2000,
            }
        ]
    )
    transfermarkt = pd.DataFrame(
        [
            {
                "transfermarkt_player_id": 10,
                "player_name": "Completely Different",
                "birth_year": 2000,
                "club_name": "Arsenal FC",
                "league_code": "GB1",
                "season": "2223",
                "position": "Forward",
            }
        ]
    )

    result = match_identities(fbref, transfermarkt)

    assert result["crosswalk"].empty
    assert result["review"].iloc[0]["candidate_1_score"] < 0.7


def test_duplicate_exact_ids_are_reviewed():
    fbref = pd.DataFrame(
        [
            {
                "player": "Vitinha",
                "team": "Marseille",
                "league": "FRA-Ligue 1",
                "season": "2223",
                "born_": 2000,
            }
        ]
    )
    transfermarkt = pd.DataFrame(
        [
            {
                "transfermarkt_player_id": 11,
                "player_name": "Vitinha",
                "birth_year": 2000,
                "club_name": "Marseille",
                "league_code": "FR1",
                "season": "2223",
                "position": "Forward",
            },
            {
                "transfermarkt_player_id": 22,
                "player_name": "Vitinha",
                "birth_year": 2000,
                "club_name": "Paris FC",
                "league_code": "FR1",
                "season": "2223",
                "position": "Midfield",
            },
        ]
    )

    result = match_identities(fbref, transfermarkt)

    assert result["exact"].empty
    assert result["review"].iloc[0]["review_reason"] == "multiple_exact_ids"
    assert {result["review"].iloc[0]["candidate_1_id"], result["review"].iloc[0]["candidate_2_id"]} == {11, 22}


def test_manual_match_is_applied_before_exact():
    fbref = pd.DataFrame(
        [
            {
                "player": "Example Player",
                "team": "Arsenal",
                "league": "ENG-Premier League",
                "season": "2223",
                "born_": 2001,
            }
        ]
    )
    transfermarkt = pd.DataFrame(
        [
            {
                "transfermarkt_player_id": 99,
                "player_name": "Example Player",
                "birth_year": 2001,
                "club_name": "Arsenal FC",
                "league_code": "GB1",
                "season": "2223",
                "position": "Forward",
            }
        ]
    )
    manual = pd.DataFrame(
        [
            {
                "fbref_player": "Example Player",
                "birth_year": 2001,
                "season": "2223",
                "league_code": "GB1",
                "fbref_team": "Arsenal",
                "transfermarkt_player_id": 99,
                "notes": "reviewed",
            }
        ]
    )

    result = match_identities(fbref, transfermarkt, manual=manual)

    assert result["exact"].empty
    assert result["crosswalk"].iloc[0]["match_method"] == "manual"
    assert result["review"].empty


def test_no_candidates_are_unmatched():
    fbref = pd.DataFrame(
        [
            {
                "player": "Unknown Player",
                "team": "Arsenal",
                "league": "ENG-Premier League",
                "season": "2223",
                "born_": 1990,
            }
        ]
    )
    transfermarkt = pd.DataFrame(
        [
            {
                "transfermarkt_player_id": 1,
                "player_name": "Someone Else",
                "birth_year": 1991,
                "club_name": "Arsenal FC",
                "league_code": "GB1",
                "season": "2223",
                "position": "Forward",
            }
        ]
    )

    result = match_identities(fbref, transfermarkt)

    assert result["exact"].empty
    assert result["review"].empty
    assert result["unmatched"].iloc[0]["unmatched_reason"] == "no_candidates"
