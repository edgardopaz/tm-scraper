"""
Shared Transfermarkt request, caching, and parsing helpers.

Running this file directly keeps Jude Bellingham as a small end-to-end example:

    python feature/transfermarkt-values/transfermarkt.py
    python feature/transfermarkt-values/transfermarkt.py --refresh
"""

import argparse
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://www.transfermarkt.com"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "transfermarkt"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0 Safari/537.36"
)

ROSTER_COLUMNS = [
    "transfermarkt_player_id",
    "player_name",
    "date_of_birth",
    "birth_year",
    "position",
    "nationality",
    "transfermarkt_club_id",
    "club_name",
    "league_code",
    "season",
]


class TransfermarktClient:
    """Small rate-limited client with filesystem caching and retries."""

    def __init__(self, min_interval=2.0, timeout=30, retries=2):
        self.min_interval = min_interval
        self.timeout = timeout
        self.retries = retries
        self._last_request_at = None

    def _wait_for_rate_limit(self):
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _download(self, url, accept):
        headers = {
            "Accept": accept,
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": USER_AGENT,
        }

        for attempt in range(self.retries + 1):
            self._wait_for_rate_limit()
            request = Request(url, headers=headers)
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                self._last_request_at = time.monotonic()
                return body
            except HTTPError as exc:
                self._last_request_at = time.monotonic()
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self.retries:
                    raise RuntimeError(
                        f"Transfermarkt returned HTTP {exc.code} for {url}"
                    ) from exc
            except (URLError, TimeoutError) as exc:
                self._last_request_at = time.monotonic()
                if attempt == self.retries:
                    raise RuntimeError(
                        f"Could not download Transfermarkt URL {url}: {exc}"
                    ) from exc

            time.sleep(2 ** (attempt + 1))

        raise RuntimeError(f"Could not download Transfermarkt URL {url}")

    def get_html(self, url, cache_path, refresh=False):
        """Return cached HTML, downloading and saving it when necessary."""
        if cache_path.exists() and not refresh:
            return cache_path.read_text(encoding="utf-8")

        body = self._download(url, "text/html")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(body)
        return body.decode("utf-8", errors="replace")

    def get_json(self, url, cache_path, refresh=False):
        """Return cached JSON, downloading and saving it when necessary."""
        if cache_path.exists() and not refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        body = self._download(url, "application/json")
        payload = json.loads(body)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return payload


def competition_url(league_slug, league_code, season_start_year):
    return (
        f"{BASE_URL}/{league_slug}/startseite/wettbewerb/{league_code}"
        f"/plus/?saison_id={season_start_year}"
    )


def squad_url(club_id, season_start_year):
    return (
        f"{BASE_URL}/-/kader/verein/{club_id}"
        f"/saison_id/{season_start_year}/plus/1"
    )


def history_url(player_id):
    return f"{BASE_URL}/ceapi/marketValueDevelopment/graph/{player_id}"


def parse_competition_clubs(page):
    """Extract stable club IDs and names from a competition page."""
    soup = BeautifulSoup(page, "html.parser")
    clubs = {}

    for anchor in soup.select('a[href*="/startseite/verein/"]'):
        match = re.search(r"/verein/(\d+)", anchor.get("href", ""))
        name = anchor.get("title") or anchor.get_text(" ", strip=True)
        if match and name:
            clubs[int(match.group(1))] = name

    if not clubs:
        raise ValueError("No clubs were found on the Transfermarkt competition page.")

    return [
        {"transfermarkt_club_id": club_id, "club_name": clubs[club_id]}
        for club_id in sorted(clubs)
    ]


def parse_squad(page, club_id, club_name, league_code, season):
    """Extract player identities from one detailed squad page."""
    soup = BeautifulSoup(page, "html.parser")
    players = []

    for row in soup.select("table.items > tbody > tr"):
        player_link = row.select_one('a[href*="/profil/spieler/"]')
        if player_link is None:
            continue

        match = re.search(r"/spieler/(\d+)", player_link.get("href", ""))
        cells = row.find_all("td", recursive=False)
        if not match or len(cells) < 4:
            continue

        player_name = player_link.get_text(" ", strip=True)
        identity_rows = cells[1].select("table.inline-table tr")
        position = (
            identity_rows[1].get_text(" ", strip=True)
            if len(identity_rows) > 1
            else None
        )

        birth_text = cells[2].get_text(" ", strip=True)
        birth_match = re.search(r"\d{2}/\d{2}/\d{4}", birth_text)
        birth_date = (
            pd.to_datetime(
                birth_match.group(0), format="%d/%m/%Y", errors="raise"
            ).date()
            if birth_match
            else None
        )

        nationalities = [
            image.get("title")
            for image in cells[3].select("img[title]")
            if image.get("title")
        ]

        players.append(
            {
                "transfermarkt_player_id": int(match.group(1)),
                "player_name": player_name,
                "date_of_birth": birth_date,
                "birth_year": birth_date.year if birth_date else None,
                "position": position,
                "nationality": "|".join(dict.fromkeys(nationalities)),
                "transfermarkt_club_id": int(club_id),
                "club_name": club_name,
                "league_code": league_code,
                "season": season,
            }
        )

    if not players:
        raise ValueError(
            f"No players were found for Transfermarkt club {club_id} ({club_name})."
        )

    return pd.DataFrame(players, columns=ROSTER_COLUMNS).drop_duplicates(
        ["transfermarkt_player_id", "transfermarkt_club_id", "season"]
    )


def load_history(client, player_id, refresh=False):
    """Load one player's complete market-value history."""
    cache_path = RAW_DIR / "history" / f"{player_id}.json"
    return client.get_json(
        history_url(player_id),
        cache_path=cache_path,
        refresh=refresh,
    )


def normalize_history(payload, player_id, player_name):
    """Convert Transfermarkt's graph response into one row per valuation."""
    history = payload.get("list")
    if not isinstance(history, list) or not history:
        raise ValueError("Transfermarkt response did not contain valuation history.")

    raw = pd.DataFrame(history)
    required = {"datum_mw", "y", "verein", "age"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Transfermarkt response is missing fields: {sorted(missing)}")

    values = pd.DataFrame(
        {
            "transfermarkt_player_id": int(player_id),
            "player_name": player_name,
            "valuation_date": pd.to_datetime(
                raw["datum_mw"], format="%d/%m/%Y", errors="raise"
            ),
            "market_value_eur": pd.to_numeric(raw["y"], errors="raise"),
            "club": raw["verein"],
            "age": pd.to_numeric(raw["age"], errors="coerce").astype("Int64"),
        }
    )

    return (
        values.sort_values("valuation_date")
        .drop_duplicates("valuation_date", keep="last")
        .reset_index(drop=True)
    )


def value_at_cutoff(values, cutoff):
    """Return the latest valuation available on or before a cutoff date."""
    eligible = values.loc[values["valuation_date"] <= cutoff]
    if eligible.empty:
        raise ValueError(f"No valuation exists on or before {cutoff.date()}.")
    return eligible.iloc[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the local cache and fetch a fresh response.",
    )
    args = parser.parse_args()

    player_id = 581678
    player_name = "Jude Bellingham"
    cutoff = pd.Timestamp("2023-05-27")
    output_path = DATA_DIR / "transfermarkt_bellingham_values.csv"

    client = TransfermarktClient()
    values = normalize_history(
        load_history(client, player_id, refresh=args.refresh),
        player_id=player_id,
        player_name=player_name,
    )
    values.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    target = value_at_cutoff(values, cutoff)
    print(f"Wrote {len(values)} valuation events to {output_path}")
    print(
        f"2022/23 target at {cutoff.date()}: "
        f"€{target['market_value_eur'] / 1_000_000:.1f}m "
        f"(published {target['valuation_date'].date()})"
    )


if __name__ == "__main__":
    main()
