from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup


FOX_URL = "https://www.foxsports.com.au/nrl/nrl-premiership/stats/players"

CURRENT_TEAMS_FILE = Path("current_team_lists.csv")
OUTPUT_FILE = Path("fox_player_stats.csv")

# Fox team abbreviations seen in the stats table.
FOX_TEAM_MAP = {
    "SOU": "Rabbitohs",
    "NEW": "Knights",
    "WAR": "Warriors",
    "DOL": "Dolphins",
    "CRO": "Sharks",
    "NQL": "Cowboys",
    "PEN": "Panthers",
    "SYD": "Roosters",

    # Other clubs included so this can later work across a full season.
    "BRI": "Broncos",
    "CBR": "Raiders",
    "CBY": "Bulldogs",
    "GLD": "Titans",
    "MAN": "Sea Eagles",
    "MEL": "Storm",
    "PAR": "Eels",
    "STG": "Dragons",
    "WST": "Wests Tigers",
}

FINALS_TEAMS = {
    "Rabbitohs",
    "Knights",
    "Warriors",
    "Dolphins",
    "Sharks",
    "Cowboys",
    "Panthers",
    "Roosters",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/152.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Referer": "https://www.foxsports.com.au/",
}

EXPECTED_HEADERS = [
    "Name",
    "G",
    "ST",
    "MIN",
    "T",
    "TA",
    "GLS",
    "GK%",
    "FG",
    "PTS",
    "R",
    "RM",
    "LB",
    "TB",
    "OFF",
    "K",
    "KM",
    "TCK",
    "MT",
    "ERR",
    "PEN",
    "SB",
    "SO",
]


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_number(value: str) -> str:
    value = clean_text(value).replace(",", "")

    if value in {"", "-", "—", "–"}:
        return ""

    return value


def load_current_players() -> dict[str, set[str]]:
    """
    Loads current_team_lists.csv so we can report how many current Finals
    players were matched by Fox Sports.

    The scraper does not depend on this file to fetch Fox stats, but using
    it gives us a useful coverage test.
    """
    result: dict[str, set[str]] = {}

    if not CURRENT_TEAMS_FILE.exists():
        print(
            f"[fox] {CURRENT_TEAMS_FILE} not found. "
            "Will scrape finals teams without squad-match testing."
        )
        return result

    with CURRENT_TEAMS_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            return result

        fields = {x.lower().strip(): x for x in reader.fieldnames}

        team_field = fields.get("team")
        player_field = fields.get("player")

        if not team_field or not player_field:
            print(
                "[fox] current_team_lists.csv does not have team/player columns. "
                "Skipping squad-match test."
            )
            return result

        for row in reader:
            team = clean_text(row.get(team_field, ""))
            player = clean_text(row.get(player_field, ""))

            if team and player:
                result.setdefault(team, set()).add(player)

    return result


def fetch_page(session: requests.Session, page: int) -> str:
    """
    Attempts a normal browser-style request.

    Fox currently serves 20 players per page, with roughly 27 pages.
    """
    params = {
        "page": page,
        "device": "DESKTOP",
        "editiondata": "none",
        "fromakamai": "true",
        "pt": "none",
    }

    response = session.get(
        FOX_URL,
        params=params,
        headers=HEADERS,
        timeout=30,
        allow_redirects=True,
    )

    print(
        f"[fox] Page {page}: "
        f"HTTP {response.status_code} "
        f"final_url={response.url}"
    )

    response.raise_for_status()

    url_lower = response.url.lower()
    text_lower = response.text.lower()

    anti_bot_markers = [
        "newskey/generator",
        "access denied",
        "captcha",
        "verify you are human",
        "akamai",
    ]

    if any(marker in url_lower or marker in text_lower for marker in anti_bot_markers):
        raise RuntimeError(
            "Fox Sports redirected the request into an anti-bot/challenge page."
        )

    return response.text


def find_stats_table(html: str):
    soup = BeautifulSoup(html, "html.parser")

    tables = soup.find_all("table")

    for table in tables:
        header_cells = table.find_all("th")
        headers = [clean_text(cell.get_text(" ", strip=True)) for cell in header_cells]

        if "Name" in headers and "G" in headers and "MIN" in headers:
            return table

    return None


def parse_player_name(raw: str):
    """
    Fox names commonly appear like:
        1 S. Drinkwater(NQL)
        14 N. Cleary(PEN)

    Returns:
        fox_name, fox_team_code
    """

    raw = clean_text(raw)

    # Remove ranking number at beginning.
    raw = re.sub(r"^\d+\s+", "", raw)

    match = re.search(r"^(.*?)\(([A-Z]{2,4})\)\s*$", raw)

    if not match:
        return raw, ""

    name = clean_text(match.group(1))
    team_code = clean_text(match.group(2)).upper()

    return name, team_code


def parse_table(table):
    headers = [
        clean_text(cell.get_text(" ", strip=True))
        for cell in table.find_all("th")
    ]

    rows = []

    tbody = table.find("tbody")
    source_rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]

    for tr in source_rows:
        cells = tr.find_all(["td", "th"])

        if not cells:
            continue

        values = [clean_text(cell.get_text(" ", strip=True)) for cell in cells]

        if len(values) < 5:
            continue

        # Sometimes the page structure can have more/less cells than headers.
        if len(values) < len(headers):
            values += [""] * (len(headers) - len(values))

        values = values[: len(headers)]

        row = dict(zip(headers, values))

        raw_name = row.get("Name", "")

        if not raw_name:
            continue

        player_name, fox_team_code = parse_player_name(raw_name)

        team = FOX_TEAM_MAP.get(fox_team_code, fox_team_code)

        if team not in FINALS_TEAMS:
            continue

        parsed = {
            "team": team,
            "fox_team_code": fox_team_code,
            "fox_player_name": player_name,
            "games": clean_number(row.get("G", "")),
            "starts": clean_number(row.get("ST", "")),
            "minutes": clean_number(row.get("MIN", "")),
            "tries": clean_number(row.get("T", "")),
            "try_assists": clean_number(row.get("TA", "")),
            "goals": clean_number(row.get("GLS", "")),
            "goal_kicking_pct": clean_number(row.get("GK%", "")),
            "field_goals": clean_number(row.get("FG", "")),
            "points": clean_number(row.get("PTS", "")),
            "runs": clean_number(row.get("R", "")),
            "run_metres": clean_number(row.get("RM", "")),
            "line_breaks": clean_number(row.get("LB", "")),
            "tackle_busts": clean_number(row.get("TB", "")),
            "offloads": clean_number(row.get("OFF", "")),
            "kicks": clean_number(row.get("K", "")),
            "kick_metres": clean_number(row.get("KM", "")),
            "tackles": clean_number(row.get("TCK", "")),
            "missed_tackles": clean_number(row.get("MT", "")),
            "errors": clean_number(row.get("ERR", "")),
            "penalties": clean_number(row.get("PEN", "")),
            "sin_bins": clean_number(row.get("SB", "")),
            "send_offs": clean_number(row.get("SO", "")),
        }

        rows.append(parsed)

    return rows


def normalise_name(name: str) -> str:
    name = clean_text(name).lower()

    name = (
        name.replace("'", "")
        .replace("-", " ")
        .replace(".", "")
    )

    name = re.sub(r"[^a-z0-9 ]+", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def possible_initial_surname(full_name: str) -> str:
    """
    Nathan Cleary -> n cleary
    Addin Fonua-Blake -> a fonua blake
    """
    bits = normalise_name(full_name).split()

    if not bits:
        return ""

    if len(bits) == 1:
        return bits[0]

    return f"{bits[0][0]} {' '.join(bits[1:])}"


def match_current_player(
    fox_name: str,
    team: str,
    current_players: dict[str, set[str]],
):
    if team not in current_players:
        return "", ""

    fox_norm = normalise_name(fox_name)

    for full_name in current_players[team]:
        if possible_initial_surname(full_name) == fox_norm:
            return full_name, "initial_surname"

    # Extra fallback using surname.
    fox_bits = fox_norm.split()

    if fox_bits:
        fox_surname = " ".join(fox_bits[1:]) if len(fox_bits) > 1 else fox_bits[-1]

        surname_matches = []

        for full_name in current_players[team]:
            full_bits = normalise_name(full_name).split()

            if not full_bits:
                continue

            surname = " ".join(full_bits[1:])

            if surname == fox_surname:
                surname_matches.append(full_name)

        if len(surname_matches) == 1:
            return surname_matches[0], "surname"

    return "", ""


def dedupe_rows(rows):
    best = {}

    for row in rows:
        key = (
            row["team"],
            normalise_name(row["fox_player_name"]),
        )

        best[key] = row

    return list(best.values())


def write_csv(rows):
    fields = [
        "team",
        "fox_team_code",
        "fox_player_name",
        "matched_player",
        "match_method",
        "games",
        "starts",
        "minutes",
        "tries",
        "try_assists",
        "goals",
        "goal_kicking_pct",
        "field_goals",
        "points",
        "runs",
        "run_metres",
        "line_breaks",
        "tackle_busts",
        "offloads",
        "kicks",
        "kick_metres",
        "tackles",
        "missed_tackles",
        "errors",
        "penalties",
        "sin_bins",
        "send_offs",
    ]

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in sorted(
            rows,
            key=lambda x: (
                x["team"],
                -int(x["games"] or 0),
                x["fox_player_name"],
            ),
        ):
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    print("[fox] Starting Fox Sports NRL player-stat test...")
    print("[fox] Finals teams only.")
    print("[fox] This script does NOT modify player_ratings.csv or predict.py.")

    current_players = load_current_players()

    if current_players:
        current_count = sum(
            len(players)
            for team, players in current_players.items()
            if team in FINALS_TEAMS
        )

        print(f"[fox] Loaded {current_count} current finals selections for matching")

    session = requests.Session()

    all_rows = []
    pages_with_data = 0

    # Fox currently shows about 535 players, 20 per page.
    # 30 gives us a little breathing room.
    for page in range(1, 31):
        try:
            html = fetch_page(session, page)
        except Exception as exc:
            print(f"[fox] ERROR fetching page {page}: {exc}")

            if page == 1:
                print()
                print("[fox] Fox blocked the very first request.")
                print("[fox] No predictor files have been changed.")
                sys.exit(2)

            break

        table = find_stats_table(html)

        if table is None:
            print(f"[fox] Page {page}: player stats table not found")

            if page == 1:
                print()
                print("[fox] Page loaded but Fox table was not present in raw HTML.")
                print("[fox] It may be JavaScript-rendered or protected.")
                print("[fox] No predictor files have been changed.")
                sys.exit(3)

            break

        rows = parse_table(table)

        print(f"[fox] Page {page}: {len(rows)} finals-player rows found")

        if rows:
            pages_with_data += 1
            all_rows.extend(rows)

        # Stop if we have moved beyond the real pages.
        # An empty table late in pagination is a useful signal.
        if page > 27 and not rows:
            break

        time.sleep(0.5)

    all_rows = dedupe_rows(all_rows)

    if not all_rows:
        print()
        print("[fox] No Finals player statistics were extracted.")
        print("[fox] No predictor files have been changed.")
        sys.exit(4)

    matched = 0

    for row in all_rows:
        full_name, method = match_current_player(
            row["fox_player_name"],
            row["team"],
            current_players,
        )

        row["matched_player"] = full_name
        row["match_method"] = method

        if full_name:
            matched += 1

    write_csv(all_rows)

    print()
    print("[fox] RESULTS")
    print(f"[fox] Pages containing finals players: {pages_with_data}")
    print(f"[fox] Finals players extracted: {len(all_rows)}")

    if current_players:
        print(f"[fox] Current-team players matched: {matched}")

    print()

    for team in sorted(FINALS_TEAMS):
        team_rows = [r for r in all_rows if r["team"] == team]

        print(f"[fox] {team}: {len(team_rows)} players")

        top_games = sorted(
            team_rows,
            key=lambda x: int(x["games"] or 0),
            reverse=True,
        )[:5]

        for player in top_games:
            print(
                "        "
                f"{player['fox_player_name']:<22} "
                f"G={player['games']:<3} "
                f"ST={player['starts']:<3} "
                f"MIN={player['minutes']:<5} "
                f"RM={player['run_metres']:<5} "
                f"TCK={player['tackles']:<5}"
            )

    print()
    print(f"[fox] Wrote {len(all_rows)} rows -> {OUTPUT_FILE}")
    print("[fox] Test complete.")
    print("[fox] Nothing has been connected to the prediction model.")


if __name__ == "__main__":
    main()
