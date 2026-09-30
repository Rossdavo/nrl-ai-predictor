#!/usr/bin/env python3
"""
collect_rlp_season_players.py

Experimental all-club player population collector for the NRL AI Predictor.\nParser revision: v2 direct HTML row parser.

Purpose
-------
Use Rugby League Project's season-wide player table to create an automatic
population of players who appeared in the NRL season.

This solves a different problem from collect_rlp_experience.py:
- this file discovers the league-wide player population
- collect_rlp_experience.py enriches those players with career experience

Inputs
------
No local input is required for the 2026 RLP collection.

Optional:
- nrl_roster_seed_2026_2027.csv
  Used only to append future/offseason roster players, especially Perth Bears.

Outputs
-------
rlp_season_players.csv
rlp_season_players_failures.csv
rlp_season_players_summary.csv

Safeguards
----------
- Does NOT alter predict.py.
- Does NOT alter team_lists_history.csv.
- Does NOT invent players or ratings.
- Does NOT treat RLP as the weekly selected-team authority.
- Keeps source team code and parsing status for audit.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup

SEASON = 2026
URL = f"https://www.rugbyleagueproject.org/seasons/nrl-{SEASON}/players.html"

OUTPUT = Path("rlp_season_players.csv")
FAILURES = Path("rlp_season_players_failures.csv")
SUMMARY = Path("rlp_season_players_summary.csv")
OPTIONAL_SEED = Path("nrl_roster_seed_2026_2027.csv")

TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (compatible; NRL-AI-Predictor-Research/1.0; "
    "+https://github.com/rossdavo/nrl-ai-predictor)"
)

TEAM_CODE_MAP = {
    "BRI": "Broncos",
    "CAN": "Bulldogs",
    "CBY": "Bulldogs",
    "CBR": "Raiders",
    "CRO": "Sharks",
    "DOL": "Dolphins",
    "GLD": "Titans",
    "GCT": "Titans",
    "MAN": "Sea Eagles",
    "MEL": "Storm",
    "NEW": "Knights",
    "NQL": "Cowboys",
    "NQ": "Cowboys",
    "NZ": "Warriors",
    "NZW": "Warriors",
    "WAR": "Warriors",
    "PAR": "Eels",
    "PEN": "Panthers",
    "SOU": "Rabbitohs",
    "STG": "Dragons",
    "SGI": "Dragons",
    "SYD": "Roosters",
    "WST": "Wests Tigers",
}

TEAM_NAME_ALIASES = {
    "brisbane": "Broncos",
    "brisbane broncos": "Broncos",
    "broncos": "Broncos",
    "canberra": "Raiders",
    "canberra raiders": "Raiders",
    "raiders": "Raiders",
    "canterbury": "Bulldogs",
    "canterbury bankstown bulldogs": "Bulldogs",
    "canterbury-bankstown bulldogs": "Bulldogs",
    "bulldogs": "Bulldogs",
    "cronulla": "Sharks",
    "cronulla sharks": "Sharks",
    "cronulla sutherland sharks": "Sharks",
    "sharks": "Sharks",
    "dolphins": "Dolphins",
    "the dolphins": "Dolphins",
    "gold coast": "Titans",
    "gold coast titans": "Titans",
    "titans": "Titans",
    "manly": "Sea Eagles",
    "manly sea eagles": "Sea Eagles",
    "manly warringah sea eagles": "Sea Eagles",
    "sea eagles": "Sea Eagles",
    "melbourne": "Storm",
    "melbourne storm": "Storm",
    "storm": "Storm",
    "newcastle": "Knights",
    "newcastle knights": "Knights",
    "knights": "Knights",
    "north qld": "Cowboys",
    "north queensland": "Cowboys",
    "north queensland cowboys": "Cowboys",
    "cowboys": "Cowboys",
    "warriors": "Warriors",
    "new zealand warriors": "Warriors",
    "nz warriors": "Warriors",
    "parramatta": "Eels",
    "parramatta eels": "Eels",
    "eels": "Eels",
    "penrith": "Panthers",
    "penrith panthers": "Panthers",
    "panthers": "Panthers",
    "souths": "Rabbitohs",
    "south sydney": "Rabbitohs",
    "south sydney rabbitohs": "Rabbitohs",
    "rabbitohs": "Rabbitohs",
    "st geo illa": "Dragons",
    "st george illawarra": "Dragons",
    "st george illawarra dragons": "Dragons",
    "dragons": "Dragons",
    "sydney": "Roosters",
    "sydney roosters": "Roosters",
    "roosters": "Roosters",
    "wests tigers": "Wests Tigers",
    "tigers": "Wests Tigers",
    "perth bears": "Perth Bears",
}

POSITION_MAP = {
    "FB": "Fullback",
    "W": "Wing",
    "C": "Centre",
    "FE": "Five-eighth",
    "5/8": "Five-eighth",
    "HB": "Halfback",
    "FR": "Front row",
    "PR": "Front row",
    "H": "Hooker",
    "HK": "Hooker",
    "SR": "Second row",
    "L": "Lock",
    "LK": "Lock",
    "B": "Bench",
}


def clean(v) -> str:
    if pd.isna(v):
        return ""
    return re.sub(r"\s+", " ", str(v)).strip()


def ascii_text(v) -> str:
    text = unicodedata.normalize("NFKD", clean(v))
    return "".join(c for c in text if not unicodedata.combining(c))


def key(v) -> str:
    text = ascii_text(v).lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' -]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def title_player(raw: str) -> str:
    """
    RLP season table commonly displays SURNAME, Firstname.
    Convert to Firstname Surname while preserving simple hyphen/apostrophe names.
    """
    raw = clean(raw)
    if "," in raw:
        surname, given = [clean(x) for x in raw.split(",", 1)]
        raw = f"{given} {surname}"
    return " ".join(
        part if any(ch.islower() for ch in part) else part.title()
        for part in raw.split()
    )


def player_key(name: str) -> str:
    return key(name)


def normalise_team_name(v) -> str:
    raw = clean(v)
    return TEAM_NAME_ALIASES.get(key(raw), raw)


def parse_team_cell(raw: str) -> list[tuple[str, int | None, str]]:
    """
    RLP team cells look like:
      PAR-21
      MEL-10, PAR-5

    Return (normalised team, appearances-for-team, raw token).
    """
    raw = clean(raw)
    out = []

    for token in [clean(x) for x in raw.split(",") if clean(x)]:
        m = re.match(r"^([A-Za-z]+)\s*-\s*(\d+)$", token)
        if not m:
            out.append(("", None, token))
            continue

        code = m.group(1).upper()
        apps = int(m.group(2))
        team = TEAM_CODE_MAP.get(code, "")
        out.append((team, apps, token))

    return out


def primary_position(raw: str) -> str:
    """
    Position cells may be W-21 or W-12, C-9 etc.
    Choose the position with the largest appearance count.
    """
    best = ("", -1)
    for token in [clean(x) for x in clean(raw).split(",") if clean(x)]:
        m = re.match(r"^([^-\d]+(?:/\d+)?)\s*-\s*(\d+)$", token)
        if not m:
            continue
        code = clean(m.group(1)).upper()
        count = int(m.group(2))
        if count > best[1]:
            best = (POSITION_MAP.get(code, code), count)
    return best[0]


def fetch_html() -> str:
    r = requests.get(
        URL,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-AU,en;q=0.9"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.text


def parse_rlp_player_rows(html_text: str) -> list[dict]:
    """
    Parse the RLP season player table directly from HTML.

    Live RLP season rows are ordered:
      Player | Age | Team(s) | Position(s) | APP | INT | TOT |
      W | L | D | W% | T | G | Perc | FG | 2FG | Pts | List

    We identify rows by:
    - a player link under /players/
    - at least 17 table cells
    - a Team(s) cell containing a team-code/appearance token such as PAR-19

    This avoids pandas.read_html() assumptions about RLP's markup.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    rows = []

    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 17:
            continue

        values = [clean(c.get_text(" ", strip=True)) for c in cells]
        first_link = cells[0].find("a", href=True)

        if not first_link:
            continue

        href = clean(first_link.get("href", ""))
        if "/players/" not in href:
            continue

        raw_player = values[0]
        raw_team = values[2]
        raw_pos = values[3]

        if not re.search(r"[A-Za-z]{2,4}\s*-\s*\d+", raw_team):
            continue

        rows.append({
            "player": raw_player,
            "age": values[1],
            "team": raw_team,
            "position": raw_pos,
            "app": values[4],
            "int": values[5],
            "tot": values[6],
            "wins": values[7],
            "losses": values[8],
            "draws": values[9],
            "win_pct": values[10],
            "tries": values[11],
            "goals": values[12],
            "goal_pct": values[13],
            "field_goals": values[14],
            "two_point_field_goals": values[15],
            "points": values[16],
            "player_href": href,
        })

    if not rows:
        raise RuntimeError(
            "Could not identify any RLP season player rows from HTML"
        )

    return rows

def collect_rlp() -> tuple[pd.DataFrame, pd.DataFrame]:
    html_text = fetch_html()
    source_rows = parse_rlp_player_rows(html_text)

    rows = []
    failures = []

    for r in source_rows:
        raw_player = clean(r.get("player", ""))
        raw_team = clean(r.get("team", ""))
        raw_pos = clean(r.get("position", ""))

        if not raw_player or raw_player.lower() == "player":
            continue

        player = title_player(raw_player)
        teams = parse_team_cell(raw_team)

        if not teams:
            failures.append({
                "season": SEASON,
                "player": player,
                "raw_team": raw_team,
                "reason": "no_team_tokens",
            })
            continue

        valid = False
        for team, team_apps, token in teams:
            if not team:
                failures.append({
                    "season": SEASON,
                    "player": player,
                    "raw_team": raw_team,
                    "reason": f"unmapped_team_token:{token}",
                })
                continue

            valid = True
            rows.append({
                "season": SEASON,
                "team": team,
                "player": player,
                "player_key": player_key(player),
                "primary_position": primary_position(raw_pos),
                "season_team_appearances": team_apps,
                "season_starts": pd.to_numeric(
                    pd.Series([r.get("app")]), errors="coerce"
                ).iloc[0],
                "season_interchange": pd.to_numeric(
                    pd.Series([r.get("int")]), errors="coerce"
                ).iloc[0],
                "season_total_appearances": pd.to_numeric(
                    pd.Series([r.get("tot")]), errors="coerce"
                ).iloc[0],
                "raw_rlp_team": raw_team,
                "raw_rlp_position": raw_pos,
                "roster_status": "appeared_in_nrl_season",
                "source": "rlp_season_players",
                "data_status": "ok",
            })

        if not valid and not any(
            f["player"] == player for f in failures
        ):
            failures.append({
                "season": SEASON,
                "player": player,
                "raw_team": raw_team,
                "reason": "no_valid_team",
            })

    out = pd.DataFrame(rows)
    fail = pd.DataFrame(failures)

    if not out.empty:
        out = out.drop_duplicates(
            ["season", "team", "player_key"], keep="last"
        ).sort_values(["team", "player_key"]).reset_index(drop=True)

    return out, fail


def append_future_seed(base: pd.DataFrame) -> pd.DataFrame:
    """
    Append future/offseason seed players without pretending they appeared in
    the 2026 NRL season. This is mainly for Perth Bears 2027.
    """
    if not OPTIONAL_SEED.exists():
        return base

    try:
        seed = pd.read_csv(OPTIONAL_SEED)
    except Exception:
        return base

    if seed.empty or "player" not in seed.columns or "team" not in seed.columns:
        return base

    extras = []
    for _, r in seed.iterrows():
        player = clean(r.get("player", ""))
        if not player:
            continue

        season = pd.to_numeric(
            pd.Series([r.get("season", 2027)]), errors="coerce"
        ).iloc[0]
        season = int(season) if pd.notna(season) else 2027
        team = normalise_team_name(r.get("team", ""))

        if not team:
            continue

        extras.append({
            "season": season,
            "team": team,
            "player": player,
            "player_key": player_key(player),
            "primary_position": clean(r.get("primary_position", "")),
            "season_team_appearances": pd.NA,
            "season_starts": pd.NA,
            "season_interchange": pd.NA,
            "season_total_appearances": pd.NA,
            "raw_rlp_team": "",
            "raw_rlp_position": "",
            "roster_status": clean(r.get("roster_status", "roster_seed")) or "roster_seed",
            "source": "nrl_roster_seed_2026_2027",
            "data_status": "seed_only",
        })

    if not extras:
        return base

    combined = pd.concat([base, pd.DataFrame(extras)], ignore_index=True)
    return combined.drop_duplicates(
        ["season", "team", "player_key"], keep="first"
    ).sort_values(["season", "team", "player_key"]).reset_index(drop=True)


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["season", "team", "players", "source_rows"]
        )

    return (
        df.groupby(["season", "team"], dropna=False)
        .agg(
            players=("player_key", "nunique"),
            source_rows=("player", "size"),
        )
        .reset_index()
        .sort_values(["season", "team"])
    )


def main() -> int:
    try:
        print("\nRLP SEASON PLAYER POPULATION")
        print("=" * 68)
        print(f"Season source: {SEASON}")
        print(f"URL: {URL}")
        print("Live predictor changed: NO\n")

        rlp, failures = collect_rlp()
        combined = append_future_seed(rlp)

        combined.to_csv(OUTPUT, index=False)
        failures.to_csv(FAILURES, index=False)

        summary = make_summary(combined)
        summary.to_csv(SUMMARY, index=False)

        rlp_2026 = combined[combined["season"] == SEASON]

        print(f"2026 player-team rows: {len(rlp_2026)}")
        print(f"2026 unique players: {rlp_2026['player_key'].nunique()}")
        print(f"2026 clubs found: {rlp_2026['team'].nunique()}")

        print("\n2026 players by club:")
        print(
            rlp_2026.groupby("team")["player_key"]
            .nunique()
            .sort_index()
            .to_string()
        )

        future = combined[combined["season"] > SEASON]
        if not future.empty:
            print("\nFuture/seed population:")
            print(
                future.groupby(["season", "team"])["player_key"]
                .nunique()
                .to_string()
            )

        print(f"\nUnmapped/review rows: {len(failures)}")
        if not failures.empty:
            print(failures.head(30).to_string(index=False))

        print(f"\nSaved: {OUTPUT}")
        print(f"Saved: {FAILURES}")
        print(f"Saved: {SUMMARY}")
        print("\nNo ratings or predictions were changed.")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
