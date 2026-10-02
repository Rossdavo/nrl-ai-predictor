#!/usr/bin/env python3
"""
collect_rlp_season_players.py
V5 content-driven Rugby League Project season population collector.

Changes from V4:
- Does NOT require /players/ hyperlinks.
- Captures ordinary HTML table rows and identifies player rows from their
  cell contents, especially the Team(s) column.
- One HTTP request only.
- No BeautifulSoup.
- No pandas.read_html().
- No player-by-player requests.
- Hard 45-second total runtime limit on Linux/GitHub Actions.

Experimental data infrastructure only.
Does not modify predict.py, run.yml, ratings, or live predictions.
"""

from __future__ import annotations

import re
import signal
import time
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import requests

SEASON = 2026
URL = f"https://www.rugbyleagueproject.org/seasons/nrl-{SEASON}/players.html"

OUTPUT = Path("rlp_season_players.csv")
FAILURES = Path("rlp_season_players_failures.csv")
SUMMARY = Path("rlp_season_players_summary.csv")
OPTIONAL_SEED = Path("nrl_roster_seed_2026_2027.csv")

CONNECT_TIMEOUT = 5
READ_TIMEOUT = 15
HARD_TOTAL_SECONDS = 45

TEAM_CODE_MAP = {
    "BRI": "Broncos",
    "CAN": "Raiders",
    "CBY": "Bulldogs",
    "CRO": "Sharks",
    "DOL": "Dolphins",
    "GLD": "Titans",
    "MAN": "Sea Eagles",
    "MEL": "Storm",
    "NEW": "Knights",
    "NQL": "Cowboys",
    "WAR": "Warriors",
    "PAR": "Eels",
    "PEN": "Panthers",
    "SOU": "Rabbitohs",
    "SGI": "Dragons",
    "SYD": "Roosters",
    "WST": "Wests Tigers",
}

EXPECTED_2026_TEAMS = set(TEAM_CODE_MAP.values())
TEAM_CODES = set(TEAM_CODE_MAP)

POSITION_MAP = {
    "FB": "Fullback",
    "W": "Wing",
    "C": "Centre",
    "FE": "Five-eighth",
    "HB": "Halfback",
    "FR": "Front row",
    "HK": "Hooker",
    "2R": "Second row",
    "L": "Lock",
    "B": "Bench",
}

TEAM_ALIASES = {
    "brisbane broncos": "Broncos",
    "broncos": "Broncos",
    "canberra raiders": "Raiders",
    "raiders": "Raiders",
    "canterbury bankstown bulldogs": "Bulldogs",
    "canterbury-bankstown bulldogs": "Bulldogs",
    "bulldogs": "Bulldogs",
    "cronulla sharks": "Sharks",
    "cronulla-sutherland sharks": "Sharks",
    "sharks": "Sharks",
    "dolphins": "Dolphins",
    "the dolphins": "Dolphins",
    "gold coast titans": "Titans",
    "titans": "Titans",
    "manly sea eagles": "Sea Eagles",
    "manly warringah sea eagles": "Sea Eagles",
    "sea eagles": "Sea Eagles",
    "melbourne storm": "Storm",
    "storm": "Storm",
    "newcastle knights": "Knights",
    "knights": "Knights",
    "north queensland cowboys": "Cowboys",
    "cowboys": "Cowboys",
    "new zealand warriors": "Warriors",
    "nz warriors": "Warriors",
    "warriors": "Warriors",
    "parramatta eels": "Eels",
    "eels": "Eels",
    "penrith panthers": "Panthers",
    "panthers": "Panthers",
    "south sydney rabbitohs": "Rabbitohs",
    "rabbitohs": "Rabbitohs",
    "st george illawarra dragons": "Dragons",
    "dragons": "Dragons",
    "sydney roosters": "Roosters",
    "roosters": "Roosters",
    "wests tigers": "Wests Tigers",
    "tigers": "Wests Tigers",
    "perth bears": "Perth Bears",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def clean(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def ascii_text(value) -> str:
    text = unicodedata.normalize("NFKD", clean(value))
    return "".join(c for c in text if not unicodedata.combining(c))


def norm_key(value) -> str:
    text = ascii_text(value).lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' -]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def player_key(name: str) -> str:
    return norm_key(name)


def player_name(raw: str) -> str:
    raw = clean(raw)
    if "," in raw:
        surname, given = [clean(x) for x in raw.split(",", 1)]
        raw = f"{given} {surname}"
    return raw.title()


def to_int(value):
    value = clean(value)
    if value in {"", "-", "–", "—"}:
        return pd.NA
    try:
        return int(value.replace(",", ""))
    except ValueError:
        return pd.NA


def parse_team_cell(raw: str):
    result = []

    for token in [clean(x) for x in clean(raw).split(",") if clean(x)]:
        match = re.fullmatch(r"([A-Za-z]{2,4})\s*-\s*(\d+)", token)
        if not match:
            result.append(("", None, token))
            continue

        code = match.group(1).upper()
        apps = int(match.group(2))
        result.append((TEAM_CODE_MAP.get(code, ""), apps, token))

    return result


def looks_like_team_cell(raw: str) -> bool:
    tokens = [clean(x) for x in clean(raw).split(",") if clean(x)]
    if not tokens:
        return False

    for token in tokens:
        match = re.fullmatch(r"([A-Za-z]{2,4})\s*-\s*(\d+)", token)
        if not match:
            return False
        if match.group(1).upper() not in TEAM_CODES:
            return False

    return True


def primary_position(raw: str) -> str:
    best_position = ""
    best_count = -1

    for token in [clean(x) for x in clean(raw).split(",") if clean(x)]:
        match = re.fullmatch(r"([A-Za-z0-9/]+)\s*-\s*(\d+)", token)
        if not match:
            continue

        code = match.group(1).upper()
        count = int(match.group(2))

        if count > best_count:
            best_position = POSITION_MAP.get(code, code)
            best_count = count

    return best_position


class CollectorTimeout(RuntimeError):
    pass


def alarm_handler(signum, frame):
    raise CollectorTimeout(
        f"collector exceeded {HARD_TOTAL_SECONDS}s hard runtime limit"
    )


class TableRowParser(HTMLParser):
    """
    Minimal streaming table parser.

    Stores text from all <td> cells in every <tr>.
    No dependency on player hyperlinks or CSS classes.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_tr = False
        self.in_td = False
        self.cells = []
        self.text_parts = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()

        if tag == "tr":
            self.in_tr = True
            self.in_td = False
            self.cells = []
            self.text_parts = []

        elif tag == "td" and self.in_tr:
            self.in_td = True
            self.text_parts = []

    def handle_data(self, data):
        if self.in_td:
            self.text_parts.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()

        if tag == "td" and self.in_td:
            self.cells.append(clean(" ".join(self.text_parts)))
            self.text_parts = []
            self.in_td = False

        elif tag == "tr" and self.in_tr:
            if self.cells:
                self.rows.append(self.cells[:])

            self.in_tr = False
            self.in_td = False
            self.cells = []
            self.text_parts = []


def fetch_html() -> str:
    log(f"Fetching ONE page: {URL}")

    response = requests.get(
        URL,
        headers={
            "User-Agent": "Mozilla/5.0 NRL-AI-Predictor research collector",
            "Accept": "text/html,application/xhtml+xml",
            "Connection": "close",
        },
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    response.raise_for_status()

    text = response.text

    log(
        f"Fetch complete: HTTP {response.status_code}, "
        f"{len(text):,} characters"
    )

    if len(text) < 20000:
        raise RuntimeError(
            f"RLP response unexpectedly small: {len(text):,} characters"
        )

    return text


def parse_rows(html_text: str):
    log("Parsing all HTML table rows with built-in HTMLParser")

    parser = TableRowParser()
    parser.feed(html_text)
    parser.close()

    log(f"Raw table rows captured: {len(parser.rows)}")

    candidate_rows = []

    for cells in parser.rows:
        # Live RLP player rows have 18 cells.
        # We allow >=17 for resilience, but require a valid Team(s) cell
        # in index 2.
        if len(cells) < 17:
            continue

        if len(cells) <= 3:
            continue

        if not looks_like_team_cell(cells[2]):
            continue

        candidate_rows.append(cells)

    log(f"Content-matched player rows: {len(candidate_rows)}")

    if len(candidate_rows) < 300:
        preview = candidate_rows[:3]
        raise RuntimeError(
            f"Only {len(candidate_rows)} content-matched player rows. "
            f"First matches: {preview}"
        )

    rows = []
    failures = []

    for cells in candidate_rows:
        raw_player = cells[0]
        raw_team = cells[2]
        raw_position = cells[3]
        name = player_name(raw_player)

        for team, team_apps, token in parse_team_cell(raw_team):
            if not team:
                failures.append(
                    {
                        "season": SEASON,
                        "player": name,
                        "raw_team": raw_team,
                        "reason": f"unmapped_team_token:{token}",
                    }
                )
                continue

            rows.append(
                {
                    "season": SEASON,
                    "team": team,
                    "player": name,
                    "player_key": player_key(name),
                    "primary_position": primary_position(raw_position),
                    "season_team_appearances": team_apps,
                    "season_starts": to_int(cells[4]),
                    "season_interchange": to_int(cells[5]),
                    "season_total_appearances": to_int(cells[6]),
                    "raw_rlp_team": raw_team,
                    "raw_rlp_position": raw_position,
                    "roster_status": "appeared_in_nrl_season",
                    "source": "rlp_season_players",
                    "data_status": "ok",
                }
            )

    output = pd.DataFrame(rows)

    failure_output = pd.DataFrame(
        failures,
        columns=["season", "player", "raw_team", "reason"],
    )

    if output.empty:
        raise RuntimeError("No valid player-team rows parsed")

    output = (
        output.drop_duplicates(
            ["season", "team", "player_key"],
            keep="last",
        )
        .sort_values(["team", "player_key"])
        .reset_index(drop=True)
    )

    log(
        f"Mapped {len(output)} player-team rows; "
        f"review rows: {len(failure_output)}"
    )

    return output, failure_output


def validate_2026(df: pd.DataFrame) -> None:
    current = df[df["season"] == SEASON]
    found = set(current["team"].dropna().astype(str))

    missing = sorted(EXPECTED_2026_TEAMS - found)
    unexpected = sorted(found - EXPECTED_2026_TEAMS)

    log(
        f"Validation: {current['player_key'].nunique()} unique 2026 players, "
        f"{len(found)} clubs"
    )

    if missing:
        raise RuntimeError(f"Missing expected 2026 clubs: {missing}")

    if unexpected:
        raise RuntimeError(f"Unexpected 2026 clubs: {unexpected}")


def append_future_seed(base: pd.DataFrame) -> pd.DataFrame:
    if not OPTIONAL_SEED.exists():
        log("No future roster seed found; skipping future population")
        return base

    log(f"Reading optional future seed: {OPTIONAL_SEED}")
    seed = pd.read_csv(OPTIONAL_SEED)

    if seed.empty or not {"team", "player"}.issubset(seed.columns):
        log("Future seed has no usable team/player rows")
        return base

    extras = []

    for _, row in seed.iterrows():
        name = clean(row.get("player", ""))
        raw_team = clean(row.get("team", ""))
        team = TEAM_ALIASES.get(norm_key(raw_team), raw_team)

        if not name or not team:
            continue

        season_value = pd.to_numeric(
            pd.Series([row.get("season", 2027)]),
            errors="coerce",
        ).iloc[0]

        season = int(season_value) if pd.notna(season_value) else 2027

        # The RLP collector owns the 2026 population.
        # The seed is for future/offseason players only.
        if season <= SEASON:
            continue

        extras.append(
            {
                "season": season,
                "team": team,
                "player": name,
                "player_key": player_key(name),
                "primary_position": clean(
                    row.get("primary_position", "")
                ),
                "season_team_appearances": pd.NA,
                "season_starts": pd.NA,
                "season_interchange": pd.NA,
                "season_total_appearances": pd.NA,
                "raw_rlp_team": "",
                "raw_rlp_position": "",
                "roster_status": clean(
                    row.get("roster_status", "")
                ) or "roster_seed",
                "source": "nrl_roster_seed_2026_2027",
                "data_status": "seed_only",
            }
        )

    if not extras:
        log("No future-player rows found in seed")
        return base

    log(f"Appending {len(extras)} future seed rows")

    return (
        pd.concat([base, pd.DataFrame(extras)], ignore_index=True)
        .drop_duplicates(
            ["season", "team", "player_key"],
            keep="first",
        )
        .sort_values(["season", "team", "player_key"])
        .reset_index(drop=True)
    )


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
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
    started = time.monotonic()

    use_alarm = hasattr(signal, "SIGALRM")
    previous_handler = None

    try:
        if use_alarm:
            previous_handler = signal.signal(
                signal.SIGALRM,
                alarm_handler,
            )
            signal.alarm(HARD_TOTAL_SECONDS)

        log("RLP SEASON POPULATION v5 CONTENT-MATCH PARSER")
        log("Hyperlink requirement: REMOVED")
        log("BeautifulSoup/read_html: REMOVED")
        log("Player-by-player requests: NO")

        html_text = fetch_html()
        players, failures = parse_rows(html_text)
        validate_2026(players)

        combined = append_future_seed(players)

        log("Writing CSV outputs")

        combined.to_csv(OUTPUT, index=False)
        failures.to_csv(FAILURES, index=False)
        build_summary(combined).to_csv(SUMMARY, index=False)

        current = combined[combined["season"] == SEASON]

        print("\n=== 2026 PLAYERS BY CLUB ===", flush=True)
        print(
            current.groupby("team")["player_key"]
            .nunique()
            .sort_index()
            .to_string(),
            flush=True,
        )

        found = set(current["team"].dropna().astype(str))
        missing = sorted(EXPECTED_2026_TEAMS - found)

        print(f"\n2026 clubs: {len(found)}", flush=True)
        print(
            "Missing 2026 clubs: "
            + ("NONE" if not missing else ", ".join(missing)),
            flush=True,
        )

        future = combined[combined["season"] > SEASON]

        if not future.empty:
            print("\n=== FUTURE / SEED PLAYERS ===", flush=True)
            print(
                future.groupby(["season", "team"])["player_key"]
                .nunique()
                .to_string(),
                flush=True,
            )

        log(f"Review/unmapped rows: {len(failures)}")
        log(f"Saved: {OUTPUT}")
        log(f"Saved: {FAILURES}")
        log(f"Saved: {SUMMARY}")
        log(
            f"COMPLETE in {time.monotonic() - started:.1f} seconds"
        )

        return 0

    except Exception as exc:
        log(
            f"FAILED after {time.monotonic() - started:.1f}s: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1

    finally:
        if use_alarm:
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(
                    signal.SIGALRM,
                    previous_handler,
                )


if __name__ == "__main__":
    raise SystemExit(main())
