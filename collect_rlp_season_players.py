#!/usr/bin/env python3

"""
collect_rlp_season_players.py

V6 - Rugby League Project 2026 season player population collector.

Important fix:
RLP uses optional HTML closing tags in its player table:

    <tr><td>PLAYER<td>AGE<td>TEAM...

rather than:

    <tr><td>PLAYER</td>...</tr>

So the parser must treat:
- a new <td> as the end of the previous cell
- a new <tr> as the end of the previous row

One HTTP request only.
No BeautifulSoup.
No pandas.read_html().
No player-by-player requests.
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

URL = (
    "https://www.rugbyleagueproject.org/"
    "seasons/nrl-2026/players.html"
)

OUTPUT = Path("rlp_season_players.csv")

FAILURES = Path(
    "rlp_season_players_failures.csv"
)

SUMMARY = Path(
    "rlp_season_players_summary.csv"
)

OPTIONAL_SEED = Path(
    "nrl_roster_seed_2026_2027.csv"
)

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


EXPECTED_2026_TEAMS = set(
    TEAM_CODE_MAP.values()
)

TEAM_CODES = set(
    TEAM_CODE_MAP
)


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


def log(message):

    print(
        f"[{time.strftime('%H:%M:%S')}] "
        f"{message}",
        flush=True,
    )


def clean(value):

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except (TypeError, ValueError):

        pass

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def ascii_text(value):

    text = unicodedata.normalize(
        "NFKD",
        clean(value),
    )

    return "".join(
        c
        for c in text
        if not unicodedata.combining(c)
    )


def norm_key(value):

    text = (
        ascii_text(value)
        .lower()
        .replace("’", "'")
    )

    text = re.sub(
        r"[^a-z0-9' -]",
        "",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def player_key(name):

    return norm_key(name)


def player_name(raw):

    raw = clean(raw)

    if "," in raw:

        surname, given = [
            clean(x)
            for x in raw.split(",", 1)
        ]

        raw = f"{given} {surname}"

    return raw.title()


def to_int(value):

    value = clean(value)

    if value in {
        "",
        "-",
        "–",
        "—",
    }:

        return pd.NA

    try:

        return int(
            value.replace(",", "")
        )

    except ValueError:

        return pd.NA


def parse_team_cell(raw):

    result = []

    tokens = [
        clean(x)
        for x in clean(raw).split(",")
        if clean(x)
    ]

    for token in tokens:

        match = re.fullmatch(
            r"([A-Za-z]{2,4})\s*-\s*(\d+)",
            token,
        )

        if not match:

            result.append(
                ("", None, token)
            )

            continue

        code = (
            match.group(1)
            .upper()
        )

        appearances = int(
            match.group(2)
        )

        result.append(
            (
                TEAM_CODE_MAP.get(
                    code,
                    "",
                ),
                appearances,
                token,
            )
        )

    return result


def looks_like_team_cell(raw):

    tokens = [
        clean(x)
        for x in clean(raw).split(",")
        if clean(x)
    ]

    if not tokens:

        return False

    for token in tokens:

        match = re.fullmatch(
            r"([A-Za-z]{2,4})\s*-\s*(\d+)",
            token,
        )

        if not match:

            return False

        code = (
            match.group(1)
            .upper()
        )

        if code not in TEAM_CODES:

            return False

    return True


def primary_position(raw):

    best_position = ""
    best_count = -1

    tokens = [
        clean(x)
        for x in clean(raw).split(",")
        if clean(x)
    ]

    for token in tokens:

        match = re.fullmatch(
            r"([A-Za-z0-9/]+)\s*-\s*(\d+)",
            token,
        )

        if not match:

            continue

        code = (
            match.group(1)
            .upper()
        )

        count = int(
            match.group(2)
        )

        if count > best_count:

            best_position = (
                POSITION_MAP.get(
                    code,
                    code,
                )
            )

            best_count = count

    return best_position


class CollectorTimeout(RuntimeError):

    pass


def alarm_handler(signum, frame):

    raise CollectorTimeout(
        "collector exceeded "
        f"{HARD_TOTAL_SECONDS}s "
        "hard runtime limit"
    )


class RLPOptionalTagParser(HTMLParser):

    """
    Parser designed for RLP's compact HTML.

    RLP omits </td> and </tr> tags.

    Therefore:

    new <td> = previous cell finished
    new <tr> = previous row finished
    """

    def __init__(self):

        super().__init__(
            convert_charrefs=True
        )

        self.in_row = False
        self.in_cell = False

        self.cells = []
        self.text_parts = []
        self.rows = []


    def finish_cell(self):

        if not self.in_cell:
            return

        text = clean(
            " ".join(
                self.text_parts
            )
        )

        self.cells.append(
            text
        )

        self.text_parts = []

        self.in_cell = False


    def finish_row(self):

        if not self.in_row:
            return

        self.finish_cell()

        if self.cells:

            self.rows.append(
                self.cells[:]
            )

        self.cells = []

        self.in_row = False


    def handle_starttag(
        self,
        tag,
        attrs,
    ):

        tag = tag.lower()

        if tag == "tr":

            self.finish_row()

            self.in_row = True

            self.cells = []

            self.text_parts = []

            self.in_cell = False


        elif (
            tag in {"td", "th"}
            and self.in_row
        ):

            self.finish_cell()

            self.in_cell = True

            self.text_parts = []


    def handle_data(
        self,
        data,
    ):

        if self.in_cell:

            self.text_parts.append(
                data
            )


    def handle_endtag(
        self,
        tag,
    ):

        tag = tag.lower()

        if tag in {
            "td",
            "th",
        }:

            self.finish_cell()


        elif tag == "tr":

            self.finish_row()


        elif tag in {
            "tbody",
            "thead",
            "table",
        }:

            self.finish_row()


    def close(self):

        super().close()

        self.finish_row()


def fetch_html():

    log(
        f"Fetching ONE page: {URL}"
    )

    response = requests.get(

        URL,

        headers={

            "User-Agent":
                "Mozilla/5.0 "
                "NRL-AI-Predictor "
                "research collector",

            "Accept":
                "text/html,"
                "application/xhtml+xml",

            "Connection":
                "close",
        },

        timeout=(
            CONNECT_TIMEOUT,
            READ_TIMEOUT,
        ),
    )

    response.raise_for_status()

    text = response.text

    log(
        "Fetch complete: "
        f"HTTP {response.status_code}, "
        f"{len(text):,} characters"
    )

    if len(text) < 20000:

        raise RuntimeError(
            "RLP response unexpectedly "
            f"small: {len(text):,} "
            "characters"
        )

    return text


def parse_rows(html_text):

    log(
        "Parsing RLP "
        "optional-closing-tag table"
    )

    parser = (
        RLPOptionalTagParser()
    )

    parser.feed(
        html_text
    )

    parser.close()

    log(
        "Raw table rows captured: "
        f"{len(parser.rows)}"
    )


    cell_counts = {}

    for row in parser.rows:

        count = len(row)

        cell_counts[count] = (
            cell_counts.get(
                count,
                0,
            )
            + 1
        )


    common_counts = sorted(

        cell_counts.items(),

        key=lambda x: (
            -x[1],
            x[0],
        ),

    )[:6]


    log(
        "Most common row cell counts: "
        f"{common_counts}"
    )


    candidate_rows = []


    for cells in parser.rows:

        if len(cells) < 18:

            continue

        if looks_like_team_cell(
            cells[2]
        ):

            candidate_rows.append(
                cells
            )


    log(
        "Content-matched player rows: "
        f"{len(candidate_rows)}"
    )


    if len(candidate_rows) < 300:

        preview = (
            parser.rows[:5]
        )

        raise RuntimeError(
            "Only "
            f"{len(candidate_rows)} "
            "content-matched player rows. "
            f"First raw rows: {preview}"
        )


    rows = []

    failures = []


    for cells in candidate_rows:

        raw_player = cells[0]

        raw_team = cells[2]

        raw_position = cells[3]

        name = player_name(
            raw_player
        )


        for (
            team,
            team_apps,
            token,
        ) in parse_team_cell(
            raw_team
        ):

            if not team:

                failures.append(
                    {
                        "season":
                            SEASON,

                        "player":
                            name,

                        "raw_team":
                            raw_team,

                        "reason":
                            "unmapped_team_token:"
                            f"{token}",
                    }
                )

                continue


            rows.append(
                {
                    "season":
                        SEASON,

                    "team":
                        team,

                    "player":
                        name,

                    "player_key":
                        player_key(name),

                    "primary_position":
                        primary_position(
                            raw_position
                        ),

                    "season_team_appearances":
                        team_apps,

                    "season_starts":
                        to_int(
                            cells[4]
                        ),

                    "season_interchange":
                        to_int(
                            cells[5]
                        ),

                    "season_total_appearances":
                        to_int(
                            cells[6]
                        ),

                    "raw_rlp_team":
                        raw_team,

                    "raw_rlp_position":
                        raw_position,

                    "roster_status":
                        "appeared_in_nrl_season",

                    "source":
                        "rlp_season_players",

                    "data_status":
                        "ok",
                }
            )


    output = pd.DataFrame(
        rows
    )


    failure_output = pd.DataFrame(

        failures,

        columns=[
            "season",
            "player",
            "raw_team",
            "reason",
        ],
    )


    if output.empty:

        raise RuntimeError(
            "No valid player-team "
            "rows parsed"
        )


    output = (

        output

        .drop_duplicates(

            [
                "season",
                "team",
                "player_key",
            ],

            keep="last",
        )

        .sort_values(

            [
                "team",
                "player_key",
            ]
        )

        .reset_index(
            drop=True
        )
    )


    log(
        f"Mapped {len(output)} "
        "player-team rows; "
        "review rows: "
        f"{len(failure_output)}"
    )


    return (
        output,
        failure_output,
    )


def validate_2026(df):

    current = df[
        df["season"] == SEASON
    ]

    found = set(
        current["team"]
        .dropna()
        .astype(str)
    )


    missing = sorted(
        EXPECTED_2026_TEAMS
        - found
    )


    unexpected = sorted(
        found
        - EXPECTED_2026_TEAMS
    )


    log(
        "Validation: "
        f"{current['player_key'].nunique()} "
        "unique 2026 players, "
        f"{len(found)} clubs"
    )


    if missing:

        raise RuntimeError(
            "Missing expected "
            f"2026 clubs: {missing}"
        )


    if unexpected:

        raise RuntimeError(
            "Unexpected 2026 clubs: "
            f"{unexpected}"
        )


def append_future_seed(base):

    if not OPTIONAL_SEED.exists():

        log(
            "No future roster seed "
            "found; skipping "
            "future population"
        )

        return base


    log(
        "Reading optional "
        f"future seed: {OPTIONAL_SEED}"
    )


    seed = pd.read_csv(
        OPTIONAL_SEED
    )


    if (
        seed.empty
        or not {
            "team",
            "player",
        }.issubset(
            seed.columns
        )
    ):

        log(
            "Future seed has no "
            "usable team/player rows"
        )

        return base


    extras = []


    for _, row in seed.iterrows():

        name = clean(
            row.get(
                "player",
                "",
            )
        )


        raw_team = clean(
            row.get(
                "team",
                "",
            )
        )


        team = TEAM_ALIASES.get(
            norm_key(raw_team),
            raw_team,
        )


        if not name or not team:

            continue


        season_value = pd.to_numeric(

            pd.Series(
                [
                    row.get(
                        "season",
                        2027,
                    )
                ]
            ),

            errors="coerce",

        ).iloc[0]


        if pd.notna(
            season_value
        ):

            season = int(
                season_value
            )

        else:

            season = 2027


        if season <= SEASON:

            continue


        extras.append(
            {
                "season":
                    season,

                "team":
                    team,

                "player":
                    name,

                "player_key":
                    player_key(name),

                "primary_position":
                    clean(
                        row.get(
                            "primary_position",
                            "",
                        )
                    ),

                "season_team_appearances":
                    pd.NA,

                "season_starts":
                    pd.NA,

                "season_interchange":
                    pd.NA,

                "season_total_appearances":
                    pd.NA,

                "raw_rlp_team":
                    "",

                "raw_rlp_position":
                    "",

                "roster_status":
                    clean(
                        row.get(
                            "roster_status",
                            "",
                        )
                    )
                    or "roster_seed",

                "source":
                    "nrl_roster_seed_2026_2027",

                "data_status":
                    "seed_only",
            }
        )


    if not extras:

        log(
            "No future-player rows "
            "found in seed"
        )

        return base


    log(
        f"Appending {len(extras)} "
        "future seed rows"
    )


    combined = pd.concat(

        [
            base,
            pd.DataFrame(extras),
        ],

        ignore_index=True,
    )


    combined = (

        combined

        .drop_duplicates(

            [
                "season",
                "team",
                "player_key",
            ],

            keep="first",
        )

        .sort_values(

            [
                "season",
                "team",
                "player_key",
            ]
        )

        .reset_index(
            drop=True
        )
    )


    return combined


def build_summary(df):

    return (

        df

        .groupby(
            [
                "season",
                "team",
            ],
            dropna=False,
        )

        .agg(
            players=(
                "player_key",
                "nunique",
            ),

            source_rows=(
                "player",
                "size",
            ),
        )

        .reset_index()

        .sort_values(
            [
                "season",
                "team",
            ]
        )
    )


def main():

    started = (
        time.monotonic()
    )


    use_alarm = hasattr(
        signal,
        "SIGALRM",
    )


    previous_handler = None


    try:

        if use_alarm:

            previous_handler = (
                signal.signal(
                    signal.SIGALRM,
                    alarm_handler,
                )
            )

            signal.alarm(
                HARD_TOTAL_SECONDS
            )


        log(
            "RLP SEASON POPULATION "
            "v6 OPTIONAL-TAG PARSER"
        )

        log(
            "Confirmed raw RLP HTML "
            "omits </td> and </tr> "
            "closing tags"
        )

        log(
            "BeautifulSoup/read_html: "
            "REMOVED"
        )

        log(
            "Player-by-player requests: "
            "NO"
        )


        html_text = fetch_html()


        players, failures = (
            parse_rows(
                html_text
            )
        )


        validate_2026(
            players
        )


        combined = (
            append_future_seed(
                players
            )
        )


        log(
            "Writing CSV outputs"
        )


        combined.to_csv(
            OUTPUT,
            index=False,
        )


        failures.to_csv(
            FAILURES,
            index=False,
        )


        build_summary(
            combined
        ).to_csv(
            SUMMARY,
            index=False,
        )


        current = combined[
            combined["season"]
            == SEASON
        ]


        print(
            "\n=== 2026 PLAYERS "
            "BY CLUB ===",
            flush=True,
        )


        print(

            current

            .groupby("team")[
                "player_key"
            ]

            .nunique()

            .sort_index()

            .to_string(),

            flush=True,
        )


        found = set(
            current["team"]
            .dropna()
            .astype(str)
        )


        missing = sorted(
            EXPECTED_2026_TEAMS
            - found
        )


        print(
            f"\n2026 clubs: "
            f"{len(found)}",
            flush=True,
        )


        print(
            "Missing 2026 clubs: "
            + (
                "NONE"
                if not missing
                else ", ".join(
                    missing
                )
            ),
            flush=True,
        )


        future = combined[
            combined["season"]
            > SEASON
        ]


        if not future.empty:

            print(
                "\n=== FUTURE / "
                "SEED PLAYERS ===",
                flush=True,
            )


            print(

                future

                .groupby(
                    [
                        "season",
                        "team",
                    ]
                )[
                    "player_key"
                ]

                .nunique()

                .to_string(),

                flush=True,
            )


        log(
            "Review/unmapped rows: "
            f"{len(failures)}"
        )


        log(
            f"Saved: {OUTPUT}"
        )


        log(
            f"Saved: {FAILURES}"
        )


        log(
            f"Saved: {SUMMARY}"
        )


        elapsed = (
            time.monotonic()
            - started
        )


        log(
            f"COMPLETE in "
            f"{elapsed:.1f} seconds"
        )


        return 0


    except Exception as exc:

        elapsed = (
            time.monotonic()
            - started
        )


        log(
            "FAILED after "
            f"{elapsed:.1f}s: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )


        return 1


    finally:

        if use_alarm:

            signal.alarm(0)

            if (
                previous_handler
                is not None
            ):

                signal.signal(
                    signal.SIGALRM,
                    previous_handler,
                )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
