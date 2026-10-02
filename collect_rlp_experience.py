#!/usr/bin/env python3

"""
collect_rlp_experience.py

Rugby League Project career-experience collector.
Parser revision: v3 league-wide player-master collector.

Purpose
-------
Enrich player_master.csv with career experience without using the separate
Fox Sports statistics experiment.

This collector is deliberately isolated from predict.py.

Input
-----
player_master.csv

Outputs
-------
rlp_player_experience.csv
rlp_player_experience_failures.csv
rlp_player_experience_cache.csv

What it attempts to collect
---------------------------
- NRL Premiership appearances
- NRL Finals appearances
- State of Origin appearances
- Senior international/Test appearances
- Super League appearances
- primary positions when exposed on the page
- RLP player URL and collection status

Important safeguards
--------------------
- One RLP lookup per unique player identity.
- Prefers the player's 2026 NRL record over a 2027 roster-seed record.
- Uses successful cached results instead of fetching them again.
- Throttles requests.
- Failure for one player does not stop the whole build.
- Ambiguous search matches are NOT guessed.
- Existing successful cache rows are preserved.
- No rating points are calculated here.
- No prediction changes are made here.
- Fox Sports data is NOT used.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote_plus
import html
import re
import sys
import time
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup


MASTER = Path("player_master.csv")

OUTPUT = Path(
    "rlp_player_experience.csv"
)

FAILURES = Path(
    "rlp_player_experience_failures.csv"
)

CACHE = Path(
    "rlp_player_experience_cache.csv"
)


BASE = (
    "https://www.rugbyleagueproject.org"
)

SEARCH_ENGINE = (
    "https://www.google.com/search?q="
)


REQUEST_DELAY_SECONDS = 1.25

TIMEOUT_SECONDS = 20


USER_AGENT = (
    "Mozilla/5.0 "
    "(compatible; "
    "NRL-AI-Predictor-Research/1.0; "
    "+https://github.com/"
    "rossdavo/nrl-ai-predictor)"
)


SUCCESS_STATUSES = {
    "ok",
    "cached_ok",
}


OUTPUT_COLUMNS = [

    "season",
    "team",
    "player",
    "player_key",

    "rlp_url",
    "rlp_name",
    "rlp_positions",

    "nrl_games",
    "nrl_finals_games",
    "state_of_origin_games",
    "test_international_games",
    "super_league_games",

    "collection_status",
    "verification_note",
    "collected_at_utc",
]


TEAM_ALIASES = {

    "brisbane broncos":
        "Broncos",

    "broncos":
        "Broncos",

    "canberra raiders":
        "Raiders",

    "raiders":
        "Raiders",

    "canterbury bankstown bulldogs":
        "Bulldogs",

    "canterbury-bankstown bulldogs":
        "Bulldogs",

    "bulldogs":
        "Bulldogs",

    "cronulla sharks":
        "Sharks",

    "cronulla-sutherland sharks":
        "Sharks",

    "sharks":
        "Sharks",

    "dolphins":
        "Dolphins",

    "the dolphins":
        "Dolphins",

    "gold coast titans":
        "Titans",

    "titans":
        "Titans",

    "manly sea eagles":
        "Sea Eagles",

    "manly warringah sea eagles":
        "Sea Eagles",

    "sea eagles":
        "Sea Eagles",

    "melbourne storm":
        "Storm",

    "storm":
        "Storm",

    "newcastle knights":
        "Knights",

    "knights":
        "Knights",

    "new zealand warriors":
        "Warriors",

    "nz warriors":
        "Warriors",

    "warriors":
        "Warriors",

    "north queensland cowboys":
        "Cowboys",

    "cowboys":
        "Cowboys",

    "parramatta eels":
        "Eels",

    "eels":
        "Eels",

    "penrith panthers":
        "Panthers",

    "panthers":
        "Panthers",

    "south sydney rabbitohs":
        "Rabbitohs",

    "rabbitohs":
        "Rabbitohs",

    "st george illawarra dragons":
        "Dragons",

    "dragons":
        "Dragons",

    "sydney roosters":
        "Roosters",

    "roosters":
        "Roosters",

    "wests tigers":
        "Wests Tigers",

    "tigers":
        "Wests Tigers",

    "perth bears":
        "Perth Bears",

    "bears":
        "Perth Bears",
}


def clean(value) -> str:

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except (
        TypeError,
        ValueError,
    ):

        pass

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def ascii_text(value) -> str:

    text = clean(
        value
    )

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    return "".join(
        ch
        for ch in text
        if not unicodedata.combining(ch)
    )


def player_key(value) -> str:

    text = (
        ascii_text(value)
        .lower()
        .replace(
            "’",
            "'",
        )
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


def norm_team(value) -> str:

    raw = clean(
        value
    )

    return TEAM_ALIASES.get(
        player_key(raw),
        raw,
    )


def read_csv(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():

        return pd.DataFrame()

    try:

        return pd.read_csv(
            path
        )

    except pd.errors.EmptyDataError:

        return pd.DataFrame()


def session() -> requests.Session:

    s = requests.Session()

    s.headers.update(
        {
            "User-Agent":
                USER_AGENT,

            "Accept-Language":
                "en-AU,en;q=0.9",
        }
    )

    return s


def slug_candidates(
    name: str,
) -> list[str]:

    """
    RLP commonly uses firstname-surname style slugs.

    Try deterministic URLs before falling back
    to search.
    """

    base = player_key(
        name
    )

    base = base.replace(
        "'",
        "",
    )

    slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        base,
    ).strip("-")


    if not slug:

        return []


    return [

        (
            f"{BASE}/players/"
            f"{slug}/summary.html"
        ),

        (
            "https://org."
            "rugbyleagueproject.com/"
            f"players/{slug}/summary.html"
        ),
    ]


def is_rlp_player_page(
    text: str,
    expected_name: str,
) -> bool:

    if not text:

        return False


    lower = text.lower()


    if (
        "playing career"
        not in lower
    ):

        return False


    expected_tokens = [

        token

        for token
        in player_key(
            expected_name
        ).split()

        if len(token) >= 2
    ]


    soup = BeautifulSoup(
        text,
        "html.parser",
    )


    page_key = player_key(

        soup.get_text(
            " ",
            strip=True,
        )
    )


    return all(

        token in page_key

        for token
        in expected_tokens
    )


def fetch(
    s: requests.Session,
    url: str,
) -> tuple[int, str]:

    try:

        response = s.get(
            url,
            timeout=TIMEOUT_SECONDS,
        )

        return (
            response.status_code,
            (
                response.text
                if response.ok
                else ""
            ),
        )

    except requests.RequestException:

        return (
            0,
            "",
        )


def search_rlp_url(
    s: requests.Session,
    name: str,
) -> tuple[str, str]:

    # First try predictable RLP slugs.

    for url in slug_candidates(
        name
    ):

        status, text = fetch(
            s,
            url,
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )


        if (
            status == 200
            and is_rlp_player_page(
                text,
                name,
            )
        ):

            return (
                url,
                "direct_slug",
            )


    # Search fallback.
    # Google may block automation.
    # Failure is handled safely.

    query = (
        'site:rugbyleagueproject.org/players '
        f'"{name}" '
        '"Playing Career"'
    )


    url = (
        SEARCH_ENGINE
        + quote_plus(query)
    )


    status, text = fetch(
        s,
        url,
    )


    time.sleep(
        REQUEST_DELAY_SECONDS
    )


    if (
        status != 200
        or not text
    ):

        return (
            "",
            "search_unavailable",
        )


    soup = BeautifulSoup(
        text,
        "html.parser",
    )


    candidates = []


    for anchor in soup.find_all(
        "a",
        href=True,
    ):

        href = html.unescape(
            anchor["href"]
        )


        match = re.search(

            (
                r"(https?://"
                r"(?:www\.|org\.)?"
                r"rugbyleagueproject\.com/"
                r"players/"
                r"[^&?#]+/"
                r"summary\.html)"
            ),

            href,

            flags=re.I,
        )


        if match:

            candidate = (
                match.group(1)
            )


            if (
                candidate
                not in candidates
            ):

                candidates.append(
                    candidate
                )


    # Never guess among multiple candidates.
    # Verify page content.

    verified = []


    for candidate in candidates[:5]:

        status, page = fetch(
            s,
            candidate,
        )


        time.sleep(
            REQUEST_DELAY_SECONDS
        )


        if (
            status == 200
            and is_rlp_player_page(
                page,
                name,
            )
        ):

            verified.append(
                candidate
            )


    if len(verified) == 1:

        return (
            verified[0],
            "search_verified",
        )


    if len(verified) > 1:

        return (
            "",
            "ambiguous_search",
        )


    return (
        "",
        "not_found",
    )


def _cell_text(
    cell,
) -> str:

    return clean(

        cell.get_text(
            " ",
            strip=True,
        )
    )


def _to_int(
    value: str,
) -> int | None:

    value = (
        clean(value)
        .replace(
            ",",
            "",
        )
    )


    if value in {
        "",
        "-",
        "–",
        "—",
    }:

        return None


    match = re.search(
        r"\d+",
        value,
    )


    if not match:

        return None


    return int(
        match.group(0)
    )


def competition_appearances_from_html(
    html_text: str,
) -> dict[str, int | None]:

    """
    Parse RLP Playing Career Statistics
    directly from HTML.

    RLP career-summary rows use:

    competition
    blank
    Comp Wins
    Starts
    Int
    APP
    ...

    APP is therefore cell index 5.

    We only accept recognised competition
    summary rows.
    """

    result = {

        "nrl_games":
            None,

        "nrl_finals_games":
            None,

        "state_of_origin_games":
            None,

        "test_international_games":
            None,

        "super_league_games":
            None,
    }


    soup = BeautifulSoup(
        html_text,
        "html.parser",
    )


    for tr in soup.find_all(
        "tr"
    ):

        cells = tr.find_all(
            [
                "th",
                "td",
            ]
        )


        if len(cells) < 6:

            continue


        values = [

            _cell_text(cell)

            for cell
            in cells
        ]


        first = player_key(
            values[0]
        )


        field = None


        if (
            first
            == "nrl premiership"
        ):

            field = (
                "nrl_games"
            )


        elif (
            first
            == "nrl finals"
        ):

            field = (
                "nrl_finals_games"
            )


        elif (
            first
            == "state of origin"
        ):

            field = (
                "state_of_origin_games"
            )


        elif first in {

            (
                "senior international "
                "matches tests"
            ),

            (
                "senior international "
                "matches"
            ),

        }:

            field = (
                "test_international_games"
            )


        elif (
            first
            == "super league"
        ):

            field = (
                "super_league_games"
            )


        if field is None:

            continue


        app = _to_int(
            values[5]
        )


        if app is None:

            continue


        # If duplicate summary rows exist,
        # retain the largest career total.

        result[field] = max(

            result[field] or 0,

            app,
        )


    return result


def parse_name_and_positions(
    html_text: str,
) -> tuple[str, str]:

    soup = BeautifulSoup(
        html_text,
        "html.parser",
    )


    text = soup.get_text(
        "\n",
        strip=True,
    )


    h1 = soup.find(
        "h1"
    )


    name = (

        clean(
            h1.get_text(
                " ",
                strip=True,
            )
        )

        if h1

        else ""
    )


    positions = ""


    match = re.search(

        (
            r"Position\(s\)"
            r"\s*\n?\s*"
            r"([^\n]+)"
        ),

        text,

        flags=re.I,
    )


    if match:

        positions = clean(
            match.group(1)
        )


    return (
        name,
        positions,
    )


def collect_one(
    s: requests.Session,
    season: int,
    team: str,
    name: str,
    pkey: str,
) -> dict:

    url, discovery = (
        search_rlp_url(
            s,
            name,
        )
    )


    now = (
        pd.Timestamp
        .now(
            tz="UTC"
        )
        .isoformat()
    )


    base = {

        "season":
            season,

        "team":
            team,

        "player":
            name,

        "player_key":
            pkey,

        "rlp_url":
            url,

        "rlp_name":
            "",

        "rlp_positions":
            "",

        "nrl_games":
            pd.NA,

        "nrl_finals_games":
            pd.NA,

        "state_of_origin_games":
            pd.NA,

        "test_international_games":
            pd.NA,

        "super_league_games":
            pd.NA,

        "collection_status":
            "",

        "verification_note":
            discovery,

        "collected_at_utc":
            now,
    }


    if not url:

        base[
            "collection_status"
        ] = "not_found"

        return base


    status, text = fetch(
        s,
        url,
    )


    time.sleep(
        REQUEST_DELAY_SECONDS
    )


    if (
        status != 200
        or not text
    ):

        base[
            "collection_status"
        ] = "fetch_failed"

        base[
            "verification_note"
        ] = (
            f"{discovery}; "
            f"http_status={status}"
        )

        return base


    if not is_rlp_player_page(
        text,
        name,
    ):

        base[
            "collection_status"
        ] = "name_mismatch"

        base[
            "verification_note"
        ] = (
            f"{discovery}; "
            "page failed "
            "name verification"
        )

        return base


    rlp_name, positions = (
        parse_name_and_positions(
            text
        )
    )


    stats = (
        competition_appearances_from_html(
            text
        )
    )


    base[
        "rlp_name"
    ] = rlp_name


    base[
        "rlp_positions"
    ] = positions


    base.update(
        stats
    )


    if (
        stats[
            "nrl_games"
        ]
        is None
    ):

        base[
            "collection_status"
        ] = "partial"


        base[
            "verification_note"
        ] = (
            f"{discovery}; "
            "verified page but "
            "NRL Premiership APP "
            "not parsed"
        )


    else:

        base[
            "collection_status"
        ] = "ok"


        base[
            "verification_note"
        ] = discovery


    return base


def load_master() -> pd.DataFrame:

    if not MASTER.exists():

        raise FileNotFoundError(

            f"Missing {MASTER}. "
            "Run "
            "build_player_master.py "
            "first."
        )


    df = pd.read_csv(
        MASTER
    )


    required = {
        "season",
        "team",
        "player",
    }


    missing = (
        required
        - set(df.columns)
    )


    if missing:

        raise ValueError(

            f"{MASTER} missing: "
            f"{', '.join(sorted(missing))}"
        )


    df[
        "season"
    ] = pd.to_numeric(

        df[
            "season"
        ],

        errors="coerce",
    )


    df = df[
        df[
            "season"
        ].notna()
    ].copy()


    df[
        "season"
    ] = df[
        "season"
    ].astype(
        int
    )


    df[
        "team"
    ] = df[
        "team"
    ].map(
        norm_team
    )


    df[
        "player"
    ] = df[
        "player"
    ].map(
        clean
    )


    if (
        "player_key"
        not in df.columns
    ):

        df[
            "player_key"
        ] = df[
            "player"
        ].map(
            player_key
        )


    else:

        df[
            "player_key"
        ] = df[
            "player_key"
        ].map(
            clean
        )


        blank = (
            df[
                "player_key"
            ]
            == ""
        )


        df.loc[
            blank,
            "player_key",
        ] = (

            df.loc[
                blank,
                "player",
            ]

            .map(
                player_key
            )
        )


    df = df[
        (
            df[
                "team"
            ]
            != ""
        )
        &
        (
            df[
                "player_key"
            ]
            != ""
        )
    ].copy()


    return df.drop_duplicates(

        [
            "season",
            "team",
            "player_key",
        ],

        keep="last",
    )


def build_unique_player_list(
    master: pd.DataFrame,
) -> pd.DataFrame:

    """
    Produce exactly one collection row
    per unique player.

    Preference:
    1. 2026 NRL record
    2. later/future roster record

    This prevents a player who appears
    in both the 2026 competition and the
    2027 Perth seed from being labelled
    only as a Perth player in the career
    collection output.
    """

    work = master.copy()


    work[
        "_season_priority"
    ] = work[
        "season"
    ].apply(

        lambda season:

            0
            if int(season) == 2026
            else 1
    )


    work = work.sort_values(

        [
            "player_key",
            "_season_priority",
            "season",
            "team",
        ],

        ascending=[
            True,
            True,
            True,
            True,
        ],
    )


    unique = (
        work

        .drop_duplicates(
            "player_key",
            keep="first",
        )

        .drop(
            columns=[
                "_season_priority"
            ]
        )

        .reset_index(
            drop=True
        )
    )


    return unique


def cache_lookup(
    df: pd.DataFrame,
) -> dict:

    if df.empty:

        return {}


    lookup = {}


    for _, row in df.iterrows():

        pkey = clean(
            row.get(
                "player_key",
                "",
            )
        )


        status = clean(
            row.get(
                "collection_status",
                "",
            )
        )


        if (
            pkey
            and status
            in SUCCESS_STATUSES
        ):

            lookup[
                pkey
            ] = row.to_dict()


    return lookup


def normalise_output(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()


    for col in OUTPUT_COLUMNS:

        if col not in out.columns:

            out[
                col
            ] = pd.NA


    return out[
        OUTPUT_COLUMNS
    ]


def main() -> int:

    try:

        master = (
            load_master()
        )


        unique_players = (
            build_unique_player_list(
                master
            )
        )


        old_cache = (
            normalise_output(
                read_csv(
                    CACHE
                )
            )
        )


        cached = (
            cache_lookup(
                old_cache
            )
        )


        rows = []


        s = session()


        total = len(
            unique_players
        )


        cache_hits = 0

        fetched = 0


        print(
            "\nRLP EXPERIENCE "
            "COLLECTOR v3"
        )


        print(
            "=" * 68
        )


        print(
            "Player-master rows: "
            f"{len(master)}"
        )


        print(
            "Unique players to resolve: "
            f"{total}"
        )


        print(
            "Successful players already "
            "available in cache: "
            f"{sum(1 for key in unique_players['player_key'] if key in cached)}"
        )


        print(
            "Fox Sports stats used: NO"
        )


        print(
            "Prediction changes: NO"
        )


        print(
            "Ratings calculated: NO\n"
        )


        for i, p in enumerate(

            unique_players.itertuples(
                index=False
            ),

            start=1,
        ):

            pkey = clean(
                p.player_key
            )


            if pkey in cached:

                rec = dict(
                    cached[
                        pkey
                    ]
                )


                # Career stats belong to the
                # player identity. Refresh the
                # season/team labels using our
                # preferred master record.

                rec[
                    "season"
                ] = int(
                    p.season
                )


                rec[
                    "team"
                ] = p.team


                rec[
                    "player"
                ] = p.player


                rec[
                    "player_key"
                ] = pkey


                rec[
                    "collection_status"
                ] = "cached_ok"


                rows.append(
                    rec
                )


                cache_hits += 1


                print(

                    f"[{i}/{total}] "
                    f"CACHE  "
                    f"{p.player} "
                    f"({p.team})"
                )


                continue


            print(

                f"[{i}/{total}] "
                f"FETCH  "
                f"{p.player} "
                f"({p.team})"
            )


            rec = collect_one(

                s,

                int(
                    p.season
                ),

                p.team,

                p.player,

                pkey,
            )


            rows.append(
                rec
            )


            fetched += 1


        result = normalise_output(

            pd.DataFrame(
                rows
            )
        )


        result = (

            result

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


        result.to_csv(
            OUTPUT,
            index=False,
        )


        failures = result[

            ~result[
                "collection_status"
            ].isin(
                SUCCESS_STATUSES
            )

        ].copy()


        failures.to_csv(
            FAILURES,
            index=False,
        )


        # Preserve old successful cache
        # records and add new successful
        # records.

        successful_new = result[

            result[
                "collection_status"
            ].isin(
                SUCCESS_STATUSES
            )

        ].copy()


        cache_rows = pd.concat(

            [
                old_cache,
                successful_new,
            ],

            ignore_index=True,
        )


        if not cache_rows.empty:

            cache_rows = (

                cache_rows

                .drop_duplicates(
                    "player_key",
                    keep="last",
                )

                .reset_index(
                    drop=True
                )
            )


        normalise_output(
            cache_rows
        ).to_csv(
            CACHE,
            index=False,
        )


        good = int(

            result[
                "collection_status"
            ]

            .isin(
                SUCCESS_STATUSES
            )

            .sum()
        )


        partial = int(

            (
                result[
                    "collection_status"
                ]
                == "partial"
            ).sum()
        )


        not_found = int(

            (
                result[
                    "collection_status"
                ]
                == "not_found"
            ).sum()
        )


        search_unavailable = int(

            (
                result[
                    "verification_note"
                ]
                == "search_unavailable"
            ).sum()
        )


        fetch_failed = int(

            (
                result[
                    "collection_status"
                ]
                == "fetch_failed"
            ).sum()
        )


        name_mismatch = int(

            (
                result[
                    "collection_status"
                ]
                == "name_mismatch"
            ).sum()
        )


        ambiguous = int(

            (
                result[
                    "verification_note"
                ]
                == "ambiguous_search"
            ).sum()
        )


        coverage = (

            (
                good
                / len(result)
            )
            * 100

            if len(result)

            else 0.0
        )


        print(
            "\n"
            + "=" * 68
        )


        print(
            "COLLECTION SUMMARY"
        )


        print(
            "=" * 68
        )


        print(
            f"Unique players: "
            f"{len(result)}"
        )


        print(
            f"Cache hits: "
            f"{cache_hits}"
        )


        print(
            f"Fresh fetches attempted: "
            f"{fetched}"
        )


        print(
            f"Resolved cleanly: "
            f"{good}/{len(result)} "
            f"({coverage:.1f}%)"
        )


        print(
            f"Partial: "
            f"{partial}"
        )


        print(
            f"Not found: "
            f"{not_found}"
        )


        print(
            "Search unavailable: "
            f"{search_unavailable}"
        )


        print(
            f"Fetch failed: "
            f"{fetch_failed}"
        )


        print(
            f"Name mismatch: "
            f"{name_mismatch}"
        )


        print(
            f"Ambiguous search: "
            f"{ambiguous}"
        )


        print(
            "Failures/review: "
            f"{len(failures)}"
        )


        print(
            "\nSaved: "
            f"{OUTPUT}"
        )


        print(
            "Saved: "
            f"{FAILURES}"
        )


        print(
            "Saved: "
            f"{CACHE}"
        )


        print(
            "\nNo player ratings or "
            "predictions were changed."
        )


        return 0


    except Exception as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
