#!/usr/bin/env python3
"""
collect_rlp_experience.py

Experimental Rugby League Project (RLP) career-experience collector.

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
- primary positions (when exposed on the page)
- RLP player URL and collection status

Important safeguards
--------------------
- Uses a cache so successfully collected players are not repeatedly fetched.
- Throttles requests.
- Failure for one player does not stop the whole build.
- Ambiguous search matches are NOT guessed.
- Existing successful cache rows are preserved.
- No rating points are calculated here.
- No prediction changes are made here.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote_plus, urljoin, urlparse, unquote
import html
import re
import sys
import time
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup

MASTER = Path("player_master.csv")
OUTPUT = Path("rlp_player_experience.csv")
FAILURES = Path("rlp_player_experience_failures.csv")
CACHE = Path("rlp_player_experience_cache.csv")

BASE = "https://www.rugbyleagueproject.org"
SEARCH_ENGINE = "https://www.google.com/search?q="
SEASON_PLAYERS_URL = "https://www.rugbyleagueproject.org/seasons/nrl-2026/players.html"

REQUEST_DELAY_SECONDS = 1.25
TIMEOUT_SECONDS = 20
USER_AGENT = (
    "Mozilla/5.0 (compatible; NRL-AI-Predictor-Research/1.0; "
    "+https://github.com/rossdavo/nrl-ai-predictor)"
)

SUCCESS_STATUSES = {"ok", "cached_ok"}

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
    "new zealand warriors": "Warriors",
    "nz warriors": "Warriors",
    "warriors": "Warriors",
    "north queensland cowboys": "Cowboys",
    "cowboys": "Cowboys",
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
    "bears": "Perth Bears",
}


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def ascii_text(value) -> str:
    text = clean(value)
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def player_key(value) -> str:
    text = ascii_text(value).lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' -]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_team(value) -> str:
    raw = clean(value)
    return TEAM_ALIASES.get(player_key(raw), raw)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-AU,en;q=0.9",
    })
    return s


def slug_candidates(name: str) -> list[str]:
    """
    RLP commonly uses firstname-surname style slugs. Try deterministic URLs
    before falling back to search.
    """
    base = player_key(name)
    base = base.replace("'", "")
    slug = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    if not slug:
        return []
    return [
        f"{BASE}/players/{slug}/summary.html",
    ]


def is_rlp_player_page(text: str, expected_name: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    if "playing career" not in lower:
        return False

    expected_tokens = [
        t for t in player_key(expected_name).split()
        if len(t) >= 2
    ]
    page_key = player_key(BeautifulSoup(text, "html.parser").get_text(" ", strip=True))
    return all(token in page_key for token in expected_tokens)


def fetch(s: requests.Session, url: str) -> tuple[int, str]:
    try:
        r = s.get(url, timeout=TIMEOUT_SECONDS)
        return r.status_code, r.text if r.ok else ""
    except requests.RequestException:
        return 0, ""


def rlp_index_name(raw_name: str) -> str:
    """Convert RLP season-index names such as 'TAPINE, Joseph' to normal order."""
    raw_name = clean(html.unescape(raw_name))
    if "," not in raw_name:
        return raw_name
    surname, given = raw_name.split(",", 1)
    return clean(f"{given} {surname}")


def build_2026_player_index(s: requests.Session) -> dict[str, str]:
    """
    Build an exact player_key -> RLP player URL map from the 2026 season page.

    This is much safer than guessing slugs. RLP has historical URL quirks and
    typos (for example apostrophes/double hyphens and misspelled slugs), while
    the season page contains RLP's own player link for the actual 2026 player.
    """
    status, text = fetch(s, SEASON_PLAYERS_URL)
    if status != 200 or not text:
        return {}

    found: dict[str, str] = {}
    # The RLP season HTML uses player anchors reliably even though some table
    # closing tags are optional/malformed.
    pattern = re.compile(
        r'<a\s+href=["\'](?P<href>/players/[^"\']+)["\'][^>]*>'
        r'(?P<name>[^<]+)</a>',
        flags=re.I,
    )
    for match in pattern.finditer(text):
        name = rlp_index_name(match.group("name"))
        key = player_key(name)
        href = html.unescape(match.group("href"))
        if not key or not href:
            continue
        # Prefer the canonical player link supplied by RLP. requests follows
        # redirects if this is an ID-style URL.
        found.setdefault(key, urljoin(BASE, href))

    return found


def search_rlp_url(
    s: requests.Session,
    name: str,
    season: int,
    season_index: dict[str, str] | None = None,
) -> tuple[str, str]:
    # For 2026 players, use RLP's own season index first. This resolves odd
    # historical slugs and strongly reduces same-name false matches.
    if season == 2026 and season_index:
        indexed = season_index.get(player_key(name), "")
        if indexed:
            status, text = fetch(s, indexed)
            time.sleep(REQUEST_DELAY_SECONDS)
            if status == 200 and is_rlp_player_page(text, name):
                return indexed, "season_index_verified"

    # Then try predictable RLP slugs.
    for url in slug_candidates(name):
        status, text = fetch(s, url)
        time.sleep(REQUEST_DELAY_SECONDS)
        if status == 200 and is_rlp_player_page(text, name):
            return url, "direct_slug"

    # Search fallback. Google may block automation; failure is handled safely.
    query = f'site:rugbyleagueproject.org/players "{name}" "Playing Career"'
    url = SEARCH_ENGINE + quote_plus(query)
    status, text = fetch(s, url)
    time.sleep(REQUEST_DELAY_SECONDS)

    if status != 200 or not text:
        return "", "search_unavailable"

    soup = BeautifulSoup(text, "html.parser")
    candidates = []

    for a in soup.find_all("a", href=True):
        href = html.unescape(a["href"])
        match = re.search(
            r"(https?://(?:www\.)?rugbyleagueproject\.org/players/"
            r"[^&?#\"']+(?:/summary\.html)?)",
            href,
            flags=re.I,
        )
        if match:
            candidate = match.group(1)
            if candidate not in candidates:
                candidates.append(candidate)

    # Never guess among many candidates. Verify the page content.
    verified = []
    for candidate in candidates[:5]:
        status, page = fetch(s, candidate)
        time.sleep(REQUEST_DELAY_SECONDS)
        if status == 200 and is_rlp_player_page(page, name):
            verified.append(candidate)

    if len(verified) == 1:
        return verified[0], "search_verified"
    if len(verified) > 1:
        return "", "ambiguous_search"

    return "", "not_found"


def extract_int_from_row_text(text: str) -> int | None:
    """
    RLP table rows are generally:
    competition | comp wins | starts | int | APP | ...
    We prefer parsed HTML table extraction elsewhere, but this is a fallback.
    """
    numbers = re.findall(r"(?<![\d.])\d{1,4}(?![\d.])", text.replace(",", ""))
    if not numbers:
        return None
    return int(numbers[-1])


def parse_html_tables(html_text: str) -> list[pd.DataFrame]:
    try:
        return pd.read_html(html_text)
    except Exception:
        return []


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        cols = []
        for col in out.columns:
            parts = [
                clean(x) for x in col
                if clean(x) and not clean(x).lower().startswith("unnamed")
            ]
            cols.append(" ".join(dict.fromkeys(parts)))
        out.columns = cols
    else:
        out.columns = [clean(c) for c in out.columns]
    return out


def find_app_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        ck = player_key(col)
        if ck == "app" or "total appearances" in ck:
            return col
    return None


def find_competition_column(df: pd.DataFrame) -> str | None:
    # Usually first column.
    for col in df.columns:
        values = " ".join(df[col].astype(str).head(8).tolist()).lower()
        if any(term in values for term in [
            "nrl premiership", "nrl finals", "state of origin",
            "super league", "international"
        ]):
            return col
    return df.columns[0] if len(df.columns) else None


def competition_appearances(
    tables: list[pd.DataFrame],
) -> dict[str, int | None]:
    result = {
        "nrl_games": None,
        "nrl_finals_games": None,
        "state_of_origin_games": None,
        "test_international_games": None,
        "super_league_games": None,
    }

    for raw in tables:
        df = flatten_columns(raw)
        if df.empty:
            continue

        app_col = find_app_column(df)
        comp_col = find_competition_column(df)
        if not app_col or not comp_col:
            continue

        for _, row in df.iterrows():
            comp = clean(row.get(comp_col, ""))
            comp_key = player_key(comp)
            app = pd.to_numeric(
                pd.Series([row.get(app_col)]), errors="coerce"
            ).iloc[0]
            if pd.isna(app):
                continue
            app = int(app)

            if (
                "nrl premiership" in comp_key
                or "arl nrl premiership" in comp_key
                or "nswrl nrl premiership" in comp_key
                or "nswrfl nswrl nrl premiership" in comp_key
            ):
                result["nrl_games"] = max(result["nrl_games"] or 0, app)

            elif "nrl finals" in comp_key or "arl nrl finals" in comp_key:
                result["nrl_finals_games"] = max(
                    result["nrl_finals_games"] or 0, app
                )

            elif "state of origin" in comp_key:
                result["state_of_origin_games"] = max(
                    result["state_of_origin_games"] or 0, app
                )

            elif (
                "tests senior international matches" in comp_key
                or comp_key == "tests"
                or "senior international matches" in comp_key
            ):
                result["test_international_games"] = max(
                    result["test_international_games"] or 0, app
                )

            elif comp_key == "super league" or "super league" in comp_key:
                result["super_league_games"] = max(
                    result["super_league_games"] or 0, app
                )

    return result


def parse_name_and_positions(html_text: str) -> tuple[str, str]:
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text("\n", strip=True)

    h1 = soup.find("h1")
    name = clean(h1.get_text(" ", strip=True)) if h1 else ""

    positions = ""
    match = re.search(
        r"Position\(s\)\s*\n?\s*([^\n]+)",
        text,
        flags=re.I,
    )
    if match:
        positions = clean(match.group(1))

    return name, positions


def collect_one(
    s: requests.Session,
    season: int,
    team: str,
    name: str,
    pkey: str,
    season_index: dict[str, str] | None = None,
) -> dict:
    url, discovery = search_rlp_url(s, name, season, season_index)
    now = pd.Timestamp.now(tz="UTC").isoformat()

    base = {
        "season": season,
        "team": team,
        "player": name,
        "player_key": pkey,
        "rlp_url": url,
        "rlp_name": "",
        "rlp_positions": "",
        "nrl_games": pd.NA,
        "nrl_finals_games": pd.NA,
        "state_of_origin_games": pd.NA,
        "test_international_games": pd.NA,
        "super_league_games": pd.NA,
        "collection_status": "",
        "verification_note": discovery,
        "collected_at_utc": now,
    }

    if not url:
        base["collection_status"] = "not_found"
        return base

    status, text = fetch(s, url)
    time.sleep(REQUEST_DELAY_SECONDS)

    if status != 200 or not text:
        base["collection_status"] = "fetch_failed"
        base["verification_note"] = f"{discovery}; http_status={status}"
        return base

    if not is_rlp_player_page(text, name):
        base["collection_status"] = "name_mismatch"
        base["verification_note"] = f"{discovery}; page failed name verification"
        return base

    rlp_name, positions = parse_name_and_positions(text)
    stats = competition_appearances(parse_html_tables(text))

    base["rlp_name"] = rlp_name
    base["rlp_positions"] = positions
    base.update(stats)

    if stats["nrl_games"] is None:
        base["collection_status"] = "partial"
        base["verification_note"] = (
            f"{discovery}; verified page but NRL Premiership APP not parsed"
        )
    else:
        base["collection_status"] = "ok"
        base["verification_note"] = discovery

    return base


def load_master() -> pd.DataFrame:
    if not MASTER.exists():
        raise FileNotFoundError(
            f"Missing {MASTER}. Run build_player_master.py first."
        )

    df = pd.read_csv(MASTER)
    required = {"season", "team", "player"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{MASTER} missing: {', '.join(sorted(missing))}"
        )

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df[df["season"].notna()].copy()
    df["season"] = df["season"].astype(int)
    df["team"] = df["team"].map(norm_team)
    df["player"] = df["player"].map(clean)

    if "player_key" not in df.columns:
        df["player_key"] = df["player"].map(player_key)
    else:
        df["player_key"] = df["player_key"].map(clean)
        blank = df["player_key"] == ""
        df.loc[blank, "player_key"] = df.loc[blank, "player"].map(player_key)

    return df[
        (df["team"] != "") & (df["player_key"] != "")
    ].drop_duplicates(
        ["season", "team", "player_key"], keep="last"
    )


def cache_lookup(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    lookup = {}
    for _, row in df.iterrows():
        pkey = clean(row.get("player_key", ""))
        status = clean(row.get("collection_status", ""))
        if pkey and status in {"ok", "cached_ok"}:
            lookup[pkey] = row.to_dict()
    return lookup


def normalise_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[OUTPUT_COLUMNS]


def main() -> int:
    try:
        master = load_master()
        old_cache = normalise_output(read_csv(CACHE))
        cached = cache_lookup(old_cache)

        rows = []
        s = session()

        print("Building exact 2026 RLP player-link index...")
        season_index = build_2026_player_index(s)
        print(f"2026 RLP player links indexed: {len(season_index)}")

        unique_players = master.sort_values(
            ["season", "team", "player_key"]
        ).drop_duplicates("player_key", keep="last")

        total = len(unique_players)
        print("\nRLP EXPERIENCE COLLECTOR")
        print("=" * 64)
        print(f"Unique players to resolve: {total}")
        print("Fox Sports stats used: NO")
        print("Prediction changes: NO\n")

        for i, p in enumerate(unique_players.itertuples(index=False), start=1):
            pkey = clean(p.player_key)

            if pkey in cached:
                rec = dict(cached[pkey])
                rec["season"] = int(p.season)
                rec["team"] = p.team
                rec["player"] = p.player
                rec["collection_status"] = "cached_ok"
                rows.append(rec)
                print(f"[{i}/{total}] CACHE {p.player}")
                continue

            print(f"[{i}/{total}] FETCH {p.player}")
            rec = collect_one(
                s,
                int(p.season),
                p.team,
                p.player,
                pkey,
                season_index,
            )
            rows.append(rec)

        result = normalise_output(pd.DataFrame(rows))
        result = result.sort_values(
            ["season", "team", "player_key"]
        ).reset_index(drop=True)

        result.to_csv(OUTPUT, index=False)

        failures = result[
            ~result["collection_status"].isin(
                ["ok", "cached_ok"]
            )
        ].copy()
        failures.to_csv(FAILURES, index=False)

        # Cache latest successful/partial record per player.
        cache_rows = pd.concat(
            [
                old_cache,
                result[result["collection_status"].isin(
                    ["ok", "cached_ok", "partial"]
                )]
            ],
            ignore_index=True,
        )
        if not cache_rows.empty:
            cache_rows = cache_rows.drop_duplicates(
                "player_key", keep="last"
            )
        normalise_output(cache_rows).to_csv(CACHE, index=False)

        good = int(result["collection_status"].isin(
            ["ok", "cached_ok"]
        ).sum())
        partial = int((result["collection_status"] == "partial").sum())

        print("\n" + "=" * 64)
        print(f"Resolved cleanly: {good}/{len(result)}")
        print(f"Partial: {partial}")
        print(f"Failures/review: {len(failures)}")
        print(f"Saved: {OUTPUT}")
        print(f"Saved: {FAILURES}")
        print(f"Saved: {CACHE}")
        print("\nNo player ratings or predictions were changed.")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
