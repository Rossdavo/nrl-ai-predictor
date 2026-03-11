print("[predict] predict.py loaded")
import gzip
import math
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from io import StringIO
import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo
import json
import os
import html as ihtml

BANKROLL = 200.0
UNIT_PCT = 0.05
UNIT_SIZE = round(BANKROLL * UNIT_PCT, 2)

MAX_ROUND_EXPOSURE_PCT = 0.40
MAX_ROUND_EXPOSURE = BANKROLL * MAX_ROUND_EXPOSURE_PCT
print(f"[predict] bankroll=${BANKROLL} | unit=${UNIT_SIZE}")


# ----------------------------
# RUN MODE
# "TRIALS" = use hardcoded fixtures (not used in this version)
# "AUTO"   = pull upcoming fixtures automatically
# ----------------------------
MODE = "AUTO"

# ----------------------------
# Team lists (optional; may fallback)
# ----------------------------
FORCE_TRY_FALLBACK = False  # set to True to force try-scorer fallback profiles
TEAMLISTS_CSV_PATH = "teamlists.csv"  # optional manual override file (date/team/num/name)

# ----------------------------
# Results source for ratings (Attack/Defence)
# ----------------------------
RESULTS_URL = "https://fixturedownload.com/results/nrl-2026"
RESULTS_CACHE_PATH = "results_cache.csv"

# ----------------------------
# AUTO FIXTURE PULL
# ----------------------------
FIXTURE_FEED_URL = "https://fixturedownload.com/feed/json/nrl-2026"
SYDNEY_TZ = ZoneInfo("Australia/Sydney")

TEAM_NAME_NORMALISE = {
    # long -> short
    "Canterbury Bulldogs": "Bulldogs",
    "St George Illawarra Dragons": "Dragons",
    "Newcastle Knights": "Knights",
    "North Queensland Cowboys": "Cowboys",
    "Melbourne Storm": "Storm",
    "Parramatta Eels": "Eels",
    "New Zealand Warriors": "Warriors",
    "Sydney Roosters": "Roosters",
    "Brisbane Broncos": "Broncos",
    "Penrith Panthers": "Panthers",
    "Cronulla Sutherland Sharks": "Sharks",
    "Gold Coast Titans": "Titans",
    "Manly Warringah Sea Eagles": "Sea Eagles",
    "Canberra Raiders": "Raiders",
    "South Sydney Rabbitohs": "Rabbitohs",
    "Dolphins": "Dolphins",
    "Wests Tigers": "Wests Tigers",

    # short -> short (safe)
    "Bulldogs": "Bulldogs",
    "Dragons": "Dragons",
    "Knights": "Knights",
    "Cowboys": "Cowboys",
    "Storm": "Storm",
    "Eels": "Eels",
    "Warriors": "Warriors",
    "Roosters": "Roosters",
    "Broncos": "Broncos",
    "Panthers": "Panthers",
    "Sharks": "Sharks",
    "Titans": "Titans",
    "Sea Eagles": "Sea Eagles",
    "Raiders": "Raiders",
    "Rabbitohs": "Rabbitohs",
    "Wests Tigers": "Wests Tigers",
}

def norm_team(name: str) -> str:
    name = str(name).strip()
    return TEAM_NAME_NORMALISE.get(name, name)

# ----------------------------
# Regions (must use SHORT names only)
# ----------------------------
TEAM_REGION = {
    "Broncos": "QLD",
    "Cowboys": "QLD",
    "Dolphins": "QLD",
    "Titans": "QLD",
    "Storm": "VIC",
    "Raiders": "ACT",
    "Warriors": "NZ",
    "Roosters": "NSW",
    "Rabbitohs": "NSW",
    "Sea Eagles": "NSW",
    "Sharks": "NSW",
    "Dragons": "NSW",
    "Wests Tigers": "NSW",
    "Bulldogs": "NSW",
    "Eels": "NSW",
    "Knights": "NSW",
    "Panthers": "NSW",
}

ALL_TEAMS = sorted(list(TEAM_REGION.keys()))

@dataclass
class Match:
    date: str  # YYYY-MM-DD
    kickoff_local: str  # HH:MM (Sydney/local)
    home: str
    away: str
    venue: str

# ----------------------------
# TRIAL FIXTURES (kept, but unused in AUTO)
# ----------------------------
FIXTURES: List[Match] = [
    Match("2026-02-12", "19:00", "Dolphins", "Titans", "Kayo Stadium"),
    Match("2026-02-13", "18:00", "Raiders", "Storm", "Seiffert Oval"),
    Match("2026-02-13", "20:00", "Cowboys", "Panthers", "Queensland Country Bank Stadium"),
    Match("2026-02-14", "15:00", "Warriors", "Sea Eagles", "Go Media Stadium"),
    Match("2026-02-14", "17:30", "Wests Tigers", "Roosters", "Leichhardt Oval"),
    Match("2026-02-14", "19:30", "Knights", "Bulldogs", "McDonald Jones Stadium"),
    Match("2026-02-14", "20:00", "Dragons", "Rabbitohs", "Netstrata Jubilee Stadium"),
    Match("2026-02-15", "16:00", "Sharks", "Eels", "PointsBet Stadium"),
]

def travel_points_adjustment(home: str, away: str, venue: str) -> Tuple[float, float]:
    """
    Returns (home_points_delta, away_points_delta).

    Conservative rule-set:
    - NZ travel is the biggest impact.
    - Cross-region Australia travel is small.
    - Same-region games: no adjustment.
    """
    h_reg = TEAM_REGION.get(home, "UNK")
    a_reg = TEAM_REGION.get(away, "UNK")

    home_delta = 0.0
    away_delta = 0.0

    # NZ travel
    if a_reg == "NZ" and h_reg != "NZ":
        away_delta -= 1.6
        home_delta += 0.2

    if h_reg == "NZ" and a_reg != "NZ":
        away_delta -= 1.2
        home_delta += 0.2

    # Cross-region within Australia (small)
    def norm(reg: str) -> str:
        return "NSW" if reg == "ACT" else reg

    h_norm = norm(h_reg)
    a_norm = norm(a_reg)

    if h_norm in {"NSW", "QLD", "VIC"} and a_norm in {"NSW", "QLD", "VIC"} and h_norm != a_norm:
        away_delta -= 0.6
        home_delta += 0.1

    return home_delta, away_delta

def fetch_upcoming_fixtures(days_ahead: int = 7) -> List[Match]:
    now = datetime.now(SYDNEY_TZ)
    end = now + pd.Timedelta(days=days_ahead)

    r = requests.get(FIXTURE_FEED_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    data = r.json()

    matches: List[Match] = []
    for item in data:
        dt_str = item.get("date") or item.get("Date") or item.get("startDate") or item.get("StartDate")
        if not dt_str:
            continue

        try:
            dt = pd.to_datetime(dt_str, utc=True).tz_convert(SYDNEY_TZ)
        except Exception:
            continue

        if dt.to_pydatetime() < now or dt.to_pydatetime() > end.to_pydatetime():
            continue

        home = item.get("home") or item.get("Home") or item.get("homeTeam") or item.get("HomeTeam")
        away = item.get("away") or item.get("Away") or item.get("awayTeam") or item.get("AwayTeam")
        venue = item.get("location") or item.get("Location") or item.get("venue") or item.get("Venue") or ""

        if not home or not away:
            continue

        home = norm_team(home)
        away = norm_team(away)

        matches.append(
            Match(
                date=dt.strftime("%Y-%m-%d"),
                kickoff_local=dt.strftime("%H:%M"),
                home=str(home).strip(),
                away=str(away).strip(),
                venue=str(venue).strip(),
            )
        )

    matches.sort(key=lambda m: (m.date, m.kickoff_local))
    return matches

# ----------------------------
# TEAM LIST SCRAPE (optional; may fallback)
# ----------------------------
TEAM_CANONICAL_PATTERNS = [
    ("Canterbury Bulldogs", "Bulldogs"),
    ("St George Illawarra Dragons", "Dragons"),
    ("Newcastle Knights", "Knights"),
    ("North Queensland Cowboys", "Cowboys"),
    ("Melbourne Storm", "Storm"),
    ("Parramatta Eels", "Eels"),
    ("New Zealand Warriors", "Warriors"),
    ("Sydney Roosters", "Roosters"),
    ("Brisbane Broncos", "Broncos"),
    ("Penrith Panthers", "Panthers"),
    ("Cronulla Sutherland Sharks", "Sharks"),
    ("Gold Coast Titans", "Titans"),
    ("Manly Warringah Sea Eagles", "Sea Eagles"),
    ("Canberra Raiders", "Raiders"),
    ("South Sydney Rabbitohs", "Rabbitohs"),
    ("Wests Tigers", "Wests Tigers"),
    ("Dolphins", "Dolphins"),
    # short forms too
    ("Bulldogs", "Bulldogs"),
    ("Dragons", "Dragons"),
    ("Knights", "Knights"),
    ("Cowboys", "Cowboys"),
    ("Storm", "Storm"),
    ("Eels", "Eels"),
    ("Warriors", "Warriors"),
    ("Roosters", "Roosters"),
    ("Broncos", "Broncos"),
    ("Panthers", "Panthers"),
    ("Sharks", "Sharks"),
    ("Titans", "Titans"),
    ("Sea Eagles", "Sea Eagles"),
    ("Raiders", "Raiders"),
    ("Rabbitohs", "Rabbitohs"),
]

def _clean_text(s: str) -> str:
    if s is None:
        return ""
    s = ihtml.unescape(str(s))
    s = s.replace("\xa0", " ")
    s = s.replace("&#160;", " ")
    s = s.replace("&nbsp;", " ")
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r"^[\s\u2022\*\-\u25cf\u25e6]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _strip_html_to_text(html: str) -> str:
    html = re.sub(r"<!--[\s\S]*?-->", " ", html)
    html = re.sub(r"<(script|style|noscript)[\s\S]*?</\1>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(
        r"</(p|div|section|article|header|footer|li|ul|ol|h1|h2|h3|h4|h5|h6|tr|td|th)>",
        "\n",
        html,
        flags=re.IGNORECASE,
    )
    html = re.sub(r"<[^>]+>", " ", html)
    text = ihtml.unescape(html)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def _html_to_lines(html: str) -> List[str]:
    text = _strip_html_to_text(html)
    lines = [_clean_text(x) for x in text.splitlines()]
    return [x for x in lines if x]

def _resolve_team_name(raw: str) -> str:
    raw_clean = _clean_text(raw)
    if not raw_clean:
        return ""

    direct = norm_team(raw_clean)
    if direct in ALL_TEAMS:
        return direct

    raw_low = raw_clean.lower()

    for alias, short in sorted(TEAM_CANONICAL_PATTERNS, key=lambda x: len(x[0]), reverse=True):
        if alias.lower() in raw_low:
            return short

    raw_clean2 = re.sub(r"[^A-Za-z \-']", " ", raw_clean)
    raw_clean2 = re.sub(r"\s+", " ", raw_clean2).strip()
    direct2 = norm_team(raw_clean2)
    if direct2 in ALL_TEAMS:
        return direct2

    return ""

def _clean_player_name(name_raw: str) -> str:
    s = _clean_text(name_raw)
    s = re.sub(r"\s*-\s*sponsored by.*$", "", s, flags=re.I)
    s = re.sub(r"\s+sponsored by.*$", "", s, flags=re.I)
    s = re.sub(r"\s*\|\s*player partner.*$", "", s, flags=re.I)
    s = re.sub(r"\s*player partner.*$", "", s, flags=re.I)
    s = re.sub(r"\s*\(c\)\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*\(vc\)\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*\[[^\]]+\]\s*$", "", s)
    s = re.sub(r"^[\-\|\:\. ]+", "", s)
    s = re.sub(r"[\-\|\:\. ]+$", "", s)
    s = re.sub(r"[^A-Za-z \-'.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if re.search(
        r"^(Interchange|Reserves|Bench|Extended Bench|Team List|Player Partner|Coach|Match Details|Venue|Kick[- ]?off|Broadcast|Tickets|Officials|Related)$",
        s,
        re.I,
    ):
        return ""

    return s

def _parse_player_line(line: str) -> Optional[Tuple[int, str]]:
    s = _clean_text(line)
    m = re.match(r"^(\d{1,2})[\.\)]?\s+(.+)$", s)
    if not m:
        return None

    try:
        num = int(m.group(1))
    except Exception:
        return None

    if not (1 <= num <= 13):
        return None

    name = _clean_player_name(m.group(2))
    if not name or len(name.split()) < 2:
        return None

    return num, name

def _score_team_candidate(players: Dict[int, str]) -> int:
    if not players:
        return 0

    score = 0
    score += sum(8 for n in range(1, 14) if n in players)
    score += sum(1 for n in players if 1 <= n <= 13)

    for name in players.values():
        if len(name.split()) >= 2:
            score += 2
        if re.search(r"(team list|match details|venue|tickets|broadcast|player partner|officials|related)", name, re.I):
            score -= 20
        if len(name) < 4:
            score -= 10

    return score

def _merge_best_team_map(base: Dict[str, Dict[int, str]], incoming: Dict[str, Dict[int, str]]) -> Dict[str, Dict[int, str]]:
    out = dict(base)
    for team, players in incoming.items():
        if team not in ALL_TEAMS:
            continue

        current = out.get(team, {})
        if _score_team_candidate(players) > _score_team_candidate(current):
            out[team] = players
        elif _score_team_candidate(players) == _score_team_candidate(current) and len(players) > len(current):
            out[team] = players
    return out

def _parse_match_centre_blocks(lines: List[str]) -> Dict[str, Dict[int, str]]:
    """
    Parses line-by-line match-centre rows like:
      Fullback for Broncos is number 1
      Reece Walsh

    and also split-name variants like:
      Fullback for Eels is number 1
      Isaiah
      Iongi
    """
    out: Dict[str, Dict[int, str]] = {}

    pat_full = re.compile(
        r"^(?:[\u2022\*\-]\s*)?"
        r"(Fullback|Wing(?:er)?|Winger|Centre|Five[- ]?Eighth|Halfback|Prop|Hooker|Second Row|2nd Row|Back Row|Lock|Interchange|Reserve|Replacement)"
        r"\s+for\s+(.+?)\s+is\s+number\s+(\d{1,2})\s+(.+)$",
        re.IGNORECASE,
    )

    pat_prefix = re.compile(
        r"^(?:[\u2022\*\-]\s*)?"
        r"(Fullback|Wing(?:er)?|Winger|Centre|Five[- ]?Eighth|Halfback|Prop|Hooker|Second Row|2nd Row|Back Row|Lock|Interchange|Reserve|Replacement)"
        r"\s+for\s+(.+?)\s+is\s+number\s+(\d{1,2})$",
        re.IGNORECASE,
    )

    def is_number_line(s: str) -> bool:
        return bool(re.fullmatch(r"\d{1,2}(?:\s+\d{1,2})*", s))

    def is_match_centre_prefix(s: str) -> bool:
        return bool(pat_prefix.match(s) or pat_full.match(s))

    def looks_like_name_piece(s: str) -> bool:
        s2 = _clean_player_name(s)
        if not s2:
            return False
        if is_number_line(s2):
            return False
        if is_match_centre_prefix(s2):
            return False
        if re.match(
            r"^(Interchange|Reserves|Bench|Extended Bench|Team List|Player Partner|Coach|Match Details|Venue|Kick[- ]?off|Broadcast|Tickets|Officials|Related)\b",
            s2,
            re.I,
        ):
            return False
        return True

    def consume_name(start_idx: int) -> Tuple[str, int]:
        """
        Join up to 3 following lines into a player name.
        Returns (name, next_index_after_name).
        """
        parts: List[str] = []
        j = start_idx

        while j < len(lines) and len(parts) < 3:
            piece = _clean_text(lines[j])

            if not looks_like_name_piece(piece):
                break

            parts.append(piece)

            candidate = _clean_player_name(" ".join(parts))
            if len(candidate.split()) >= 2:
                return candidate, j + 1

            j += 1

        candidate = _clean_player_name(" ".join(parts))
        return candidate, j

    i = 0
    n = len(lines)

    while i < n:
        s = _clean_text(lines[i])

        if is_number_line(s):
            i += 1
            continue

        # Case 1: everything on one line
        m = pat_full.match(s)
        if m:
            _pos, team_raw, num_s, name_raw = m.groups()
            team = _resolve_team_name(team_raw)
            if team:
                try:
                    num = int(num_s)
                except Exception:
                    num = 0

                if 1 <= num <= 13:
                    name = _clean_player_name(name_raw)
                    if len(name.split()) >= 2:
                        out.setdefault(team, {})
                        out[team][num] = name
            i += 1
            continue

        # Case 2: name is on following line(s)
        m = pat_prefix.match(s)
        if m:
            _pos, team_raw, num_s = m.groups()
            team = _resolve_team_name(team_raw)

            try:
                num = int(num_s)
            except Exception:
                num = 0

            if team and 1 <= num <= 13:
                name, next_i = consume_name(i + 1)
                if len(name.split()) >= 2:
                    out.setdefault(team, {})
                    out[team][num] = name
                    i = next_i
                    continue

        i += 1

    return out
def _parse_team_heading_blocks(lines: List[str]) -> Dict[str, Dict[int, str]]:
    """
    Backup parser for article-style blocks like:
      Broncos
      1 Reece Walsh
      2 Jesse Arthars
      ...
    """
    out: Dict[str, Dict[int, str]] = {}
    i = 0
    n = len(lines)

    while i < n:
        line = _clean_text(lines[i])
        team = _resolve_team_name(line)

        if not team or len(line.split()) > 6:
            i += 1
            continue

        players: Dict[int, str] = {}
        j = i + 1
        non_player_run = 0

        while j < n:
            nxt = _clean_text(lines[j])
            next_team = _resolve_team_name(nxt)

            if next_team and next_team != team and len(players) >= 5 and len(nxt.split()) <= 6:
                break

            parsed = _parse_player_line(nxt)
            if parsed:
                num, name = parsed
                players[num] = name
                non_player_run = 0
            else:
                if re.match(
                    r"^(Interchange:?|Reserves:?|Bench:?|Extended Bench:?|Team List:?|Player Partner:?|Coach:?|Match Details:?|Venue:?|Kick[- ]?off:?|Broadcast:?|Tickets:?|Officials:?|Related\b)",
                    nxt,
                    re.I,
                ):
                    pass
                else:
                    non_player_run += 1
                    if len(players) >= 7 and non_player_run >= 4:
                        break

            j += 1

        if len(players) >= 7:
            existing = out.get(team, {})
            if _score_team_candidate(players) > _score_team_candidate(existing):
                out[team] = players

        i += 1

    return out

def _parse_compact_team_runs(text: str) -> Dict[str, Dict[int, str]]:
    """
    Backup parser for flattened team blocks.
    """
    out: Dict[str, Dict[int, str]] = {}

    team_stop = (
        r"(?:Canterbury Bulldogs|St George Illawarra Dragons|Newcastle Knights|North Queensland Cowboys|"
        r"Melbourne Storm|Parramatta Eels|New Zealand Warriors|Sydney Roosters|Brisbane Broncos|"
        r"Penrith Panthers|Cronulla Sutherland Sharks|Gold Coast Titans|Manly Warringah Sea Eagles|"
        r"Canberra Raiders|South Sydney Rabbitohs|Wests Tigers|Dolphins|Bulldogs|Dragons|Knights|"
        r"Cowboys|Storm|Eels|Warriors|Roosters|Broncos|Panthers|Sharks|Titans|Sea Eagles|Raiders|Rabbitohs)"
    )

    for alias, short_team in sorted(TEAM_CANONICAL_PATTERNS, key=lambda x: len(x[0]), reverse=True):
        pat = re.compile(
            rf"{re.escape(alias)}(?P<body>.*?)(?={team_stop}\b|$)",
            re.IGNORECASE | re.DOTALL,
        )

        for m in pat.finditer(text):
            body = _clean_text(m.group("body"))
            if not body:
                continue

            players: Dict[int, str] = {}
            for pm in re.finditer(
                r"(\d{1,2})[\.\)]?\s+([A-Z][A-Za-z' .\-]+?)(?=\s+\d{1,2}[\.\)]?\s+|Interchange:?|Reserves:?|Bench:?|Extended Bench:?|$)",
                body,
            ):
                try:
                    num = int(pm.group(1))
                except Exception:
                    continue

                if not (1 <= num <= 13):
                    continue

                name = _clean_player_name(pm.group(2))
                if not name or len(name.split()) < 2:
                    continue

                players[num] = name

            if len(players) >= 7:
                existing = out.get(short_team, {})
                if _score_team_candidate(players) > _score_team_candidate(existing):
                    out[short_team] = players

    return out

def fetch_starters_by_team(url: str) -> Dict[str, Dict[int, str]]:
    """
    Attempts to scrape named starters from an NRL team list article.
    Returns: { "TeamShortName": {1:"Name", 2:"Name", ... 13:"Name"} }
    """
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()

        text = _strip_html_to_text(r.text)
        lines = _html_to_lines(r.text)

        print("[debug] relevant teamlist lines sample:")
        shown = 0
        for idx, line in enumerate(lines):
            if "for " in line and " is number " in line:
                print("   ", repr(line))
                if idx + 1 < len(lines):
                    print("      next:", repr(lines[idx + 1]))
                if idx + 2 < len(lines):
                    print("      next2:", repr(lines[idx + 2]))
                shown += 1
                if shown >= 6:
                    break

        parsed_mc = _parse_match_centre_blocks(lines)
        print(
            "[debug] match-centre parsed teams:",
            ", ".join(f"{t}:{len(p)}" for t, p in sorted(parsed_mc.items()))
            if parsed_mc else "(none)"
        )

        parsed_heading = _parse_team_heading_blocks(lines)
        parsed_compact = _parse_compact_team_runs(text)

        candidates: List[Dict[str, Dict[int, str]]] = [
            parsed_mc,
            parsed_heading,
            parsed_compact,
        ]

        starters: Dict[str, Dict[int, str]] = {}
        for cand in candidates:
            starters = _merge_best_team_map(starters, cand)

        starters = {
            team: {k: v for k, v in sorted(players.items()) if 1 <= k <= 13}
            for team, players in starters.items()
            if sum(1 for n in range(1, 14) if n in players) >= 7
        }

        if starters:
            sample = sorted(starters.items(), key=lambda x: (-len(x[1]), x[0]))[:16]
            print("[info] teamlist scrape sample:", ", ".join([f"{t}:{len(p)}" for t, p in sample]))

            missing = [t for t in ALL_TEAMS if len(starters.get(t, {})) < 7]
            if missing:
                print("[warn] teamlist teams with weak/missing scrape:", ", ".join(missing))

            for team in sorted(ALL_TEAMS):
                print(f"[debug] scrape team {team}: starters={len(starters.get(team, {}))}")
        else:
            print("[warn] teamlist scrape returned 0 teams")
            print("[debug] first 40 teamlist lines:")
            for line in lines[:40]:
                print("   ", line)

        return starters

    except Exception as e:
        print(f"[warn] teamlist scrape failed: {e}")
        return {}

# ----------------------------
# MANUAL TEAMLIST OVERRIDES
# ----------------------------
def load_manual_teamlists(path: str = TEAMLISTS_CSV_PATH) -> Dict[str, Dict[str, Dict[int, str]]]:
    """
    Loads manual team lists from CSV:
      date, team, num, name

    Returns:
      manual_by_date[date][team][num] = name
    """
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[warn] could not read {path}: {e}")
        return {}

    required = {"date", "team", "num", "name"}
    if not required.issubset(set(df.columns)):
        print(f"[warn] {path} missing columns. Need {sorted(required)}")
        return {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["team"] = df["team"].astype(str).apply(norm_team)
    df["num"] = pd.to_numeric(df["num"], errors="coerce")
    df["name"] = df["name"].astype(str).str.strip()

    df = df.dropna(subset=["date", "team", "num", "name"])
    df = df[(df["num"] >= 1) & (df["num"] <= 17)]

    manual_by_date: Dict[str, Dict[str, Dict[int, str]]] = {}
    for _, r in df.iterrows():
        d = str(r["date"])
        t = str(r["team"])
        n = int(r["num"])
        nm = str(r["name"])
        manual_by_date.setdefault(d, {}).setdefault(t, {})[n] = nm

    print(f"[info] Loaded manual teamlists: {path} (dates={len(manual_by_date)})")
    return manual_by_date

# ----------------------------
# RESULTS INGEST (for Attack/Defence fitting)
# ----------------------------
def fetch_completed_results() -> pd.DataFrame:
    """
    Returns dataframe with columns: date, home, away, home_pts, away_pts

    Fixed behaviour:
    - Load cache if present (even if valid)
    - ALWAYS attempt web fetch from RESULTS_URL
    - Merge + dedupe (keeps latest) and write back to RESULTS_CACHE_PATH
    - If web fails, return cache (or empty if no cache)
    """
    needed = {"date", "home", "away", "home_pts", "away_pts"}

    cache = pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])
    if os.path.exists(RESULTS_CACHE_PATH):
        try:
            cached = pd.read_csv(RESULTS_CACHE_PATH)
            if needed.issubset(set(cached.columns)) and len(cached) > 0:
                cache = cached.copy()
                cache["date"] = pd.to_datetime(cache["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                cache["home"] = cache["home"].astype(str).apply(norm_team)
                cache["away"] = cache["away"].astype(str).apply(norm_team)
                cache["home_pts"] = pd.to_numeric(cache["home_pts"], errors="coerce")
                cache["away_pts"] = pd.to_numeric(cache["away_pts"], errors="coerce")
                cache = cache.dropna(subset=["date", "home", "away", "home_pts", "away_pts"])
                print(f"[info] Loaded cached results: {RESULTS_CACHE_PATH} ({len(cache)} rows)")
            else:
                print(f"[warn] Cache exists but invalid. cols={list(cached.columns)} rows={len(cached)}")
        except Exception as e:
            print(f"[warn] Could not read cached results: {e}")

    headers = {"User-Agent": "Mozilla/5.0"}
    html = None
    last_err = None

    for attempt in range(3):
        try:
            r = requests.get(RESULTS_URL, timeout=45, headers=headers)
            r.raise_for_status()
            html = r.text
            break
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))

    if html is None:
        print(f"[warn] results fetch failed: {last_err} -> using cache only")
        return cache.reset_index(drop=True)

    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        print(f"[warn] pd.read_html failed: {e} -> using cache only")
        return cache.reset_index(drop=True)

    if not tables:
        print("[warn] No tables found on results page -> using cache only")
        return cache.reset_index(drop=True)

    df = tables[0].copy()
    cols = set(df.columns)

    out = pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    if {"Home", "Away", "HomeScore", "AwayScore"}.issubset(cols):
        if "Date" in cols:
            date_series = pd.to_datetime(df["Date"], errors="coerce")
        else:
            date_series = pd.Series([pd.NaT] * len(df))

        out = pd.DataFrame({
            "date": date_series.dt.strftime("%Y-%m-%d"),
            "home": df["Home"].astype(str).apply(norm_team),
            "away": df["Away"].astype(str).apply(norm_team),
            "home_pts": pd.to_numeric(df["HomeScore"], errors="coerce"),
            "away_pts": pd.to_numeric(df["AwayScore"], errors="coerce"),
        })

    elif {"Home Team", "Away Team", "Result"}.issubset(cols):
        def extract_scores(x: object) -> Tuple[float, float]:
            s = str(x)
            m = re.search(r"(\d+)\s*[-–]\s*(\d+)", s)
            if not m:
                return (np.nan, np.nan)
            return (float(m.group(1)), float(m.group(2)))

        scores = df["Result"].apply(extract_scores)

        if "Date" in cols:
            date_series = pd.to_datetime(df["Date"], errors="coerce")
        else:
            date_series = pd.Series([pd.NaT] * len(df))

        out = pd.DataFrame({
            "date": date_series.dt.strftime("%Y-%m-%d"),
            "home": df["Home Team"].astype(str).apply(norm_team),
            "away": df["Away Team"].astype(str).apply(norm_team),
            "home_pts": scores.apply(lambda t: t[0]),
            "away_pts": scores.apply(lambda t: t[1]),
        })

    else:
        print(f"[warn] Results table missing required columns. Found cols={list(df.columns)} -> using cache only")
        return cache.reset_index(drop=True)

    out = out.dropna(subset=["date", "home", "away", "home_pts", "away_pts"]).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["home_pts"] = pd.to_numeric(out["home_pts"], errors="coerce")
    out["away_pts"] = pd.to_numeric(out["away_pts"], errors="coerce")
    out = out.dropna(subset=["date", "home", "away", "home_pts", "away_pts"]).copy()

    print(f"[info] Web fetched results rows={len(out)}")

    merged = pd.concat([cache, out], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged["home"] = merged["home"].astype(str).apply(norm_team)
    merged["away"] = merged["away"].astype(str).apply(norm_team)
    merged["home_pts"] = pd.to_numeric(merged["home_pts"], errors="coerce")
    merged["away_pts"] = pd.to_numeric(merged["away_pts"], errors="coerce")
    merged = merged.dropna(subset=["date", "home", "away", "home_pts", "away_pts"])
    merged = merged.drop_duplicates(subset=["date", "home", "away"], keep="last").reset_index(drop=True)

    try:
        merged.to_csv(RESULTS_CACHE_PATH, index=False)
        print(f"[info] Cache updated: {RESULTS_CACHE_PATH} ({len(merged)} rows)")
    except Exception as e:
        print(f"[warn] Could not write cache: {e}")

    return merged

def fit_attack_defence(
    results: pd.DataFrame,
    teams: List[str],
    half_life_days: int = 56,
) -> Optional[Dict[str, object]]:
    """
    Weighted ridge least squares fit:
      HomePts = mu + home_adv + atk_home - def_away
      AwayPts = mu          + atk_away - def_home
    """
    if results is None or results.empty:
        return None

    results = results.dropna(subset=["home", "away", "home_pts", "away_pts"]).copy()
    if results.empty:
        return None

    if len(results) < 4:
        return None

    now = pd.Timestamp.now(tz=None).normalize()

    if "date" in results.columns:
        d = pd.to_datetime(results["date"], errors="coerce")
        age_days = (now - d).dt.days
        age_days = age_days.fillna(0).clip(lower=0)
        weights = (0.5 ** (age_days / float(half_life_days))).astype(float).values
    else:
        weights = np.ones(len(results), dtype=float)

    team_to_i = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    p = 2 + 2 * n_teams

    X_rows = []
    y_vals = []
    w_vals = []

    weights_by_pos = list(weights)

    for pos, (_, rrow) in enumerate(results.iterrows()):
        h = rrow["home"]
        a = rrow["away"]
        if h not in team_to_i or a not in team_to_i:
            continue

        w = float(weights_by_pos[pos]) if pos < len(weights_by_pos) else 1.0
        hi = team_to_i[h]
        ai = team_to_i[a]

        row = np.zeros(p)
        row[0] = 1.0
        row[1] = 1.0
        row[2 + hi] = 1.0
        row[2 + n_teams + ai] = -1.0
        X_rows.append(row)
        y_vals.append(float(rrow["home_pts"]))
        w_vals.append(w)

        row = np.zeros(p)
        row[0] = 1.0
        row[1] = 0.0
        row[2 + ai] = 1.0
        row[2 + n_teams + hi] = -1.0
        X_rows.append(row)
        y_vals.append(float(rrow["away_pts"]))
        w_vals.append(w)

    if len(y_vals) < 8:
        return None

    X = np.vstack(X_rows)
    y = np.array(y_vals)
    w = np.array(w_vals)

    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw

    ridge = 1.0
    XtX = Xw.T @ Xw + ridge * np.eye(p)
    Xty = Xw.T @ yw
    beta = np.linalg.solve(XtX, Xty)

    mu = float(beta[0])
    home_adv = float(beta[1])

    atk = beta[2:2 + n_teams].copy()
    dfn = beta[2 + n_teams:2 + 2 * n_teams].copy()

    atk -= atk.mean()
    dfn -= dfn.mean()

    atk_map = {t: float(atk[team_to_i[t]]) for t in teams}
    dfn_map = {t: float(dfn[team_to_i[t]]) for t in teams}

    return {"mu": mu, "home_adv": home_adv, "atk": atk_map, "dfn": dfn_map}

def load_adjustments(path: str = "adjustments.csv") -> Dict[str, Dict[str, float]]:
    """
    Returns: {team: {"atk": float, "def": float, "notes": str}}
    """
    try:
        df = pd.read_csv(path)
        out = {}
        for _, r in df.iterrows():
            team = str(r.get("team", "")).strip()
            if not team:
                continue
            out[team] = {
                "atk": float(r.get("atk_delta_pts", 0.0)),
                "def": float(r.get("def_delta_pts", 0.0)),
                "notes": str(r.get("notes", "")).strip(),
            }
        return out
    except Exception:
        return {}

def expected_points(model: Dict[str, object], home: str, away: str, venue: str, adj: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    mu = model["mu"]
    ha = model["home_adv"]
    atk = model["atk"]
    dfn = model["dfn"]

    home_pts = mu + ha + atk.get(home, 0.0) - dfn.get(away, 0.0)
    away_pts = mu + atk.get(away, 0.0) - dfn.get(home, 0.0)

    h_adj, a_adj = travel_points_adjustment(home, away, venue)
    home_pts += h_adj
    away_pts += a_adj

    home_pts += adj.get(home, {}).get("atk", 0.0)
    away_pts += adj.get(away, {}).get("atk", 0.0)

    away_pts += adj.get(home, {}).get("def", 0.0)
    home_pts += adj.get(away, {}).get("def", 0.0)

    return (
        max(4.0, min(40.0, home_pts)),
        max(4.0, min(40.0, away_pts)),
    )

def simulate_match_ad(
    model: Dict[str, object],
    home: str,
    away: str,
    venue: str,
    adj: Dict[str, Dict[str, float]],
    n: int = 20000,
    seed: int = 7
) -> Tuple[float, float, float, float]:
    random.seed(seed)
    hw = 0
    margins = []
    totals = []

    exp_home, exp_away = expected_points(model, home, away, venue, adj)
    sd = 8.5

    for _ in range(n):
        h = max(0, int(round(random.gauss(exp_home, sd) / 2.0) * 2))
        a = max(0, int(round(random.gauss(exp_away, sd) / 2.0) * 2))
        if h > a:
            hw += 1
        margins.append(h - a)
        totals.append(h + a)

    win_prob = hw / n
    exp_margin = sum(margins) / n
    exp_total = sum(totals) / n
    conf = min(0.80, 0.50 + abs(win_prob - 0.5) * 0.9)
    return win_prob, exp_margin, exp_total, conf

# ----------------------------
# TRY SCORERS (optional; will fallback if scrape fails)
# ----------------------------
def _try_probs_named(starters: Dict[int, str], team_exp_points: float) -> List[Tuple[str, float]]:
    exp_tries = max(1.0, team_exp_points / 4.2)
    weights_by_num = {2: 0.24, 5: 0.24, 1: 0.14, 3: 0.12, 4: 0.12, 11: 0.08, 12: 0.08}

    if not starters or len(starters) < 7:
        return []

    remaining_share = 1.0 - sum(weights_by_num.values())
    other_nums = [n for n in range(1, 14) if n not in weights_by_num]
    per_other = max(0.0, remaining_share / len(other_nums))

    out = []
    for num in range(1, 14):
        name = starters.get(num)
        if not name:
            continue
        share = weights_by_num.get(num, per_other)
        lam = exp_tries * share
        p = 1 - math.exp(-lam)
        out.append((name, p))

    out.sort(key=lambda x: x[1], reverse=True)
    return out[:3]

def _try_profiles_fallback(team_exp_points: float) -> List[Tuple[str, float]]:
    exp_tries = max(1.0, team_exp_points / 4.2)
    buckets = [("Winger", 0.44), ("Centre", 0.28), ("Fullback", 0.12), ("Edge", 0.10), ("Other", 0.06)]
    out = []
    for name, share in buckets:
        lam = exp_tries * share
        p = 1 - math.exp(-lam)
        out.append((name, p))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:3]

def _has_valid_named_teamlist(starters_by_team: Dict[str, Dict[int, str]], team: str) -> bool:
    players = starters_by_team.get(team, {})
    return isinstance(players, dict) and len(players) >= 7

def load_odds(path: str = "odds.csv") -> Dict[Tuple[str, str, str], Dict[str, float]]:
    try:
        df = pd.read_csv(path)
        out = {}
        for _, r in df.iterrows():
            date = str(r.get("date", "")).strip()
            home = norm_team(r.get("home", ""))
            away = norm_team(r.get("away", ""))

            if not date or not home or not away:
                continue

            home_odds = pd.to_numeric(r.get("home_odds"), errors="coerce")
            away_odds = pd.to_numeric(r.get("away_odds"), errors="coerce")

            out[(date, home, away)] = {
                "home_odds": float(home_odds) if pd.notna(home_odds) else float("nan"),
                "away_odds": float(away_odds) if pd.notna(away_odds) else float("nan"),
            }
        return out
    except Exception:
        return {}

def value_edge(model_prob: float, decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        return float("nan")
    return model_prob - (1.0 / decimal_odds)

# ----------------------------
# RATINGS PERSISTENCE
# ----------------------------
RATINGS_PATH = "ratings.json"

def load_saved_ratings(path: str = RATINGS_PATH) -> Optional[Dict[str, object]]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            model = json.load(f)
        if not isinstance(model, dict):
            return None
        if "mu" not in model or "home_adv" not in model or "atk" not in model or "dfn" not in model:
            return None
        return model
    except Exception:
        return None

def save_ratings(model: Dict[str, object], path: str = RATINGS_PATH) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        pass

def fixtures_from_odds_csv(path: str = "odds.csv") -> List[Match]:
    if not os.path.exists(path):
        return []
    try:
        o = pd.read_csv(path)
    except Exception:
        return []

    needed = {"date", "home", "away"}
    if not needed.issubset(set(o.columns)):
        return []

    fixtures: List[Match] = []
    for _, r in o.iterrows():
        date = str(r.get("date", "")).strip()
        home = norm_team(r.get("home", ""))
        away = norm_team(r.get("away", ""))

        if not date or not home or not away:
            continue

        fixtures.append(Match(
            date=date,
            kickoff_local="",
            home=home,
            away=away,
            venue=""
        ))

    fixtures.sort(key=lambda m: (m.date, m.kickoff_local))
    return fixtures

SITEMAP_INDEX = "https://www.nrl.com/sitemap/sitemap.xml"

def fetch_latest_teamlist_url() -> str:
    """
    Find the best team lists article by walking NRL's sitemap index.
    Strongly prefers current-season round-based team list articles.
    Handles .xml.gz child sitemaps properly (gzip decompress).
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
    }

    def fetch_xml(url: str) -> str:
        r = requests.get(url, timeout=30, headers=headers)
        r.raise_for_status()

        content = r.content
        is_gz = url.lower().endswith(".gz") or content[:2] == b"\x1f\x8b"
        if is_gz:
            try:
                content = gzip.decompress(content)
            except Exception as e:
                print(f"[warn] gzip decompress failed for {url}: {e}")
                return ""

        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return content.decode(errors="ignore")

    def extract_year_from_news_url(loc_low: str) -> int:
        m = re.search(r"/news/(\d{4})/", loc_low)
        return int(m.group(1)) if m else 0

    def score_url(loc_low: str, year: int) -> int:
        score = 0
        if ("team-lists" in loc_low) or ("nrl-team-lists" in loc_low):
            score += 10

        if "nrl-team-lists-round-" in loc_low or "team-lists-round-" in loc_low:
            score += 200
        elif "round-" in loc_low:
            score += 80

        if "las-vegas" in loc_low:
            score -= 120

        if "nrl-team-lists" in loc_low:
            score += 20

        current_year = datetime.now(SYDNEY_TZ).year
        if year == current_year:
            score += 300
        elif year == current_year - 1:
            score -= 200
        elif year and year < current_year - 1:
            score -= 500

        return score

    try:
        print(f"[debug] fetching sitemap index: {SITEMAP_INDEX}")
        idx_xml = fetch_xml(SITEMAP_INDEX)
        print(f"[debug] sitemap index chars={len(idx_xml)}")

        sitemap_locs = re.findall(r"<loc>\s*([^<]+)\s*</loc>", idx_xml, flags=re.IGNORECASE)
        print(f"[debug] sitemap index locs={len(sitemap_locs)}")

        if not sitemap_locs:
            return ""

        best_url = ""
        best_lastmod = ""
        best_score = -10_000
        hits = 0

        current_year = datetime.now(SYDNEY_TZ).year

        for sm_url in sitemap_locs:
            try:
                print(f"[debug] scanning child sitemap: {sm_url}")
                sm_xml = fetch_xml(sm_url)
                sm_low = sm_xml.lower()
                print(f"[debug] child chars={len(sm_xml)} has_team_terms={('team-lists' in sm_low) or ('team list' in sm_low)}")

                if ("team-lists" not in sm_low) and ("team list" not in sm_low) and ("nrl-team-lists" not in sm_low):
                    continue

                for blk in re.findall(r"<url>.*?</url>", sm_xml, flags=re.DOTALL | re.IGNORECASE):
                    m_loc = re.search(r"<loc>\s*([^<]+)\s*</loc>", blk, flags=re.IGNORECASE)
                    if not m_loc:
                        continue

                    loc = m_loc.group(1).strip()
                    loc_low = loc.lower()

                    if "/news/" not in loc_low:
                        continue
                    if ("team-lists" not in loc_low) and ("nrl-team-lists" not in loc_low) and ("team-list" not in loc_low):
                        continue

                    hits += 1

                    year = extract_year_from_news_url(loc_low)
                    if year and year != current_year:
                        continue

                    m_mod = re.search(r"<lastmod>\s*([^<]+)\s*</lastmod>", blk, flags=re.IGNORECASE)
                    lastmod = m_mod.group(1).strip() if m_mod else ""

                    s = score_url(loc_low, year)

                    if (s > best_score) or (s == best_score and lastmod > best_lastmod):
                        best_score = s
                        best_lastmod = lastmod
                        best_url = loc

            except Exception as e:
                print(f"[warn] sitemap child fetch failed: {sm_url} err={e}")

        print(f"[debug] teamlist hits={hits}")
        print(f"[debug] best_teamlist_url={best_url!r} score={best_score} lastmod={best_lastmod!r}")

        return best_url or ""

    except Exception as e:
        print(f"[warn] Could not auto-find TEAMLIST_URL via sitemap index: {e}")
        return ""

# ----------------------------
# BUILD OUTPUT
# ----------------------------
def build_predictions() -> pd.DataFrame:
    fixtures: List[Match] = []

    if MODE == "AUTO":
        fixtures = fixtures_from_odds_csv("odds.csv")

        if not fixtures:
            fixtures = fetch_upcoming_fixtures(days_ahead=21)

        if not fixtures:
            raise SystemExit("[stop] No upcoming fixtures found from odds.csv or the fixture feed. Not showing trial games.")
    else:
        raise SystemExit("[stop] MODE is not AUTO. Not publishing trial fixtures.")

    teams = sorted(list(TEAM_REGION.keys()))

    saved_model = load_saved_ratings()

    results = fetch_completed_results()
    results = (
        results
        .drop_duplicates(subset=["date", "home", "away"], keep="last")
        .reset_index(drop=True)
    )
    print(f"[debug] combined results rows={len(results)} (deduped)")

    fresh_model = fit_attack_defence(results, teams)

    if fresh_model:
        ad_model = fresh_model
        save_ratings(ad_model)
    elif saved_model:
        ad_model = saved_model
    else:
        ad_model = None

    teamlist_url = fetch_latest_teamlist_url()
    if teamlist_url:
        print(f"[info] Using team lists from: {teamlist_url}")
    else:
        print("[warn] No team list article found yet — using try-scorer fallback profiles.")

    starters_by_team = fetch_starters_by_team(teamlist_url) if teamlist_url else {}
    print("[debug] starters_by_team keys sample:", list(sorted(starters_by_team.keys()))[:30])
    for m in fixtures:
        print(f"[debug] {m.home} home_len={len(starters_by_team.get(m.home, {}))} | {m.away} away_len={len(starters_by_team.get(m.away, {}))}")

    manual_by_date = load_manual_teamlists(TEAMLISTS_CSV_PATH)
    for m in fixtures:
        manual_for_date = manual_by_date.get(m.date, {})
        if not manual_for_date:
            continue

        if m.home in manual_for_date:
            starters_by_team.setdefault(m.home, {}).update(manual_for_date[m.home])
        if m.away in manual_for_date:
            starters_by_team.setdefault(m.away, {}).update(manual_for_date[m.away])

    adj = load_adjustments()
    odds = load_odds()

    missing_keys = []
    for m in fixtures:
        key = (m.date, m.home, m.away)
        o = odds.get(key)
        if (
            not o
            or math.isnan(o.get("home_odds", float("nan")))
            or math.isnan(o.get("away_odds", float("nan")))
        ):
            missing_keys.append(key)

    if missing_keys:
        print("⚠️ Missing odds for these fixtures (date, home, away):")
        for k in missing_keys:
            print("  ", k)
        raise SystemExit("Stopping because odds are missing. Update odds.csv (or wait until Tuesday) then rerun.")

    rows = []

    MIN_EDGE = 0.03
    MIN_CONF = 0.55

    for m in fixtures:
        if ad_model:
            win_prob, exp_margin, exp_total, conf = simulate_match_ad(ad_model, m.home, m.away, m.venue, adj)
            exp_home_pts = (exp_total + exp_margin) / 2.0
            exp_away_pts = (exp_total - exp_margin) / 2.0
            rating_mode = "ATTACK_DEFENCE"
        else:
            win_prob, exp_margin, exp_total, conf = 0.50, 0.0, 40.0, 0.45
            exp_home_pts = exp_total / 2.0
            exp_away_pts = exp_total / 2.0
            rating_mode = "FALLBACK"

        if FORCE_TRY_FALLBACK:
            home_named = _try_profiles_fallback(exp_home_pts)
            away_named = _try_profiles_fallback(exp_away_pts)
        else:
            home_has_list = _has_valid_named_teamlist(starters_by_team, m.home)
            away_has_list = _has_valid_named_teamlist(starters_by_team, m.away)

            home_named = _try_probs_named(starters_by_team.get(m.home, {}), exp_home_pts) if home_has_list else []
            away_named = _try_probs_named(starters_by_team.get(m.away, {}), exp_away_pts) if away_has_list else []

            if not home_named:
                home_named = _try_profiles_fallback(exp_home_pts)
            if not away_named:
                away_named = _try_profiles_fallback(exp_away_pts)

        key = (m.date, m.home, m.away)
        o = odds.get(key, {})
        home_odds = o.get("home_odds", float("nan"))
        away_odds = o.get("away_odds", float("nan"))

        home_edge = float("nan")
        away_edge = float("nan")
        value_flag = ""
        pick = ""
        edge = float("nan")
        stake = 0.0

        if rating_mode != "ATTACK_DEFENCE":
            value_flag = "MODEL OFF (FALLBACK)"
        else:
            if not math.isnan(home_odds):
                home_edge = value_edge(win_prob, home_odds)
            if not math.isnan(away_odds):
                away_edge = value_edge(1 - win_prob, away_odds)

            if not math.isnan(home_edge) and home_edge >= 0.03:
                value_flag = f"HOME VALUE +{home_edge:.0%}"
            elif not math.isnan(away_edge) and away_edge >= 0.03:
                value_flag = f"AWAY VALUE +{away_edge:.0%}"

            best_side = ""
            best_edge = float("-inf")

            if not math.isnan(home_edge) and home_edge > best_edge:
                best_side = "HOME"
                best_edge = home_edge
            if not math.isnan(away_edge) and away_edge > best_edge:
                best_side = "AWAY"
                best_edge = away_edge

            if best_side and best_edge >= MIN_EDGE and conf >= MIN_CONF:
                pick = best_side
                edge = best_edge

                stake = 1.0
                if edge >= 0.10:
                    stake = 2.0
                if edge >= 0.15:
                    stake = 3.0

        rows.append({
            "mode": MODE,
            "rating_mode": rating_mode,
            "date": m.date,
            "kickoff_local": m.kickoff_local,
            "venue": m.venue,
            "home": m.home,
            "away": m.away,
            "home_win_prob": round(win_prob, 3),
            "exp_margin_home": round(exp_margin, 1),
            "exp_total": round(exp_total, 1),
            "confidence": round(conf, 2),
            "home_odds": home_odds,
            "away_odds": away_odds,
            "value_flag": value_flag,
            "pick": pick,
            "edge": round(edge, 3) if not math.isnan(edge) else 0.0,
            "stake": float(stake),
            "stake_units": float(stake),
            "stake_dollars": round(stake * UNIT_SIZE, 2),
            "home_top_try": " | ".join([f"{n} {p:.0%}" for n, p in home_named]),
            "away_top_try": " | ".join([f"{n} {p:.0%}" for n, p in away_named]),
            "teamlist_source": teamlist_url if starters_by_team else "fallback (no scrape)",
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        })

    df = pd.DataFrame(rows).sort_values(["date", "kickoff_local"]).reset_index(drop=True)

    # ---------------------------------
    # Keep only the current round
    # ---------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    round_start = df["date"].min()
    round_end = round_start + pd.Timedelta(days=3)
    df = df[(df["date"] >= round_start) & (df["date"] <= round_end)].copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # ---------------------------------
    # Round summary for logs
    # ---------------------------------
    round_label = f"Round window {round_start.strftime('%Y-%m-%d')} to {round_end.strftime('%Y-%m-%d')}"
    bet_count = int((pd.to_numeric(df.get("stake_units", 0), errors="coerce").fillna(0) > 0).sum())
    exposure = float(pd.to_numeric(df.get("stake_dollars", 0), errors="coerce").fillna(0).sum())
    avg_edge = float(pd.to_numeric(df.get("edge", 0), errors="coerce").fillna(0).replace(0, pd.NA).dropna().mean())

    print(f"[predict] {round_label}")
    print(f"[predict] current round fixtures={len(df)} bets={bet_count} exposure=${exposure:.2f} avg_edge={avg_edge:.3f}")

    return df

def load_results_csv(path: str) -> pd.DataFrame:
    """
    Loads results from a manual CSV like:
    date,home,away,home_pts,away_pts
    """
    if not os.path.exists(path):
        print(f"[warn] results file not found: {path}")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[warn] could not read {path}: {e}")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    needed = {"date", "home", "away", "home_pts", "away_pts"}
    if not needed.issubset(set(df.columns)):
        print(f"[warn] {path} missing required columns. Need {sorted(needed)}")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d")
    df["home"] = df["home"].astype(str).apply(norm_team)
    df["away"] = df["away"].astype(str).apply(norm_team)
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")

    df = df.dropna(subset=["date", "home", "away", "home_pts", "away_pts"])

    print(f"[info] Loaded {path}: {len(df)} rows")
    return df[["date", "home", "away", "home_pts", "away_pts"]]

if __name__ == "__main__":
    df = build_predictions()
    df.to_csv("predictions.csv", index=False)

    if "stake" in df.columns:
        stake_series = pd.to_numeric(df["stake"], errors="coerce").fillna(0.0)
        bet_count = int((stake_series > 0).sum())
        print(f"[predict] rows={len(df)} bets={bet_count} max_stake={stake_series.max()}")
    else:
        print(f"[predict] rows={len(df)} (no stake column)")
