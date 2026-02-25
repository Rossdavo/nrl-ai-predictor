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

# ----------------------------
# RUN MODE
# "TRIALS" = use hardcoded fixtures (not used in this version)
# "AUTO"   = pull upcoming fixtures automatically
# ----------------------------
MODE = "AUTO"

# ----------------------------
# Team lists (trials page) – optional
# ----------------------------

FORCE_TRY_FALLBACK = False  # set to False once Round 1 team lists are live

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
def _strip_html_to_text(html: str) -> str:
    html = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_starters_by_team(url: str) -> Dict[str, Dict[int, str]]:
    """
    Attempts to scrape named starters from an NRL team list article.
    Returns: { "TeamShortName": {1:"Name", 2:"Name", ... 13:"Name"} }
    """
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = _strip_html_to_text(r.text)

        pat = re.compile(r"for ([A-Za-z \-']+?) is number (\d{1,2}) ([A-Za-z \-'.]+)", re.IGNORECASE)

        starters: Dict[str, Dict[int, str]] = {}
        for team, num_s, name in pat.findall(text):
            team = norm_team(team.strip())
            num = int(num_s)
            name = name.strip()

            if not (1 <= num <= 13):
                continue

            name = re.sub(
                r"\b(Fullback|Winger|Centre|Five-Eighth|Halfback|Prop|Hooker|Second Row|2nd Row|Lock)\b.*$",
                "",
                name,
                flags=re.IGNORECASE,
            ).strip()

            if not name:
                continue

            starters.setdefault(team, {})
            starters[team][num] = name

        if starters:
            sample = sorted(starters.items(), key=lambda x: (-len(x[1]), x[0]))[:5]
            print("[info] teamlist scrape sample:", ", ".join([f"{t}:{len(p)}" for t, p in sample]))
        else:
            print("[warn] teamlist scrape returned 0 teams")

        return starters
    except Exception as e:
        print(f"[warn] teamlist scrape failed: {e}")
        return {}

# ----------------------------
# RESULTS INGEST (for Attack/Defence fitting)
# ----------------------------
def fetch_completed_results() -> pd.DataFrame:
    """
    Returns dataframe with columns: date, home, away, home_pts, away_pts

    Behaviour:
    - Try local cache FIRST (results_cache.csv) if it looks valid
    - Else fetch from RESULTS_URL and parse (supports multiple table formats)
    - If web fails, return empty
    """
    if os.path.exists(RESULTS_CACHE_PATH):
        try:
            cached = pd.read_csv(RESULTS_CACHE_PATH)
            needed = {"date", "home", "away", "home_pts", "away_pts"}

            if needed.issubset(set(cached.columns)) and len(cached) >= 4:
                cached["home"] = cached["home"].apply(norm_team)
                cached["away"] = cached["away"].apply(norm_team)
                print(f"[info] Using cached results: {RESULTS_CACHE_PATH} ({len(cached)} rows)")
                return cached
            else:
                print(f"[warn] Cache exists but invalid/too small. cols={list(cached.columns)} rows={len(cached)}")
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
        print(f"[warn] results fetch failed: {last_err}")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        print(f"[warn] pd.read_html failed: {e}")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    if not tables:
        print("[warn] No tables found on results page")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    df = tables[0].copy()
    cols = set(df.columns)

    if {"Home", "Away", "HomeScore", "AwayScore"}.issubset(cols):
        date_series = pd.to_datetime(df["Date"], errors="coerce") if "Date" in cols else pd.Series([pd.Timestamp.utcnow()] * len(df))
        out = pd.DataFrame({
            "date": date_series.dt.strftime("%Y-%m-%d"),
            "home": df["Home"].astype(str).apply(norm_team),
            "away": df["Away"].astype(str).apply(norm_team),
            "home_pts": pd.to_numeric(df["HomeScore"], errors="coerce"),
            "away_pts": pd.to_numeric(df["AwayScore"], errors="coerce"),
        }).dropna()

    elif {"Home Team", "Away Team", "Result"}.issubset(cols):
        def extract_scores(x: object) -> Tuple[float, float]:
            s = str(x)
            m = re.search(r"(\d+)\s*[-–]\s*(\d+)", s)
            if not m:
                return (np.nan, np.nan)
            return (float(m.group(1)), float(m.group(2)))

        scores = df["Result"].apply(extract_scores)
        date_series = pd.to_datetime(df["Date"], errors="coerce") if "Date" in cols else pd.Series([pd.Timestamp.utcnow()] * len(df))

        out = pd.DataFrame({
            "date": date_series.dt.strftime("%Y-%m-%d"),
            "home": df["Home Team"].astype(str).apply(norm_team),
            "away": df["Away Team"].astype(str).apply(norm_team),
            "home_pts": scores.apply(lambda t: t[0]),
            "away_pts": scores.apply(lambda t: t[1]),
        }).dropna()

    else:
        print(f"[warn] Results table missing required columns. Found cols={list(df.columns)}")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    print(f"[info] Web fetched results rows={len(out)}")
    try:
        if len(out) > 0:
            out.to_csv(RESULTS_CACHE_PATH, index=False)
            print(f"[info] Saved fetched results to {RESULTS_CACHE_PATH}")
    except Exception:
        pass

    return out

def load_results_csv(path: str) -> pd.DataFrame:
    """
    Loads results from a manual CSV like:
    date,home,away,home_pts,away_pts
    """
    if not os.path.exists(path):
        print(f"[warn] results file not found: {path}")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    df = pd.read_csv(path)

    needed = {"date", "home", "away", "home_pts", "away_pts"}
    if not needed.issubset(set(df.columns)):
        print(f"[warn] {path} missing required columns.")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d")
    df["home"] = df["home"].astype(str).apply(norm_team)
    df["away"] = df["away"].astype(str).apply(norm_team)
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")

    df = df.dropna(subset=["date", "home", "away", "home_pts", "away_pts"])

    print(f"[info] Loaded {path}: {len(df)} rows")
    return df[["date", "home", "away", "home_pts", "away_pts"]]

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
    p = 2 + 2 * n_teams  # [mu, home_adv, atk..., def...]

    X_rows = []
    y_vals = []
    w_vals = []

    # ensure we index weights correctly in results order
    weights_by_pos = list(weights)

    for pos, (_, rrow) in enumerate(results.iterrows()):
        h = rrow["home"]
        a = rrow["away"]
        if h not in team_to_i or a not in team_to_i:
            continue

        w = float(weights_by_pos[pos]) if pos < len(weights_by_pos) else 1.0
        hi = team_to_i[h]
        ai = team_to_i[a]

        # Home points equation
        row = np.zeros(p)
        row[0] = 1.0
        row[1] = 1.0
        row[2 + hi] = 1.0
        row[2 + n_teams + ai] = -1.0
        X_rows.append(row); y_vals.append(float(rrow["home_pts"])); w_vals.append(w)

        # Away points equation
        row = np.zeros(p)
        row[0] = 1.0
        row[1] = 0.0
        row[2 + ai] = 1.0
        row[2 + n_teams + hi] = -1.0
        X_rows.append(row); y_vals.append(float(rrow["away_pts"])); w_vals.append(w)

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
    atk_delta_pts: added to team expected points scored
    def_delta_pts: added to opponent expected points scored (i.e., worse defence => +ve)
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

    # player availability adjustments
    home_pts += adj.get(home, {}).get("atk", 0.0)
    away_pts += adj.get(away, {}).get("atk", 0.0)

    away_pts += adj.get(home, {}).get("def", 0.0)
    home_pts += adj.get(away, {}).get("def", 0.0)

    return (max(4.0, min(40.0, home_pts)),
            max(4.0, min(40.0, away_pts)))

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
    return isinstance(players, dict) and len(players) >= 10

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



SITEMAP_CURRENT_GZ = "https://www.nrl.com/sitemap/current.xml.gz"

def fetch_latest_teamlist_url() -> str:
    """
    Robust: pull the latest nrl-team-lists article from NRL's current sitemap.
    Avoids JS-rendered topic/search pages that requests can't see.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
    }

    try:
        print(f"[debug] fetching sitemap: {SITEMAP_CURRENT_GZ}")
        r = requests.get(SITEMAP_CURRENT_GZ, timeout=30, headers=headers)
        r.raise_for_status()

        print(f"[debug] sitemap http={r.status_code} bytes={len(r.content)}")

        # current.xml.gz is gzip compressed XML
        xml_bytes = gzip.decompress(r.content)
        xml = xml_bytes.decode("utf-8", errors="ignore")

        print(f"[debug] sitemap xml chars={len(xml)}")
        print(f"[debug] sitemap contains '/news/': {'/news/' in xml}")
        print(f"[debug] sitemap contains 'nrl-team-lists-': {'nrl-team-lists-' in xml}")

        # Extract <url> blocks containing a team lists article
        url_blocks = re.findall(r"<url>.*?</url>", xml, flags=re.DOTALL | re.IGNORECASE)
        print(f"[debug] sitemap url_blocks={len(url_blocks)}")

        best_url = ""
        best_lastmod = ""
        hits = 0

        for blk in url_blocks:
            m_loc = re.search(r"<loc>\s*([^<]+)\s*</loc>", blk, flags=re.IGNORECASE)
            m_mod = re.search(r"<lastmod>\s*([^<]+)\s*</lastmod>", blk, flags=re.IGNORECASE)

            if not m_loc:
                continue

            loc = m_loc.group(1).strip()

            # Only team list articles
            if "/news/" not in loc:
                continue
            if "nrl-team-lists-" not in loc:
                continue

            hits += 1
            lastmod = (m_mod.group(1).strip() if m_mod else "")

            # Pick the newest by lastmod (ISO strings compare lexicographically well)
            if lastmod > best_lastmod:
                best_lastmod = lastmod
                best_url = loc

        print(f"[debug] teamlist sitemap hits={hits}")
        print(f"[debug] best_teamlist_url={best_url!r} lastmod={best_lastmod!r}")

        return best_url or ""

    except Exception as e:
        print(f"[warn] Could not auto-find TEAMLIST_URL via sitemap: {e}")
        return ""
    except Exception as e:
        print(f"[warn] Could not auto-find TEAMLIST_URL via sitemap: {e}")
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

    results_2026 = fetch_completed_results()
    results_2025 = load_results_csv("data/results_2025.csv")
    results = pd.concat([results_2025, results_2026], ignore_index=True)

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

    adj = load_adjustments()
    odds = load_odds()

    # STOP EARLY IF ODDS MISSING (novice-safe)
    missing_keys = []
    for m in fixtures:
        key = (m.date, m.home, m.away)
        o = odds.get(key)
        if (not o
            or math.isnan(o.get("home_odds", float("nan")))
            or math.isnan(o.get("away_odds", float("nan")))):
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

        # Try scorers
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

        # Odds lookup
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

        # Output row (ALWAYS appended once per fixture)
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
            "home_top_try": " | ".join([f"{n} {p:.0%}" for n, p in home_named]),
            "away_top_try": " | ".join([f"{n} {p:.0%}" for n, p in away_named]),
            "teamlist_source": teamlist_url if starters_by_team else "fallback (no scrape)",
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        })

    df = pd.DataFrame(rows).sort_values(["date", "kickoff_local"])
    return df

if __name__ == "__main__":
    df = build_predictions()
    df.to_csv("predictions.csv", index=False)

    if "stake" in df.columns:
        stake_series = pd.to_numeric(df["stake"], errors="coerce").fillna(0.0)
        bet_count = int((stake_series > 0).sum())
        print(f"[predict] rows={len(df)} bets={bet_count} max_stake={stake_series.max()}")
    else:
        print(f"[predict] rows={len(df)} (no stake column)")
