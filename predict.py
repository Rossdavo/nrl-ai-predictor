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
# "TRIALS" = use hardcoded fixtures
# "AUTO"   = pull upcoming fixtures automatically
# ----------------------------
MODE = "AUTO"

# ----------------------------
# Team lists (trials page) – optional
# ----------------------------
TEAMLIST_URL = "https://www.nrl.com/news/2026/02/10/witzer-pre-season-challenge-team-lists-round-2/"

# ----------------------------
# Results source for ratings (Attack/Defence)
# ----------------------------
RESULTS_URL = "https://fixturedownload.com/results/nrl-2026"

RESULTS_CACHE_PATH = "results_cache.csv"

@dataclass
class Match:
    date: str  # YYYY-MM-DD
    kickoff_local: str  # HH:MM (Sydney/local)
    home: str
    away: str
    venue: str

# ----------------------------
# TRIAL FIXTURES (hardcoded)
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

# ----------------------------
# AUTO FIXTURE PULL (optional)
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

    # NZ travel (biggest)
    # Warriors playing away in Australia
    if a_reg == "NZ" and h_reg != "NZ":
        away_delta -= 1.6
        home_delta += 0.2

    # Australian team travelling to NZ (Warriors home)
    if h_reg == "NZ" and a_reg != "NZ":
        away_delta -= 1.2
        home_delta += 0.2

    # Cross-region within Australia (small)
    # ACT treated similar to NSW for travel purposes
    def norm(reg: str) -> str:
        return "NSW" if reg == "ACT" else reg

    h_norm = norm(h_reg)
    a_norm = norm(a_reg)

    if h_norm in {"NSW", "QLD", "VIC"} and a_norm in {"NSW", "QLD", "VIC"} and h_norm != a_norm:
        away_delta -= 0.6  # small fatigue/tempo penalty
        home_delta += 0.1

    return home_delta, away_delta
    
def fetch_upcoming_fixtures(days_ahead: int = 7) -> List[Match]:
    now = datetime.now(SYDNEY_TZ)
    end = now + pd.Timedelta(days=days_ahead)

    r = requests.get(
        FIXTURE_FEED_URL,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
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
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = _strip_html_to_text(r.text)

        pat = re.compile(r"for ([A-Za-z \-']+?) is number (\d{1,2}) ([A-Za-z \-'.]+)")
        starters: Dict[str, Dict[int, str]] = {}

        for team, num_s, name in pat.findall(text):
            team = team.strip()
            num = int(num_s)
            name = name.strip()
            if not (1 <= num <= 13):
                continue
            name = re.sub(
                r"\b(Fullback|Winger|Centre|Five-Eighth|Halfback|Prop|Hooker|2nd Row|Lock)\b.*$",
                "",
                name,
            ).strip()
            if not name:
                continue
            starters.setdefault(team, {})
            starters[team].setdefault(num, name)

        return starters
    except Exception:
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

    # 1) Local cache first
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
                print(f"[warn] Cache exists but is invalid/too small. cols={list(cached.columns)} rows={len(cached)}")

        except Exception as e:
            print(f"[warn] Could not read cached results: {e}")

    # 2) Fetch from website
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

    # FORMAT 1 (older structure)
    if {"Home", "Away", "HomeScore", "AwayScore"}.issubset(cols):

        date_series = pd.to_datetime(df["Date"], errors="coerce") if "Date" in cols else pd.Series([pd.Timestamp.utcnow()] * len(df))

        out = pd.DataFrame({
            "date": date_series.dt.strftime("%Y-%m-%d"),
            "home": df["Home"].astype(str).apply(norm_team),
            "away": df["Away"].astype(str).apply(norm_team),
            "home_pts": pd.to_numeric(df["HomeScore"], errors="coerce"),
            "away_pts": pd.to_numeric(df["AwayScore"], errors="coerce"),
        }).dropna()

    # FORMAT 2 (new structure: Home Team / Away Team / Result)
    elif {"Home Team", "Away Team", "Result"}.issubset(cols):

        def extract_scores(x):
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
        else:
            print("[info] Not saving empty web results to cache")
    except Exception:
        pass

    return out


# ------------------------------------------------
# MANUAL RESULTS CSV LOADER (e.g. results_2025.csv)
# ------------------------------------------------

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
    # 2) Web fetch
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
        print(f"[warn] results fetch failed: {last_err} — no cache available")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    # Parse HTML tables
    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        print(f"[warn] pd.read_html failed: {e}")
        tables = []

    if not tables:
        print("[warn] No tables found on results page")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    df = tables[0].copy()
    cols = set(df.columns)

    # ---------- FORMAT A (old): Home/Away/HomeScore/AwayScore ----------
    if {"Home", "Away", "HomeScore", "AwayScore"}.issubset(cols):
        date_series = pd.to_datetime(df["Date"], errors="coerce") if "Date" in cols else pd.Series([pd.Timestamp.utcnow()] * len(df))
        out = pd.DataFrame({
            "date": date_series.dt.strftime("%Y-%m-%d"),
            "home": df["Home"].astype(str).apply(norm_team),
            "away": df["Away"].astype(str).apply(norm_team),
            "home_pts": pd.to_numeric(df["HomeScore"], errors="coerce"),
            "away_pts": pd.to_numeric(df["AwayScore"], errors="coerce"),
        }).dropna(subset=["home_pts", "away_pts", "home", "away"])

    # ---------- FORMAT B (new): Home Team/Away Team/Result ----------
    elif {"Home Team", "Away Team", "Result"}.issubset(cols):
        # Result examples we handle: "24 - 16", "24–16", "24-16", "W 24-16", etc.
        def extract_scores(x: object) -> Tuple[float, float]:
            s = str(x)
            m = re.search(r"(\d+)\s*[-–]\s*(\d+)", s)
            if not m:
                return (np.nan, np.nan)
            return (float(m.group(1)), float(m.group(2)))

        date_series = pd.to_datetime(df["Date"], errors="coerce") if "Date" in cols else pd.Series([pd.Timestamp.utcnow()] * len(df))
        scores = df["Result"].apply(extract_scores)

        out = pd.DataFrame({
            "date": date_series.dt.strftime("%Y-%m-%d"),
            "home": df["Home Team"].astype(str).apply(norm_team),
            "away": df["Away Team"].astype(str).apply(norm_team),
            "home_pts": scores.apply(lambda t: t[0]),
            "away_pts": scores.apply(lambda t: t[1]),
        }).dropna(subset=["home_pts", "away_pts", "home", "away"])

    else:
        print(f"[warn] Results table missing required columns. Found cols={list(df.columns)}")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    print(f"[info] Web fetched results rows={len(out)}")
    if len(out):
        print(out.head(5).to_string(index=False))

    # Save cache for next run
    try:
        out.to_csv(RESULTS_CACHE_PATH, index=False)
        print(f"[info] Saved fetched results to {RESULTS_CACHE_PATH} ({len(out)} rows)")
    except Exception as e:
        print(f"[warn] Could not save cache: {e}")

    return out
def fit_attack_defence(
    results: pd.DataFrame,
    teams: List[str],
    half_life_days: int = 56,   # ~8 weeks
) -> Optional[Dict[str, object]]:
    """
    Weighted ridge least squares fit:
      HomePts = mu + home_adv + atk_home - def_away
      AwayPts = mu          + atk_away - def_home

    Recency weighting:
      weight = 0.5 ** (age_days / half_life_days)
    """
    if results is None or results.empty:
        return None

    results = results.dropna(subset=["home", "away", "home_pts", "away_pts"]).copy()
    if results.empty:
        return None

    # Reduce threshold so model can start earlier
    if len(results) < 4:
        return None

    now = pd.Timestamp.now(tz=None).normalize()

    # If date missing/unparseable, treat as age 0 (weight 1)
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

    for idx, rrow in results.iterrows():
        h = rrow["home"]
        a = rrow["away"]
        if h not in team_to_i or a not in team_to_i:
            continue

        w = float(weights[list(results.index).index(idx)]) if len(weights) == len(results) else 1.0
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

    # Centre (identifiability)
    atk -= atk.mean()
    dfn -= dfn.mean()

    atk_map = {t: float(atk[team_to_i[t]]) for t in teams}
    dfn_map = {t: float(dfn[team_to_i[t]]) for t in teams}

    return {"mu": mu, "home_adv": home_adv, "atk": atk_map, "dfn": dfn_map}
def load_adjustments(path: str = "adjustments.csv") -> Dict[str, Dict[str, float]]:
    """
    Reads manual adjustments file.
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

    # travel adjustment
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

def simulate_match_ad(model: Dict[str, object], home: str, away: str, venue: str, adj: Dict[str, Dict[str, float]], n: int = 20000, seed: int = 7) -> Tuple[float, float, float, float]:
    random.seed(seed)
    hw = 0
    margins = []
    totals = []

    exp_home, exp_away = expected_points(model, home, away, venue,adj)
    # score noise (keeps it realistic without needing a full Poisson conversion)
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
# TRY SCORERS (still optional; will fallback if scrape fails)
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
    """
    Positive means model_prob > implied_prob (value).
    """
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
            kickoff_local="",   # unknown from odds file
            home=home,
            away=away,
            venue=""            # unknown from odds file
        ))

    fixtures.sort(key=lambda m: (m.date, m.kickoff_local))
    return fixtures
# ----------------------------
# BUILD OUTPUT
# ----------------------------
def build_predictions():

   # --- Fixture selection ---
    fixtures: List[Match] = []

    if MODE == "AUTO":
        # 1) Prefer odds.csv fixtures (fastest + best signal that markets are live)
        fixtures = fixtures_from_odds_csv("odds.csv")

        # 2) If no odds fixtures yet, try the feed
        if not fixtures:
            fixtures = fetch_upcoming_fixtures(days_ahead=21)

        # 3) If still nothing, STOP (do not fall back to trials)
        if not fixtures:
            raise SystemExit("[stop] No upcoming fixtures found from odds.csv or the fixture feed. Not showing trial games.")
    else:
        raise SystemExit("[stop] MODE is not AUTO. Not publishing trial fixtures.") 

    teams = ALL_TEAMS
    # --- Team selection ---
    teams = sorted(list(TEAM_REGION.keys()))  # Define teams directly; avoids shadowing issues
    # Load saved ratings first (so we can still run if results fetch is empty/slow)
    saved_model = load_saved_ratings()
    results_2026 = fetch_completed_results()
    results_2025 = load_results_csv("data/results_2025.csv")

    results = pd.concat([results_2025, results_2026], ignore_index=True)

    print(f"[debug] combined results rows={len(results)}")

    fresh_model = fit_attack_defence(results, teams)
    # Prefer freshly fitted ratings; otherwise fall back to saved ratings
    if fresh_model:
        ad_model = fresh_model
        save_ratings(ad_model)
    elif saved_model:
        ad_model = saved_model
    else:
        ad_model = None
    starters_by_team = fetch_starters_by_team(TEAMLIST_URL)
    adj = load_adjustments()
    odds = load_odds()

    # -----------------------------------------
    # STOP EARLY IF ODDS MISSING (novice-safe)
    # -----------------------------------------
    missing_keys = []
    for m in fixtures:
        key = (m.date, m.home, m.away)
        o = odds.get(key)
        if not o or math.isnan(o.get("home_odds", float("nan"))) or math.isnan(o.get("away_odds", float("nan"))):
            missing_keys.append(key)

    if missing_keys:
        print("⚠️ Missing odds for these fixtures (date, home, away):")
        for k in missing_keys:
            print("  ", k)
        raise SystemExit("Stopping because odds are missing. Update odds.csv (or wait until Tuesday) then rerun.")
    rows = []
    for m in fixtures:
        if ad_model:
            win_prob, exp_margin, exp_total, conf = simulate_match_ad(
                ad_model, m.home, m.away, m.venue, adj
            )
            exp_home_pts = (exp_total + exp_margin) / 2.0
            exp_away_pts = (exp_total - exp_margin) / 2.0
            rating_mode = "ATTACK_DEFENCE"
        else:
            win_prob, exp_margin, exp_total, conf = 0.50, 0.0, 40.0, 0.45
            exp_home_pts = exp_total / 2.0
            exp_away_pts = exp_total / 2.0
            rating_mode = "FALLBACK"
        home_named = _try_probs_named(starters_by_team.get(m.home, {}), exp_home_pts)
        away_named = _try_probs_named(starters_by_team.get(m.away, {}), exp_away_pts)
        if not home_named:
            home_named = _try_profiles_fallback(exp_home_pts)
        if not away_named:
            away_named = _try_profiles_fallback(exp_away_pts)
        # Odds + value detection
        key = (m.date, m.home, m.away)
        o = odds.get(key, {})
        home_odds = o.get("home_odds", float("nan"))
        away_odds = o.get("away_odds", float("nan"))

        # ✅ If we're in FALLBACK mode, do NOT calculate value
if rating_mode != "ATTACK_DEFENCE":
    home_edge = float("nan")
    away_edge = float("nan")
    value_flag = "MODEL OFF (FALLBACK)"
else:
    home_edge = value_edge(win_prob, home_odds) if not math.isnan(home_odds) else float("nan")
    away_edge = value_edge(1 - win_prob, away_odds) if not math.isnan(away_odds) else float("nan")

    value_flag = ""
    if not math.isnan(home_edge) and home_edge >= 0.03:
        value_flag = f"HOME VALUE +{home_edge:.0%}"
    elif not math.isnan(away_edge) and away_edge >= 0.03:
        value_flag = f"AWAY VALUE +{away_edge:.0%}"
        if not math.isnan(home_edge) and home_edge >= 0.03:
            value_flag = f"HOME VALUE +{home_edge:.0%}"
        elif not math.isnan(away_edge) and away_edge >= 0.03:
            value_flag = f"AWAY VALUE +{away_edge:.0%}"
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
            "home_top_try": " | ".join([f"{n} {p:.0%}" for n, p in home_named]),
            "away_top_try": " | ".join([f"{n} {p:.0%}" for n, p in away_named]),
            "teamlist_source": TEAMLIST_URL if starters_by_team else "fallback (no scrape)",
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        })
    df = pd.DataFrame(rows).sort_values(["date", "kickoff_local"])
    return df
if __name__ == "__main__":
    df = build_predictions()
    df.to_csv("predictions.csv", index=False)
    print(df.to_string(index=False))
