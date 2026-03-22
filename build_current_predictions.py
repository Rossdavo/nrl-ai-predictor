print("[predict] predict.py loaded")

import math
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional
from io import StringIO
import json
import os

import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo


# ----------------------------
# BANKROLL / STAKING
# ----------------------------
BANKROLL = 200.0
UNIT_PCT = 0.05
UNIT_SIZE = round(BANKROLL * UNIT_PCT, 2)

MAX_ROUND_EXPOSURE_PCT = 0.50
MAX_ROUND_EXPOSURE = round(BANKROLL * MAX_ROUND_EXPOSURE_PCT, 2)

MAX_SINGLE_BET_PCT = 0.25
MAX_SINGLE_BET = round(BANKROLL * MAX_SINGLE_BET_PCT, 2)

MIN_BET_SHORT = 20.0   # odds < 2.00
MIN_BET_DOG = 10.0     # odds >= 2.00

TARGET_MIN_BETS = 3
TARGET_MAX_BETS = 5

print(f"[predict] bankroll=${BANKROLL} | unit=${UNIT_SIZE}")
print(f"[predict] max_round_exposure=${MAX_ROUND_EXPOSURE} | max_single_bet=${MAX_SINGLE_BET}")


# ----------------------------
# RUN MODE
# ----------------------------
MODE = "AUTO"


# ----------------------------
# Results sources for ratings
# ----------------------------
RESULTS_URLS = {
    2026: "https://fixturedownload.com/results/nrl-2026",
    2025: "https://fixturedownload.com/results/nrl-2025",
}
RESULTS_CACHE_PATH = "results_cache.csv"


# ----------------------------
# AUTO FIXTURE PULL
# ----------------------------
FIXTURE_FEED_URL = "https://fixturedownload.com/feed/json/nrl-2026"
SYDNEY_TZ = ZoneInfo("Australia/Sydney")


# ----------------------------
# MODEL SETTINGS
# ----------------------------
YEAR_WEIGHTS = {
    2026: 1.00,
    2025: 0.32,
}
RECENCY_HALF_LIFE_DAYS = 35

UPSET_MANUAL_PATH = "upset_flags.csv"
UPSET_PROB_SHIFT_PER_POINT = 0.0125
UPSET_PROB_SHIFT_CAP = 0.05
UPSET_FLAG_THRESHOLD = 2.0


TEAM_NAME_NORMALISE = {
    "Canterbury Bulldogs": "Bulldogs",
    "Canterbury-Bankstown Bulldogs": "Bulldogs",
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
    "Cronulla-Sutherland Sharks": "Sharks",
    "Gold Coast Titans": "Titans",
    "Manly Warringah Sea Eagles": "Sea Eagles",
    "Manly-Warringah Sea Eagles": "Sea Eagles",
    "Canberra Raiders": "Raiders",
    "South Sydney Rabbitohs": "Rabbitohs",
    "Dolphins": "Dolphins",
    "The Dolphins": "Dolphins",
    "Wests Tigers": "Wests Tigers",
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
    "Dolphins": "Dolphins",
}

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


# ----------------------------
# Futures market prior / team tiers
# ----------------------------
OUTRIGHT_ODDS = {
    "Panthers": 5.20,
    "Storm": 6.00,
    "Broncos": 8.50,
    "Roosters": 8.50,
    "Bulldogs": 13.00,
    "Sharks": 16.00,
    "Warriors": 16.00,
    "Rabbitohs": 18.00,
    "Raiders": 20.00,
    "Dolphins": 31.00,
    "Eels": 34.00,
    "Wests Tigers": 34.00,
    "Knights": 51.00,
    "Sea Eagles": 81.00,
    "Cowboys": 81.00,
    "Dragons": 81.00,
    "Titans": 101.00,
}

PREMIUM_HOME_TEAMS = {"Storm", "Raiders", "Sharks", "Eels", "Warriors", "Knights"}
ELITE_HOME_TEAMS = {"Storm", "Raiders"}
WEAK_HOME_TEAMS = {"Cowboys", "Titans", "Dragons", "Sea Eagles"}

TEAM_HOME_EDGE_FLOOR = {
    "Storm": 1.80,
    "Raiders": 1.45,
    "Sharks": 1.00,
    "Eels": 1.00,
    "Warriors": 1.10,
    "Knights": 0.75,
}

TEAM_HOME_EDGE_CAP = {
    "Cowboys": 0.60,
    "Titans": 0.45,
    "Dragons": 0.60,
    "Sea Eagles": 0.75,
}


@dataclass
class Match:
    date: str
    kickoff_local: str
    home: str
    away: str
    venue: str


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def norm_team(name: str) -> str:
    name = str(name).strip()
    return TEAM_NAME_NORMALISE.get(name, name)


def outright_strength_score(team: str) -> float:
    odds = float(OUTRIGHT_ODDS.get(team, 34.0))
    min_odds = min(OUTRIGHT_ODDS.values())
    max_odds = max(OUTRIGHT_ODDS.values())
    return (math.log(max_odds) - math.log(odds)) / (math.log(max_odds) - math.log(min_odds))


def apply_outright_strength_adjustment(home_prob: float, home: str, away: str) -> float:
    home_s = outright_strength_score(home)
    away_s = outright_strength_score(away)
    diff = home_s - away_s

    home_prob += diff * 0.045

    if home in ELITE_HOME_TEAMS:
        home_prob += 0.012
    elif home in PREMIUM_HOME_TEAMS:
        home_prob += 0.006

    return min(0.92, max(0.08, home_prob))


def build_current_season_record(results: pd.DataFrame, season_year: int = 2026) -> Dict[str, Dict[str, float]]:
    out = {t: {"wins": 0.0, "games": 0.0} for t in ALL_TEAMS}
    if results is None or results.empty:
        return out

    df = results.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df[df["date"].dt.year == season_year].copy()

    for _, r in df.iterrows():
        home = r["home"]
        away = r["away"]
        hp = float(r["home_pts"])
        ap = float(r["away_pts"])

        if home in out:
            out[home]["games"] += 1.0
            if hp > ap:
                out[home]["wins"] += 1.0

        if away in out:
            out[away]["games"] += 1.0
            if ap > hp:
                out[away]["wins"] += 1.0

    return out


def apply_early_season_matchup_moderation(
    home_prob: float,
    home: str,
    away: str,
    season_record: Dict[str, Dict[str, float]],
) -> float:
    h = season_record.get(home, {"wins": 0.0, "games": 0.0})
    a = season_record.get(away, {"wins": 0.0, "games": 0.0})

    h_wins, h_games = float(h["wins"]), float(h["games"])
    a_wins, a_games = float(a["wins"]), float(a["games"])

    both_winless = h_games >= 2 and a_games >= 2 and h_wins == 0 and a_wins == 0
    weak_vs_weak = OUTRIGHT_ODDS.get(home, 34.0) >= 60 and OUTRIGHT_ODDS.get(away, 34.0) >= 60

    if both_winless:
        home_prob = 0.5 + (home_prob - 0.5) * 0.60

        if weak_vs_weak:
            home_prob = min(max(home_prob, 0.46), 0.54)
        else:
            home_prob = min(max(home_prob, 0.44), 0.56)

    return min(0.92, max(0.08, home_prob))


def travel_points_adjustment(home: str, away: str, venue: str) -> Tuple[float, float]:
    h_reg = TEAM_REGION.get(home, "UNK")
    a_reg = TEAM_REGION.get(away, "UNK")

    home_delta = 0.0
    away_delta = 0.0

    if a_reg == "NZ" and h_reg != "NZ":
        away_delta -= 1.6
        home_delta += 0.2
    if h_reg == "NZ" and a_reg != "NZ":
        away_delta -= 1.2
        home_delta += 0.2

    def norm(reg: str) -> str:
        return "NSW" if reg == "ACT" else reg

    h_norm = norm(h_reg)
    a_norm = norm(a_reg)

    if h_norm in {"NSW", "QLD", "VIC"} and a_norm in {"NSW", "QLD", "VIC"} and h_norm != a_norm:
        away_delta -= 0.6
        home_delta += 0.1

    return home_delta, away_delta


def fetch_upcoming_fixtures(days_ahead: int = 21) -> List[Match]:
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
            venue="",
        ))

    fixtures.sort(key=lambda m: (m.date, m.kickoff_local))
    return fixtures


def _dedupe_fixtures(fixtures: List[Match]) -> List[Match]:
    seen = set()
    out: List[Match] = []
    for m in sorted(fixtures, key=lambda x: (x.date, x.kickoff_local, x.home, x.away)):
        key = (m.date, m.home, m.away)
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def _filter_current_round_fixtures(fixtures: List[Match]) -> List[Match]:
    if not fixtures:
        return fixtures

    dates = pd.to_datetime([m.date for m in fixtures], errors="coerce")
    dates = pd.Series(dates).dropna().sort_values()
    if dates.empty:
        return fixtures

    round_start = dates.min()
    round_end = round_start + pd.Timedelta(days=3)

    out = []
    for m in fixtures:
        d = pd.to_datetime(m.date, errors="coerce")
        if pd.isna(d):
            continue
        if round_start <= d <= round_end:
            out.append(m)

    return out


def _extract_results_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)

    if {"Home", "Away", "HomeScore", "AwayScore"}.issubset(cols):
        if "Date" in cols:
            date_series = pd.to_datetime(df["Date"], errors="coerce")
        else:
            date_series = pd.Series([pd.NaT] * len(df))

        return pd.DataFrame({
            "date": date_series.dt.strftime("%Y-%m-%d"),
            "home": df["Home"].astype(str).apply(norm_team),
            "away": df["Away"].astype(str).apply(norm_team),
            "home_pts": pd.to_numeric(df["HomeScore"], errors="coerce"),
            "away_pts": pd.to_numeric(df["AwayScore"], errors="coerce"),
        })

    if {"Home Team", "Away Team", "Result"}.issubset(cols):
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

        return pd.DataFrame({
            "date": date_series.dt.strftime("%Y-%m-%d"),
            "home": df["Home Team"].astype(str).apply(norm_team),
            "away": df["Away Team"].astype(str).apply(norm_team),
            "home_pts": scores.apply(lambda t: t[0]),
            "away_pts": scores.apply(lambda t: t[1]),
        })

    colmap = {str(c).strip().lower(): c for c in df.columns}
    required = {"date", "home team", "away team", "result"}
    if required.issubset(set(colmap.keys())):
        def extract_scores_3(x: object) -> Tuple[float, float]:
            s = str(x)
            m = re.search(r"(\d+)\s*[-–]\s*(\d+)", s)
            if not m:
                return (np.nan, np.nan)
            return (float(m.group(1)), float(m.group(2)))

        scores = df[colmap["result"]].apply(extract_scores_3)
        return pd.DataFrame({
            "date": pd.to_datetime(df[colmap["date"]], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d"),
            "home": df[colmap["home team"]].astype(str).apply(norm_team),
            "away": df[colmap["away team"]].astype(str).apply(norm_team),
            "home_pts": scores.apply(lambda t: t[0]),
            "away_pts": scores.apply(lambda t: t[1]),
        })

    return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])


def fetch_results_for_year(year: int, url: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    html = None
    last_err = None

    for attempt in range(3):
        try:
            r = requests.get(url, timeout=45, headers=headers)
            r.raise_for_status()
            html = r.text
            break
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))

    if html is None:
        print(f"[warn] results fetch failed for {year}: {last_err}")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        print(f"[warn] pd.read_html failed for {year}: {e}")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    if not tables:
        print(f"[warn] No tables found on results page for {year}")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])

    best = pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts"])
    for t in tables:
        extracted = _extract_results_table(t)
        if len(extracted) > len(best):
            best = extracted.copy()

    if best.empty:
        print(f"[warn] Could not extract results table for {year}")
        return best

    best = best.dropna(subset=["date", "home", "away", "home_pts", "away_pts"]).copy()
    best["date"] = pd.to_datetime(best["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    best["home_pts"] = pd.to_numeric(best["home_pts"], errors="coerce")
    best["away_pts"] = pd.to_numeric(best["away_pts"], errors="coerce")
    best = best.dropna(subset=["date", "home", "away", "home_pts", "away_pts"]).copy()
    best["season_year"] = year

    print(f"[info] Web fetched results rows={len(best)} for {year}")
    return best


def fetch_completed_results() -> pd.DataFrame:
    needed = {"date", "home", "away", "home_pts", "away_pts"}
    cache = pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts", "season_year"])

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
                if "season_year" not in cache.columns:
                    cache["season_year"] = pd.to_datetime(cache["date"], errors="coerce").dt.year
                cache["season_year"] = pd.to_numeric(cache["season_year"], errors="coerce")
                cache = cache.dropna(subset=["date", "home", "away", "home_pts", "away_pts"])
                cache = cache[cache["season_year"].fillna(0) >= 2025].copy()
                print(f"[info] Loaded cached results: {RESULTS_CACHE_PATH} ({len(cache)} rows)")
            else:
                print(f"[warn] Cache exists but invalid. cols={list(cached.columns)} rows={len(cached)}")
        except Exception as e:
            print(f"[warn] Could not read cached results: {e}")

    fetched_frames = []
    for year, url in RESULTS_URLS.items():
        yr_df = fetch_results_for_year(year, url)
        if not yr_df.empty:
            fetched_frames.append(yr_df)

    if fetched_frames:
        web_df = pd.concat(fetched_frames, ignore_index=True)
    else:
        web_df = pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts", "season_year"])

    merged = pd.concat([cache, web_df], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged["season_year"] = pd.to_datetime(merged["date"], errors="coerce").dt.year
    merged["home"] = merged["home"].astype(str).apply(norm_team)
    merged["away"] = merged["away"].astype(str).apply(norm_team)
    merged["home_pts"] = pd.to_numeric(merged["home_pts"], errors="coerce")
    merged["away_pts"] = pd.to_numeric(merged["away_pts"], errors="coerce")
    merged = merged.dropna(subset=["date", "home", "away", "home_pts", "away_pts"]).copy()
    merged = merged[merged["season_year"].fillna(0) >= 2025].copy()
    merged = merged.drop_duplicates(subset=["date", "home", "away"], keep="last").reset_index(drop=True)
    merged = merged.sort_values(["date", "home", "away"]).reset_index(drop=True)

    try:
        merged.to_csv(RESULTS_CACHE_PATH, index=False)
        print(f"[info] Cache updated: {RESULTS_CACHE_PATH} ({len(merged)} rows)")
    except Exception as e:
        print(f"[warn] Could not write cache: {e}")

    return merged


def fit_attack_defence(
    results: pd.DataFrame,
    teams: List[str],
    half_life_days: int = RECENCY_HALF_LIFE_DAYS,
    year_weights: Optional[Dict[int, float]] = None,
) -> Optional[Dict[str, object]]:
    if results is None or results.empty:
        return None

    results = results.dropna(subset=["home", "away", "home_pts", "away_pts"]).copy()
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(subset=["date"]).copy()
    results = results[results["date"].dt.year >= 2025].copy()

    if results.empty or len(results) < 8:
        return None

    year_weights = year_weights or YEAR_WEIGHTS
    now = pd.Timestamp.now(tz=None).normalize()
    age_days = (now - results["date"]).dt.days.fillna(0).clip(lower=0).astype(float)
    recency_weights = (0.5 ** (age_days / float(half_life_days))).astype(float)

    season_weights = results["date"].dt.year.map(lambda y: year_weights.get(int(y), 0.0)).astype(float)
    weights = (recency_weights * season_weights).astype(float)

    results = results[weights > 0].copy()
    weights = weights[weights > 0].values

    if len(results) < 8:
        return None

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

    if len(y_vals) < 12:
        return None

    X = np.vstack(X_rows)
    y = np.array(y_vals, dtype=float)
    w = np.array(w_vals, dtype=float)

    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw

    ridge = 1.2
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

    return {
        "mu": mu,
        "home_adv": home_adv,
        "atk": atk_map,
        "dfn": dfn_map,
        "rows_used": len(results),
    }


def build_team_home_ground_edges(results: pd.DataFrame, teams: List[str]) -> Dict[str, float]:
    out = {t: 0.0 for t in teams}
    if results is None or results.empty:
        return out

    df = results.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["season_year"] = df["date"].dt.year

    for team in teams:
        home = df[df["home"] == team].copy()
        away = df[df["away"] == team].copy()

        if home.empty or away.empty:
            continue

        home["w"] = home["season_year"].map(lambda y: YEAR_WEIGHTS.get(int(y), 0.0)).fillna(0.0)
        away["w"] = away["season_year"].map(lambda y: YEAR_WEIGHTS.get(int(y), 0.0)).fillna(0.0)

        home = home[home["w"] > 0].copy()
        away = away[away["w"] > 0].copy()

        if len(home) < 3 or len(away) < 3:
            continue

        home_win = np.average((home["home_pts"] > home["away_pts"]).astype(float), weights=home["w"])
        away_win = np.average((away["away_pts"] > away["home_pts"]).astype(float), weights=away["w"])

        delta = float(home_win - away_win)
        pts = delta * 5.0
        pts = float(max(-1.2, min(2.2, pts)))

        if team in TEAM_HOME_EDGE_FLOOR:
            pts = max(pts, TEAM_HOME_EDGE_FLOOR[team])

        if team in TEAM_HOME_EDGE_CAP:
            pts = min(pts, TEAM_HOME_EDGE_CAP[team])

        out[team] = pts

    return out


def load_adjustments(path: str = "adjustments.csv") -> Dict[str, Dict[str, float]]:
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


def expected_points(
    model: Dict[str, object],
    home: str,
    away: str,
    venue: str,
    adj: Dict[str, Dict[str, float]],
) -> Tuple[float, float]:
    mu = model["mu"]
    ha = model["home_adv"]
    atk = model["atk"]
    dfn = model["dfn"]
    home_ground_edges = model.get("team_home_edge", {})

    home_pts = mu + ha + atk.get(home, 0.0) - dfn.get(away, 0.0)
    away_pts = mu + atk.get(away, 0.0) - dfn.get(home, 0.0)

    h_adj, a_adj = travel_points_adjustment(home, away, venue)
    home_pts += h_adj
    away_pts += a_adj

    hge = float(home_ground_edges.get(home, 0.0))
    home_pts += hge
    away_pts -= hge * 0.15

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
    seed: int = 7,
) -> Tuple[float, float, float, float]:
    random.seed(seed)
    hw = 0
    margins = []
    totals = []

    exp_home, exp_away = expected_points(model, home, away, venue, adj)
    sd = 8.2

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
    conf = min(0.78, 0.50 + abs(win_prob - 0.5) * 0.72)

    return win_prob, exp_margin, exp_total, conf


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


def fair_probs_from_odds(home_odds: float, away_odds: float) -> Tuple[float, float]:
    if any(math.isnan(x) for x in [home_odds, away_odds]):
        return (float("nan"), float("nan"))
    if home_odds <= 1.0 or away_odds <= 1.0:
        return (float("nan"), float("nan"))

    imp_home = 1.0 / home_odds
    imp_away = 1.0 / away_odds
    total = imp_home + imp_away
    if total <= 0:
        return (float("nan"), float("nan"))

    return imp_home / total, imp_away / total


def value_edge(model_prob: float, decimal_odds: float) -> float:
    if decimal_odds <= 1.0 or math.isnan(decimal_odds):
        return float("nan")
    return model_prob - (1.0 / decimal_odds)


def compress_prob(p: float) -> float:
    p = min(0.95, max(0.05, p))
    dist = abs(p - 0.5)

    if dist < 0.06:
        factor = 0.94
    elif dist < 0.12:
        factor = 0.91
    else:
        factor = 0.88

    return min(0.95, max(0.05, 0.5 + (p - 0.5) * factor))


def load_saved_ratings(path: str = "ratings.json") -> Optional[Dict[str, object]]:
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


def save_ratings(model: Dict[str, object], path: str = "ratings.json") -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        pass


def load_manual_upset_flags(path: str = UPSET_MANUAL_PATH) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    out = {}
    for _, r in df.iterrows():
        date = str(r.get("date", "")).strip()
        home = norm_team(r.get("home", ""))
        away = norm_team(r.get("away", ""))
        upset_team = norm_team(r.get("upset_team", ""))
        if not date or not home or not away:
            continue

        score = pd.to_numeric(r.get("manual_upset_score", 0.0), errors="coerce")
        notes = str(r.get("notes", "")).strip()

        out[(date, home, away)] = {
            "upset_team": upset_team if upset_team in {home, away} else "",
            "manual_upset_score": float(score) if pd.notna(score) else 0.0,
            "notes": notes,
        }
    return out


def build_recent_form_stats(results: pd.DataFrame, teams: List[str], recent_n: int = 5) -> Dict[str, Dict[str, float]]:
    out = {t: {"recent_margin": 0.0, "recent_win_rate": 0.5, "games": 0.0} for t in teams}
    if results is None or results.empty:
        return out

    df = results.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    rows = []
    for _, r in df.iterrows():
        home = r["home"]
        away = r["away"]
        hp = float(r["home_pts"])
        ap = float(r["away_pts"])
        rows.append({"date": r["date"], "team": home, "margin": hp - ap, "win": 1.0 if hp > ap else 0.0})
        rows.append({"date": r["date"], "team": away, "margin": ap - hp, "win": 1.0 if ap > hp else 0.0})

    tf = pd.DataFrame(rows)
    for team in teams:
        tdf = tf[tf["team"] == team].sort_values("date", ascending=False).head(recent_n).copy()
        if tdf.empty:
            continue

        tdf["yr_weight"] = tdf["date"].dt.year.map(lambda y: YEAR_WEIGHTS.get(int(y), 0.0)).fillna(0.0)
        tdf = tdf[tdf["yr_weight"] > 0].copy()
        if tdf.empty:
            continue

        weights = np.linspace(1.25, 0.85, len(tdf))
        weights = weights * tdf["yr_weight"].values
        if weights.sum() <= 0:
            continue

        out[team] = {
            "recent_margin": float(np.average(tdf["margin"].values, weights=weights)),
            "recent_win_rate": float(np.average(tdf["win"].values, weights=weights)),
            "games": float(len(tdf)),
        }
    return out


def build_team_volatility(results: pd.DataFrame, teams: List[str], recent_n: int = 6) -> Dict[str, float]:
    out = {t: 0.0 for t in teams}
    if results is None or results.empty:
        return out

    df = results.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    rows = []
    for _, r in df.iterrows():
        home = r["home"]
        away = r["away"]
        hp = float(r["home_pts"])
        ap = float(r["away_pts"])
        rows.append({"date": r["date"], "team": home, "margin": hp - ap})
        rows.append({"date": r["date"], "team": away, "margin": ap - hp})

    tf = pd.DataFrame(rows)

    for team in teams:
        tdf = tf[tf["team"] == team].sort_values("date", ascending=False).head(recent_n).copy()
        if len(tdf) >= 3:
            out[team] = float(np.std(tdf["margin"].values))
    return out


def recent_h2h_margin(results: pd.DataFrame, home: str, away: str, n_games: int = 4) -> float:
    if results is None or results.empty:
        return 0.0

    df = results.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date", ascending=False)

    margins = []
    for _, r in df.iterrows():
        if {r["home"], r["away"]} != {home, away}:
            continue
        if r["home"] == home:
            margins.append(float(r["home_pts"]) - float(r["away_pts"]))
        else:
            margins.append(float(r["away_pts"]) - float(r["home_pts"]))
        if len(margins) >= n_games:
            break

    if not margins:
        return 0.0
    return float(np.mean(margins))


def compute_auto_upset_signal(
    home: str,
    away: str,
    home_model_prob: float,
    home_market_prob: float,
    home_odds: float,
    away_odds: float,
    form_stats: Dict[str, Dict[str, float]],
    h2h_margin_home: float,
) -> Dict[str, object]:
    if not math.isnan(home_market_prob):
        fav = home if home_market_prob >= 0.50 else away
    else:
        fav = home if home_model_prob >= 0.50 else away
    dog = away if fav == home else home

    fav_stats = form_stats.get(fav, {})
    dog_stats = form_stats.get(dog, {})

    fav_recent_margin = float(fav_stats.get("recent_margin", 0.0))
    dog_recent_margin = float(dog_stats.get("recent_margin", 0.0))
    fav_recent_wr = float(fav_stats.get("recent_win_rate", 0.5))
    dog_recent_wr = float(dog_stats.get("recent_win_rate", 0.5))

    score = 0.0
    reasons = []

    if (dog_recent_margin - fav_recent_margin) >= 6.0:
        score += 1.0
        reasons.append("dog_recent_margin")
    if (dog_recent_wr - fav_recent_wr) >= 0.18:
        score += 1.0
        reasons.append("dog_recent_winrate")

    fav_market_prob = home_market_prob if fav == home else (1.0 - home_market_prob if not math.isnan(home_market_prob) else float("nan"))
    fav_model_prob = home_model_prob if fav == home else 1.0 - home_model_prob
    fav_odds = home_odds if fav == home else away_odds

    if not math.isnan(fav_market_prob) and fav_market_prob >= 0.62 and fav_model_prob <= 0.56:
        score += 1.0
        reasons.append("fav_short_market_only")
    if not math.isnan(fav_odds) and fav_odds <= 1.55 and fav_model_prob <= 0.55:
        score += 0.5
        reasons.append("very_short_favourite")

    dog_h2h_edge = -h2h_margin_home if dog == away else h2h_margin_home
    if dog_h2h_edge >= 4.0:
        score += 1.0
        reasons.append("h2h_dog_edge")

    return {
        "favourite_team": fav,
        "underdog_team": dog,
        "auto_upset_score": round(score, 2),
        "auto_upset_reasons": "|".join(reasons),
    }


def apply_upset_probability_adjustment(
    home_prob: float,
    upset_team: str,
    final_upset_score: float,
    home: str,
    away: str,
) -> float:
    if final_upset_score <= 0 or upset_team not in {home, away}:
        return home_prob

    shift = min(UPSET_PROB_SHIFT_CAP, final_upset_score * UPSET_PROB_SHIFT_PER_POINT)
    if upset_team == home:
        home_prob += shift
    else:
        home_prob -= shift

    return min(0.95, max(0.05, home_prob))


def confidence_band(prob: float, conf: float, abs_margin: float) -> str:
    if prob >= 0.66 and conf >= 0.64 and abs_margin >= 7.0:
        return "High"
    if prob >= 0.58 and conf >= 0.57 and abs_margin >= 3.5:
        return "Medium"
    return "Low"


def dynamic_required_edge(decimal_odds: float, side_team: str, is_home: bool, is_favourite: bool) -> float:
    if decimal_odds < 1.50:
        req = 0.030
    elif decimal_odds < 1.65:
        req = 0.032
    elif decimal_odds < 1.85:
        req = 0.038
    elif decimal_odds < 2.05:
        req = 0.045
    elif decimal_odds < 2.30:
        req = 0.055
    elif decimal_odds < 2.80:
        req = 0.065
    else:
        req = 0.075

    if is_home and is_favourite and side_team in PREMIUM_HOME_TEAMS:
        req -= 0.010
    if is_home and is_favourite and side_team in ELITE_HOME_TEAMS:
        req -= 0.005
    if is_home and is_favourite and side_team in WEAK_HOME_TEAMS:
        req += 0.012

    return max(0.020, req)


def score_bet_opportunity(
    pick_prob: float,
    edge: float,
    odds: float,
    conf: float,
    team: str,
    is_home: bool,
    is_favourite: bool,
    volatility: float,
    final_upset_score: float,
    exp_margin: float,
    min_games: float,
) -> float:
    score = 0.0
    score += max(0.0, (pick_prob - 0.50) * 100.0)
    score += max(0.0, edge * 140.0)
    score += max(0.0, (conf - 0.50) * 80.0)
    score += min(10.0, max(0.0, abs(exp_margin) * 0.8))

    if is_home and is_favourite and team in PREMIUM_HOME_TEAMS:
        score += 4.0
    if is_home and is_favourite and team in ELITE_HOME_TEAMS:
        score += 2.0
    if is_home and is_favourite and team in WEAK_HOME_TEAMS:
        score -= 5.0

    if odds >= 2.60:
        score -= 5.0
    elif odds >= 2.20:
        score -= 2.5

    if volatility >= 12.0:
        score -= 6.0
    elif volatility >= 10.0:
        score -= 3.0

    if final_upset_score >= 2.0 and is_favourite:
        score -= 5.0
    if final_upset_score >= 2.0 and not is_favourite:
        score += 2.0

    if min_games < 3:
        score -= 5.0
    elif min_games < 5:
        score -= 2.0

    return score


def assign_bet_grade(
    pick_prob: float,
    edge: float,
    odds: float,
    conf: float,
    team: str,
    is_home: bool,
    is_favourite: bool,
    volatility: float,
    final_upset_score: float,
    exp_margin: float,
    min_games: float,
) -> str:
    req_edge = dynamic_required_edge(odds, team, is_home, is_favourite)

    if edge < max(0.0, req_edge - 0.015) or pick_prob < 0.52:
        return "No Bet"

    score = score_bet_opportunity(
        pick_prob=pick_prob,
        edge=edge,
        odds=odds,
        conf=conf,
        team=team,
        is_home=is_home,
        is_favourite=is_favourite,
        volatility=volatility,
        final_upset_score=final_upset_score,
        exp_margin=exp_margin,
        min_games=min_games,
    )

    if edge >= req_edge + 0.020 and pick_prob >= 0.60 and conf >= 0.60 and score >= 22.0:
        return "Strong Bet"

    if edge >= req_edge and pick_prob >= 0.56 and conf >= 0.56 and score >= 14.0:
        return "Small Bet"

    if edge >= max(0.0, req_edge - 0.010) and pick_prob >= 0.53 and score >= 8.0:
        return "Lean"

    return "No Bet"


def stake_from_grade(grade: str, odds: float) -> float:
    if grade == "Strong Bet":
        base = 40.0 if odds < 2.0 else 30.0
    elif grade == "Small Bet":
        base = 20.0 if odds < 2.0 else 10.0
    else:
        return 0.0

    floor = MIN_BET_SHORT if odds < 2.0 else MIN_BET_DOG
    return round(min(MAX_SINGLE_BET, max(floor, base)), 2)


def apply_round_exposure_cap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "stake_dollars" not in df.columns:
        return df

    work = df.copy()
    work["stake_dollars"] = pd.to_numeric(work["stake_dollars"], errors="coerce").fillna(0.0)

    bet_df = work[work["stake_dollars"] > 0].copy()
    if bet_df.empty:
        return work

    grade_rank = {"Strong Bet": 3, "Small Bet": 2, "Lean": 1, "No Bet": 0}
    bet_df["grade_rank"] = bet_df["bet_grade"].map(grade_rank).fillna(0).astype(int)

    bet_df = bet_df.sort_values(
        ["grade_rank", "edge", "win_probability", "confidence"],
        ascending=[False, False, False, False]
    ).reset_index()

    running_exposure = 0.0
    keep_original_idx = []

    for _, row in bet_df.iterrows():
        stake_amt = float(row["stake_dollars"])
        if running_exposure + stake_amt > MAX_ROUND_EXPOSURE:
            continue
        keep_original_idx.append(int(row["index"]))
        running_exposure += stake_amt

    excluded_mask = (work["stake_dollars"] > 0) & (~work.index.isin(keep_original_idx))
    work.loc[excluded_mask, "bet_grade"] = "Lean"
    work.loc[excluded_mask, "stake_dollars"] = 0.0
    work.loc[excluded_mask, "stake_units"] = 0.0
    work.loc[excluded_mask, "stake"] = 0.0
    work.loc[excluded_mask, "pick"] = ""
    work.loc[excluded_mask, "recommended_bet"] = "No Bet"

    print(f"[predict] exposure cap applied: ${running_exposure:.2f} / ${MAX_ROUND_EXPOSURE:.2f}")
    return work


def build_predictions() -> pd.DataFrame:
    fixtures: List[Match] = []

    if MODE == "AUTO":
        fixtures = fixtures_from_odds_csv("odds.csv")
        if not fixtures:
            fixtures = fetch_upcoming_fixtures(days_ahead=21)
        if not fixtures:
            raise SystemExit("[stop] No upcoming fixtures found from odds.csv or the fixture feed.")

        fixtures = _dedupe_fixtures(fixtures)
        fixtures = _filter_current_round_fixtures(fixtures)

        print(f"[debug] fixtures after dedupe/current-round filter: {len(fixtures)}")
        for m in fixtures:
            print(f"[debug] current-round fixture: {m.date} {m.home} v {m.away}")
    else:
        raise SystemExit("[stop] MODE is not AUTO.")

    teams = sorted(list(TEAM_REGION.keys()))
    run_id = make_run_id()
    run_utc = utc_now_str()

    saved_model = load_saved_ratings()
    results = fetch_completed_results()
    results = results.drop_duplicates(subset=["date", "home", "away"], keep="last").reset_index(drop=True)
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(subset=["date"]).copy()
    results = results[results["date"].dt.year >= 2025].copy()
    results["date"] = results["date"].dt.strftime("%Y-%m-%d")

    print(f"[debug] combined results rows={len(results)} (2025+ only, deduped)")

    fresh_model = fit_attack_defence(results, teams)
    if fresh_model:
        ad_model = fresh_model
        save_ratings(ad_model)
    elif saved_model:
        ad_model = saved_model
    else:
        ad_model = None

    adj = load_adjustments()
    odds = load_odds()
    manual_upsets = load_manual_upset_flags()
    form_stats = build_recent_form_stats(results, teams)
    team_volatility = build_team_volatility(results, teams)
    team_home_edge = build_team_home_ground_edges(results, teams)
    season_record = build_current_season_record(results, 2026)

    if ad_model is not None:
        ad_model["team_home_edge"] = team_home_edge

    rows = []

    for m in fixtures:
        key = (m.date, m.home, m.away)
        o = odds.get(key, {})
        home_odds = o.get("home_odds", float("nan"))
        away_odds = o.get("away_odds", float("nan"))

        if ad_model:
            model_home_prob, exp_margin, exp_total, conf = simulate_match_ad(ad_model, m.home, m.away, m.venue, adj)
            rating_mode = "ATTACK_DEFENCE"
        else:
            model_home_prob, exp_margin, exp_total, conf = 0.50, 0.0, 40.0, 0.45
            rating_mode = "FALLBACK"

        market_home_prob, market_away_prob = fair_probs_from_odds(home_odds, away_odds)

        if math.isnan(market_home_prob):
            blended_home_prob = model_home_prob
        else:
            fav_odds = min(home_odds, away_odds)
            if fav_odds <= 1.60:
                market_w = 0.62
            elif fav_odds <= 1.85:
                market_w = 0.59
            elif fav_odds <= 2.10:
                market_w = 0.56
            else:
                market_w = 0.53

            model_w = 1.0 - market_w
            blended_home_prob = (model_w * model_home_prob) + (market_w * market_home_prob)

        blended_home_prob = apply_outright_strength_adjustment(blended_home_prob, m.home, m.away)
        blended_home_prob = apply_early_season_matchup_moderation(
            blended_home_prob, m.home, m.away, season_record
        )

        h2h_margin_home = recent_h2h_margin(results, m.home, m.away)
        auto_upset = compute_auto_upset_signal(
            home=m.home,
            away=m.away,
            home_model_prob=model_home_prob,
            home_market_prob=market_home_prob,
            home_odds=home_odds,
            away_odds=away_odds,
            form_stats=form_stats,
            h2h_margin_home=h2h_margin_home,
        )

        manual_upset = manual_upsets.get(key, {"upset_team": "", "manual_upset_score": 0.0, "notes": ""})
        manual_upset_team = manual_upset.get("upset_team", "")
        manual_upset_score = float(manual_upset.get("manual_upset_score", 0.0))

        upset_team = manual_upset_team if manual_upset_team in {m.home, m.away} else auto_upset["underdog_team"]
        final_upset_score = float(auto_upset["auto_upset_score"]) + manual_upset_score
        final_upset_flag = 1 if final_upset_score >= UPSET_FLAG_THRESHOLD else 0

        final_home_prob = apply_upset_probability_adjustment(
            home_prob=blended_home_prob,
            upset_team=upset_team,
            final_upset_score=final_upset_score,
            home=m.home,
            away=m.away,
        )
        final_home_prob = compress_prob(final_home_prob)

        home_win_prob = final_home_prob
        away_win_prob = 1.0 - final_home_prob

        if home_win_prob >= away_win_prob:
            predicted_winner = m.home
            win_probability = home_win_prob
            pick_side = "HOME"
            pick_odds = home_odds
            pick_edge = value_edge(home_win_prob, home_odds)
            is_home = True
            expected_margin_for_pick = exp_margin
        else:
            predicted_winner = m.away
            win_probability = away_win_prob
            pick_side = "AWAY"
            pick_odds = away_odds
            pick_edge = value_edge(away_win_prob, away_odds)
            is_home = False
            expected_margin_for_pick = -exp_margin

        market_favourite = ""
        if not math.isnan(home_odds) and not math.isnan(away_odds):
            market_favourite = m.home if home_odds <= away_odds else m.away

        is_favourite = predicted_winner == market_favourite if market_favourite else False
        side_volatility = float(team_volatility.get(predicted_winner, 0.0))

        home_games = float(form_stats.get(m.home, {}).get("games", 0.0))
        away_games = float(form_stats.get(m.away, {}).get("games", 0.0))
        min_games = min(home_games, away_games)

        band = confidence_band(win_probability, conf, abs(expected_margin_for_pick))

        if math.isnan(pick_edge):
            bet_grade = "No Bet"
            stake_dollars = 0.0
            required_edge = np.nan
        else:
            required_edge = dynamic_required_edge(
                decimal_odds=pick_odds,
                side_team=predicted_winner,
                is_home=is_home,
                is_favourite=is_favourite,
            )
            bet_grade = assign_bet_grade(
                pick_prob=win_probability,
                edge=pick_edge,
                odds=pick_odds,
                conf=conf,
                team=predicted_winner,
                is_home=is_home,
                is_favourite=is_favourite,
                volatility=side_volatility,
                final_upset_score=final_upset_score,
                exp_margin=expected_margin_for_pick,
                min_games=min_games,
            )
            stake_dollars = stake_from_grade(bet_grade, pick_odds)

        if bet_grade == "Lean":
            stake_dollars = 0.0

        pick = pick_side if stake_dollars > 0 else ""
        stake_units = round(stake_dollars / UNIT_SIZE, 2) if UNIT_SIZE > 0 else 0.0
        recommended_bet = f"${stake_dollars:.2f} {predicted_winner}" if stake_dollars > 0 else "No Bet"
        value_flag = f"{bet_grade} {predicted_winner}" if bet_grade != "No Bet" else ""
        fragile_favourite = int(is_favourite and final_upset_score >= UPSET_FLAG_THRESHOLD and pick_odds < 1.70)

        rows.append({
            "run_id": run_id,
            "run_utc": run_utc,
            "mode": MODE,
            "rating_mode": rating_mode,
            "date": m.date,
            "kickoff_local": m.kickoff_local,
            "venue": m.venue,
            "home": m.home,
            "away": m.away,

            "predicted_winner": predicted_winner,
            "win_probability": round(win_probability, 3),
            "confidence_band": band,
            "bet_grade": bet_grade,

            "model_home_win_prob": round(model_home_prob, 3),
            "market_home_win_prob": round(market_home_prob, 3) if not math.isnan(market_home_prob) else np.nan,
            "final_home_win_prob": round(final_home_prob, 3),
            "final_away_win_prob": round(1.0 - final_home_prob, 3),

            "exp_margin_home": round(exp_margin, 1),
            "exp_total": round(exp_total, 1),
            "confidence": round(conf, 2),

            "home_odds": home_odds,
            "away_odds": away_odds,
            "predicted_winner_odds": pick_odds,
            "edge": round(pick_edge, 3) if not math.isnan(pick_edge) else np.nan,

            "favourite_team": auto_upset["favourite_team"],
            "underdog_team": auto_upset["underdog_team"],
            "auto_upset_score": round(float(auto_upset["auto_upset_score"]), 2),
            "auto_upset_reasons": auto_upset["auto_upset_reasons"],
            "manual_upset_team": manual_upset_team,
            "manual_upset_score": round(manual_upset_score, 2),
            "manual_upset_notes": manual_upset.get("notes", ""),
            "upset_team": upset_team,
            "final_upset_score": round(final_upset_score, 2),
            "upset_flag": final_upset_flag,

            # compatibility fields for old downstream scripts
            "fragile_favourite": fragile_favourite,
            "required_edge": round(required_edge, 3) if pd.notna(required_edge) else np.nan,
            "value_flag": value_flag,
            "pick": pick,
            "stake": float(stake_units),
            "stake_units": float(stake_units),
            "stake_dollars": float(stake_dollars),
            "recommended_bet": recommended_bet,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        })

    df = pd.DataFrame(rows).sort_values(["date", "kickoff_local"]).reset_index(drop=True)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    round_start = df["date"].min()
    round_end = round_start + pd.Timedelta(days=3)
    df = df[(df["date"] >= round_start) & (df["date"] <= round_end)].copy()

    df = apply_round_exposure_cap(df)

    real_bets = df[df["stake_dollars"] > 0].copy()
    if len(real_bets) > TARGET_MAX_BETS:
        real_bets = real_bets.sort_values(
            ["bet_grade", "edge", "win_probability", "confidence"],
            ascending=[False, False, False, False]
        ).reset_index()

        keep_idx = set(real_bets.head(TARGET_MAX_BETS)["index"].tolist())
        excess_mask = (df["stake_dollars"] > 0) & (~df.index.isin(keep_idx))
        df.loc[excess_mask, "bet_grade"] = "Lean"
        df.loc[excess_mask, "stake"] = 0.0
        df.loc[excess_mask, "stake_units"] = 0.0
        df.loc[excess_mask, "stake_dollars"] = 0.0
        df.loc[excess_mask, "pick"] = ""
        df.loc[excess_mask, "recommended_bet"] = "No Bet"

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    round_label = f"Round window {round_start.strftime('%Y-%m-%d')} to {round_end.strftime('%Y-%m-%d')}"
    bet_count = int((pd.to_numeric(df.get("stake_dollars", 0), errors="coerce").fillna(0) > 0).sum())
    exposure = float(pd.to_numeric(df.get("stake_dollars", 0), errors="coerce").fillna(0).sum())
    avg_edge_series = pd.to_numeric(df.get("edge", np.nan), errors="coerce").dropna()
    avg_edge = float(avg_edge_series.mean()) if not avg_edge_series.empty else 0.0

    print(f"[predict] {round_label}")
    print(f"[predict] current round fixtures={len(df)} bets={bet_count} exposure=${exposure:.2f} avg_edge={avg_edge:.3f}")

    return df


if __name__ == "__main__":
    df = build_predictions()
    df.to_csv("predictions.csv", index=False)

    if "stake_dollars" in df.columns:
        bet_count = int((pd.to_numeric(df["stake_dollars"], errors="coerce").fillna(0.0) > 0).sum())
        print(f"[predict] rows={len(df)} bets={bet_count}")
    else:
        print(f"[predict] rows={len(df)}")
