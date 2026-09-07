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
# TIPPING / AVAILABILITY SETTINGS
# ----------------------------
# The predictor is now tipping-first. Legacy staking columns are still written
# as zero/blank values for compatibility with older archive/report scripts.

# Rest is calculated automatically from completed results.
DEFAULT_REST_DAYS = 7
REST_PROB_PER_POINT = 0.008
REST_PROB_SHIFT_CAP = 0.03

# Player/team-list layer. These files are optional. If they do not exist the
# model runs normally, so GitHub Actions will not fail.
PLAYER_RATINGS_PATH = "player_ratings.csv"
TEAM_CHANGES_PATH = "team_changes.csv"
AVAILABILITY_PROB_PER_POINT = 0.012
AVAILABILITY_PROB_SHIFT_CAP = 0.08
AVAILABILITY_POINTS_CAP = 8.0

# Pre-season futures prices are useful early in a season but stale by finals.
# Leave False for the finals. It can be switched back on for a new season.
USE_OUTRIGHT_PRIOR = False

# ----------------------------
# RUN MODE
# ----------------------------
MODE = "AUTO"


# ----------------------------
# Results sources for ratings
# ----------------------------
RESULTS_URLS = {}
RESULTS_CACHE_PATH = "results_cache.csv"
MANUAL_RESULTS_2026_PATH = "results_2026.csv"


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

ADJUSTMENTS_PATH = "adjustments.csv"

# Market anchoring
MARKET_BLEND_MIN = 0.40
MARKET_BLEND_MAX = 0.65
MARKET_STRONG_FAV_THRESHOLD = 1.60
MARKET_MED_FAV_THRESHOLD = 1.85
MARKET_CLOSE_THRESHOLD = 2.10
MARKET_ANCHOR_MAX_SHIFT = 0.12
MARKET_DISAGREEMENT_NO_BET = 0.10

# Ladder / form / injury weighting
LADDER_ADJ_CAP = 3.0
FORM_ADJ_CAP = 2.0
HOME_AWAY_FORM_CAP = 1.8
INJURY_IMPACT_CAP = 5.0

# Bet guardrails
AWAY_DOG_EXTRA_EDGE = 0.030
AWAY_TEAM_EXTRA_EDGE = 0.015
VOLATILE_TEAM_EXTRA_EDGE = 0.015
INJURED_TEAM_EXTRA_EDGE = 0.020

AGGRESSIVE_HOME_TEAMS = {"Bulldogs", "Storm", "Warriors"}
ELITE_BOUNCE_TEAMS = {"Storm", "Panthers", "Roosters"}


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
    "Manly": "Sea Eagles",
    "Canberra Raiders": "Raiders",
    "South Sydney Rabbitohs": "Rabbitohs",
    "Dolphins": "Dolphins",
    "The Dolphins": "Dolphins",
    "Wests Tigers": "Wests Tigers",
    "Tigers": "Wests Tigers",
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
    "West Tigers": "Wests Tigers",
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


def build_ladder_stats(results: pd.DataFrame, teams: List[str], season_year: int = 2026) -> Dict[str, Dict[str, float]]:
    out = {
        t: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "points_for": 0.0,
            "points_against": 0.0,
            "diff": 0.0,
            "win_pct": 0.5,
            "avg_margin": 0.0,
            "comp_points": 0.0,
        }
        for t in teams
    }

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
            out[home]["games"] += 1
            out[home]["points_for"] += hp
            out[home]["points_against"] += ap
            out[home]["diff"] += (hp - ap)
            if hp > ap:
                out[home]["wins"] += 1
                out[home]["comp_points"] += 2
            elif hp < ap:
                out[home]["losses"] += 1
            else:
                out[home]["draws"] += 1
                out[home]["comp_points"] += 1

        if away in out:
            out[away]["games"] += 1
            out[away]["points_for"] += ap
            out[away]["points_against"] += hp
            out[away]["diff"] += (ap - hp)
            if ap > hp:
                out[away]["wins"] += 1
                out[away]["comp_points"] += 2
            elif ap < hp:
                out[away]["losses"] += 1
            else:
                out[away]["draws"] += 1
                out[away]["comp_points"] += 1

    for t in teams:
        g = out[t]["games"]
        if g > 0:
            out[t]["win_pct"] = out[t]["wins"] / g
            out[t]["avg_margin"] = out[t]["diff"] / g

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


def ladder_strength_adjustment(home: str, away: str, ladder: Dict[str, Dict[str, float]]) -> float:
    h = ladder.get(home, {})
    a = ladder.get(away, {})

    h_margin = float(h.get("avg_margin", 0.0))
    a_margin = float(a.get("avg_margin", 0.0))
    h_win = float(h.get("win_pct", 0.5))
    a_win = float(a.get("win_pct", 0.5))
    h_games = float(h.get("games", 0.0))
    a_games = float(a.get("games", 0.0))

    games_scale = min(1.0, min(h_games, a_games) / 6.0)
    margin_diff = (h_margin - a_margin) * 0.18
    win_diff = (h_win - a_win) * 4.0

    adj = (margin_diff + win_diff) * games_scale
    return max(-LADDER_ADJ_CAP, min(LADDER_ADJ_CAP, adj))


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

        fixtures.append(
            Match(
                date=date,
                kickoff_local="",
                home=home,
                away=away,
                venue="",
            )
        )

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
    round_end = round_start + pd.Timedelta(days=4)  # includes Monday

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

        return pd.DataFrame(
            {
                "date": date_series.dt.strftime("%Y-%m-%d"),
                "home": df["Home"].astype(str).apply(norm_team),
                "away": df["Away"].astype(str).apply(norm_team),
                "home_pts": pd.to_numeric(df["HomeScore"], errors="coerce"),
                "away_pts": pd.to_numeric(df["AwayScore"], errors="coerce"),
            }
        )

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

        return pd.DataFrame(
            {
                "date": date_series.dt.strftime("%Y-%m-%d"),
                "home": df["Home Team"].astype(str).apply(norm_team),
                "away": df["Away Team"].astype(str).apply(norm_team),
                "home_pts": scores.apply(lambda t: t[0]),
                "away_pts": scores.apply(lambda t: t[1]),
            }
        )

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
        return pd.DataFrame(
            {
                "date": pd.to_datetime(df[colmap["date"]], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d"),
                "home": df[colmap["home team"]].astype(str).apply(norm_team),
                "away": df[colmap["away team"]].astype(str).apply(norm_team),
                "home_pts": scores.apply(lambda t: t[0]),
                "away_pts": scores.apply(lambda t: t[1]),
            }
        )

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


def load_manual_results_2026(path: str = MANUAL_RESULTS_2026_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts", "season_year"])

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Could not load {path}: {e}")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts", "season_year"])

    needed = {"date", "home", "away", "home_pts", "away_pts"}
    if not needed.issubset(set(df.columns)):
        print(f"[warn] {path} missing required columns: {sorted(needed)}")
        return pd.DataFrame(columns=["date", "home", "away", "home_pts", "away_pts", "season_year"])

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["home"] = df["home"].astype(str).apply(norm_team)
    df["away"] = df["away"].astype(str).apply(norm_team)
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")
    df = df.dropna(subset=["date", "home", "away", "home_pts", "away_pts"]).copy()
    df["season_year"] = 2026

    print(f"[info] Loaded manual {path} ({len(df)} rows)")
    return df


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

    manual_2026_df = load_manual_results_2026()

    merged = pd.concat([cache, web_df, manual_2026_df], ignore_index=True)
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

        if len(home) < 2 or len(away) < 2:
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


def load_adjustments(path: str = ADJUSTMENTS_PATH) -> Dict[str, Dict[str, float]]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    out = {}
    for _, r in df.iterrows():
        team = norm_team(r.get("team", ""))
        if not team:
            continue

        atk_delta = pd.to_numeric(r.get("atk_delta_pts", 0.0), errors="coerce")
        def_delta = pd.to_numeric(r.get("def_delta_pts", 0.0), errors="coerce")
        spine_out = pd.to_numeric(r.get("spine_out_count", 0.0), errors="coerce")
        key_out = pd.to_numeric(r.get("key_out_count", 0.0), errors="coerce")
        market_concern = pd.to_numeric(r.get("market_concern", 0.0), errors="coerce")

        out[team] = {
            "atk": float(atk_delta) if pd.notna(atk_delta) else 0.0,
            "def": float(def_delta) if pd.notna(def_delta) else 0.0,
            "spine_out_count": float(spine_out) if pd.notna(spine_out) else 0.0,
            "key_out_count": float(key_out) if pd.notna(key_out) else 0.0,
            "market_concern": float(market_concern) if pd.notna(market_concern) else 0.0,
            "notes": str(r.get("notes", "")).strip(),
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


def build_home_away_form_stats(results: pd.DataFrame, teams: List[str], recent_n: int = 4) -> Dict[str, Dict[str, float]]:
    out = {
        t: {
            "home_margin": 0.0,
            "home_win_rate": 0.5,
            "home_games": 0.0,
            "away_margin": 0.0,
            "away_win_rate": 0.5,
            "away_games": 0.0,
        }
        for t in teams
    }

    if results is None or results.empty:
        return out

    df = results.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    for team in teams:
        hdf = df[df["home"] == team].sort_values("date", ascending=False).head(recent_n).copy()
        if not hdf.empty:
            hdf["yr_weight"] = hdf["date"].dt.year.map(lambda y: YEAR_WEIGHTS.get(int(y), 0.0)).fillna(0.0)
            hdf = hdf[hdf["yr_weight"] > 0].copy()
            if not hdf.empty:
                h_weights = np.linspace(1.20, 0.90, len(hdf)) * hdf["yr_weight"].values
                margins = (hdf["home_pts"] - hdf["away_pts"]).astype(float).values
                wins = (hdf["home_pts"] > hdf["away_pts"]).astype(float).values
                out[team]["home_margin"] = float(np.average(margins, weights=h_weights))
                out[team]["home_win_rate"] = float(np.average(wins, weights=h_weights))
                out[team]["home_games"] = float(len(hdf))

        adf = df[df["away"] == team].sort_values("date", ascending=False).head(recent_n).copy()
        if not adf.empty:
            adf["yr_weight"] = adf["date"].dt.year.map(lambda y: YEAR_WEIGHTS.get(int(y), 0.0)).fillna(0.0)
            adf = adf[adf["yr_weight"] > 0].copy()
            if not adf.empty:
                a_weights = np.linspace(1.20, 0.90, len(adf)) * adf["yr_weight"].values
                margins = (adf["away_pts"] - adf["home_pts"]).astype(float).values
                wins = (adf["away_pts"] > adf["home_pts"]).astype(float).values
                out[team]["away_margin"] = float(np.average(margins, weights=a_weights))
                out[team]["away_win_rate"] = float(np.average(wins, weights=a_weights))
                out[team]["away_games"] = float(len(adf))

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


def home_away_form_adjustment(home: str, away: str, ha_form: Dict[str, Dict[str, float]]) -> float:
    hs = ha_form.get(home, {})
    as_ = ha_form.get(away, {})

    h_margin = float(hs.get("home_margin", 0.0))
    a_margin = float(as_.get("away_margin", 0.0))
    h_win = float(hs.get("home_win_rate", 0.5))
    a_win = float(as_.get("away_win_rate", 0.5))
    h_games = float(hs.get("home_games", 0.0))
    a_games = float(as_.get("away_games", 0.0))

    game_scale = min(1.0, min(h_games, a_games) / 4.0)
    adj = ((h_margin - a_margin) * 0.22) + ((h_win - a_win) * 3.2)
    adj *= game_scale

    return max(-HOME_AWAY_FORM_CAP, min(HOME_AWAY_FORM_CAP, adj))


def form_strength_adjustment(home: str, away: str, form_stats: Dict[str, Dict[str, float]]) -> float:
    hs = form_stats.get(home, {})
    as_ = form_stats.get(away, {})

    h_margin = float(hs.get("recent_margin", 0.0))
    a_margin = float(as_.get("recent_margin", 0.0))
    h_win = float(hs.get("recent_win_rate", 0.5))
    a_win = float(as_.get("recent_win_rate", 0.5))
    h_games = float(hs.get("games", 0.0))
    a_games = float(as_.get("games", 0.0))

    game_scale = min(1.0, min(h_games, a_games) / 5.0)
    adj = ((h_margin - a_margin) * 0.17) + ((h_win - a_win) * 2.8)
    adj *= game_scale

    return max(-FORM_ADJ_CAP, min(FORM_ADJ_CAP, adj))


def team_injury_impact(team: str, adj: Dict[str, Dict[str, float]]) -> float:
    a = adj.get(team, {})
    spine_out = float(a.get("spine_out_count", 0.0))
    key_out = float(a.get("key_out_count", 0.0))
    market_concern = float(a.get("market_concern", 0.0))
    atk_delta = float(a.get("atk", 0.0))
    def_delta = float(a.get("def", 0.0))

    impact = 0.0
    impact += spine_out * 1.9
    impact += key_out * 0.9
    impact += market_concern * 1.0
    impact += max(0.0, -atk_delta) * 0.40
    impact += max(0.0, -def_delta) * 0.30

    return max(0.0, min(INJURY_IMPACT_CAP, impact))


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
    ladder_stats = model.get("ladder_stats", {})
    form_stats = model.get("form_stats", {})
    home_away_form = model.get("home_away_form", {})

    home_pts = mu + ha + atk.get(home, 0.0) - dfn.get(away, 0.0)
    away_pts = mu + atk.get(away, 0.0) - dfn.get(home, 0.0)

    ladder_adj = ladder_strength_adjustment(home, away, ladder_stats)
    form_adj = form_strength_adjustment(home, away, form_stats)
    ha_form_adj = home_away_form_adjustment(home, away, home_away_form)

    home_pts += ladder_adj + form_adj + ha_form_adj
    away_pts -= ladder_adj + form_adj + ha_form_adj

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
    conf = min(0.82, 0.50 + abs(win_prob - 0.5) * 0.78)

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
        factor = 0.95
    elif dist < 0.12:
        factor = 0.93
    else:
        factor = 0.90

    return min(0.95, max(0.05, 0.5 + (p - 0.5) * factor))


def market_weight_from_prices(home_odds: float, away_odds: float) -> float:
    if math.isnan(home_odds) or math.isnan(away_odds):
        return MARKET_BLEND_MIN

    fav_odds = min(home_odds, away_odds)

    if fav_odds <= MARKET_STRONG_FAV_THRESHOLD:
        return MARKET_BLEND_MAX
    if fav_odds <= MARKET_MED_FAV_THRESHOLD:
        return 0.63
    if fav_odds <= MARKET_CLOSE_THRESHOLD:
        return 0.58
    return MARKET_BLEND_MIN


def anchor_to_market(model_prob: float, market_prob: float, weight: float) -> float:
    if math.isnan(market_prob):
        return model_prob

    blended = ((1.0 - weight) * model_prob) + (weight * market_prob)
    diff = blended - market_prob
    if abs(diff) > MARKET_ANCHOR_MAX_SHIFT:
        blended = market_prob + math.copysign(MARKET_ANCHOR_MAX_SHIFT, diff)

    return min(0.94, max(0.06, blended))


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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        n = float(value)
        if math.isnan(n):
            return default
        return n
    except Exception:
        return default


def norm_player_name(name: object) -> str:
    return re.sub(r"\\s+", " ", str(name or "").strip()).casefold()


def _normalise_position(position: object) -> str:
    p = str(position or "").strip().casefold().replace("_", " ").replace("-", " ")
    aliases = {
        "1": "fullback", "fb": "fullback", "full back": "fullback",
        "6": "five eighth", "five eighth": "five eighth", "fiveeighth": "five eighth", "fe": "five eighth",
        "7": "halfback", "hb": "halfback", "half back": "halfback",
        "9": "hooker", "hk": "hooker",
        "8": "prop", "10": "prop", "front row": "prop", "front rower": "prop",
        "11": "second row", "12": "second row", "second rower": "second row",
        "13": "lock", "lock forward": "lock",
        "2": "wing", "5": "wing", "winger": "wing",
        "3": "centre", "4": "centre", "center": "centre",
        "14": "bench", "15": "bench", "16": "bench", "17": "bench", "interchange": "bench", "reserve": "bench",
    }
    return aliases.get(p, p)


DEFAULT_POSITION_IMPACT = {
    "halfback": 5.0,
    "five eighth": 4.2,
    "fullback": 4.0,
    "hooker": 4.0,
    "prop": 2.5,
    "lock": 2.5,
    "second row": 2.0,
    "centre": 1.6,
    "wing": 1.2,
    "bench": 1.0,
}
SPINE_POSITIONS = {"fullback", "five eighth", "halfback", "hooker"}
OUT_STATUSES = {"out", "late out", "injured", "injury", "suspended", "suspension", "origin", "origin out", "unavailable"}
RETURN_STATUSES = {"in", "return", "returns", "returning", "back", "available"}
DOUBTFUL_STATUSES = {"doubtful", "questionable", "test", "50/50"}


def load_player_ratings(path: str = PLAYER_RATINGS_PATH) -> Dict[Tuple[str, str], Dict[str, object]]:
    """Load optional player impact ratings.

    Preferred columns:
      team, player, impact_points, position

    impact_points is the player's approximate value in scoreboard points relative
    to a replacement-level player. Missing files or columns are handled safely.
    """
    if not os.path.exists(path):
        print(f"[availability] {path} not found - using position/default impacts only")
        return {}

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Could not load {path}: {e}")
        return {}

    out: Dict[Tuple[str, str], Dict[str, object]] = {}
    for _, r in df.iterrows():
        team = norm_team(r.get("team", ""))
        player = str(r.get("player", "")).strip()
        if not team or not player:
            continue

        position = _normalise_position(r.get("position", ""))
        impact = pd.to_numeric(r.get("impact_points", np.nan), errors="coerce")
        if pd.isna(impact):
            # Support a generic rating column if one already exists. Ratings above
            # 10 are treated as 0-100 style ratings and converted conservatively.
            rating = pd.to_numeric(r.get("rating", np.nan), errors="coerce")
            if pd.notna(rating):
                rating = float(rating)
                impact = max(0.5, min(6.5, (rating - 50.0) / 8.0)) if rating > 10 else rating
            else:
                impact = DEFAULT_POSITION_IMPACT.get(position, 1.5)

        out[(team, norm_player_name(player))] = {
            "player": player,
            "team": team,
            "position": position,
            "impact_points": max(0.0, min(6.5, float(impact))),
        }

    print(f"[availability] Loaded {len(out)} player ratings from {path}")
    return out


def load_team_changes(path: str = TEAM_CHANGES_PATH) -> pd.DataFrame:
    """Load optional team changes for upcoming matches.

    Recommended columns:
      date,team,player,status,replacement_player,position,impact_points,
      replacement_impact_points,notes

    `date` should normally be the match date. A team-list publication date also
    works because the most recent row up to seven days before kickoff is accepted.
    """
    cols = [
        "date", "team", "player", "status", "replacement_player", "position",
        "impact_points", "replacement_impact_points", "notes",
    ]
    if not os.path.exists(path):
        print(f"[availability] {path} not found - no player-change adjustment applied")
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Could not load {path}: {e}")
        return pd.DataFrame(columns=cols)

    for col in cols:
        if col not in df.columns:
            df[col] = ""

    df = df[cols].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["team"] = df["team"].astype(str).apply(norm_team)
    df["player"] = df["player"].astype(str).str.strip()
    df["status"] = df["status"].astype(str).str.strip().str.casefold()
    df["replacement_player"] = df["replacement_player"].astype(str).str.strip()
    df["position"] = df["position"].astype(str).apply(_normalise_position)
    df["impact_points"] = pd.to_numeric(df["impact_points"], errors="coerce")
    df["replacement_impact_points"] = pd.to_numeric(df["replacement_impact_points"], errors="coerce")
    df = df[(df["team"] != "") & (df["player"] != "")].copy()

    print(f"[availability] Loaded {len(df)} team-change rows from {path}")
    return df


def _player_impact(
    team: str,
    player: object,
    position: object,
    ratings: Dict[Tuple[str, str], Dict[str, object]],
    explicit_impact: object = np.nan,
) -> float:
    explicit = pd.to_numeric(explicit_impact, errors="coerce")
    if pd.notna(explicit):
        return max(0.0, min(6.5, float(explicit)))

    rec = ratings.get((team, norm_player_name(player)))
    if rec:
        return max(0.0, min(6.5, _safe_float(rec.get("impact_points"), 1.5)))

    return DEFAULT_POSITION_IMPACT.get(_normalise_position(position), 1.5)


def team_availability_adjustment(
    team: str,
    fixture_date: str,
    team_changes: pd.DataFrame,
    ratings: Dict[Tuple[str, str], Dict[str, object]],
) -> Dict[str, object]:
    """Return a team-strength adjustment in scoreboard points.

    Negative = weakened team. Positive = strengthened team.
    Missing player value is offset by the replacement player's value where known.
    """
    empty = {
        "points": 0.0,
        "missing_impact": 0.0,
        "returning_impact": 0.0,
        "change_count": 0,
        "spine_out": 0,
        "key_out": 0,
        "summary": "",
    }
    if team_changes is None or team_changes.empty:
        return empty

    fd = pd.to_datetime(fixture_date, errors="coerce")
    if pd.isna(fd):
        return empty

    tdf = team_changes[team_changes["team"] == team].copy()
    if tdf.empty:
        return empty

    dated = tdf.dropna(subset=["date"]).copy()
    if not dated.empty:
        exact = dated[dated["date"].dt.normalize() == fd.normalize()].copy()
        if not exact.empty:
            tdf = exact
        else:
            recent = dated[(dated["date"] <= fd) & (dated["date"] >= fd - pd.Timedelta(days=7))].copy()
            if not recent.empty:
                latest_date = recent["date"].max()
                tdf = recent[recent["date"] == latest_date].copy()
            else:
                # Dated rows exist but none are relevant to this match.
                tdf = tdf[tdf["date"].isna()].copy()

    if tdf.empty:
        return empty

    points = 0.0
    missing_impact = 0.0
    returning_impact = 0.0
    spine_out = 0
    key_out = 0
    summaries = []

    for _, r in tdf.iterrows():
        status = str(r.get("status", "")).strip().casefold()
        player = str(r.get("player", "")).strip()
        position = _normalise_position(r.get("position", ""))
        player_impact = _player_impact(team, player, position, ratings, r.get("impact_points", np.nan))

        replacement = str(r.get("replacement_player", "")).strip()
        replacement_impact = pd.to_numeric(r.get("replacement_impact_points", np.nan), errors="coerce")
        if pd.isna(replacement_impact) and replacement:
            replacement_impact = _player_impact(team, replacement, position, ratings)
        replacement_impact = float(replacement_impact) if pd.notna(replacement_impact) else 0.0

        net = max(0.0, player_impact - replacement_impact)

        if status in OUT_STATUSES:
            points -= net
            missing_impact += net
            if position in SPINE_POSITIONS:
                spine_out += 1
            if player_impact >= 3.0:
                key_out += 1
            summaries.append(f"OUT {player} (-{net:.1f})")
        elif status in RETURN_STATUSES:
            points += net
            returning_impact += net
            summaries.append(f"IN {player} (+{net:.1f})")
        elif status in DOUBTFUL_STATUSES:
            points -= net * 0.50
            missing_impact += net * 0.50
            if position in SPINE_POSITIONS:
                spine_out += 1
            summaries.append(f"TEST {player} (-{net * 0.50:.1f})")
        else:
            # Unknown/non-impact statuses are retained in the source file but do
            # not alter the model. This avoids accidental penalties from typos.
            continue

    # Combination penalties. Losing multiple spine players or several key players
    # is more damaging than treating each change as fully independent.
    if spine_out >= 2:
        combo_penalty = min(1.5, 0.75 * (spine_out - 1))
        points -= combo_penalty
        missing_impact += combo_penalty
        summaries.append(f"spine disruption (-{combo_penalty:.1f})")

    if key_out >= 3:
        pack_penalty = 0.5
        points -= pack_penalty
        missing_impact += pack_penalty
        summaries.append("multiple key outs (-0.5)")

    points = max(-AVAILABILITY_POINTS_CAP, min(AVAILABILITY_POINTS_CAP, points))

    return {
        "points": round(points, 2),
        "missing_impact": round(missing_impact, 2),
        "returning_impact": round(returning_impact, 2),
        "change_count": int(len(tdf)),
        "spine_out": int(spine_out),
        "key_out": int(key_out),
        "summary": " | ".join(summaries),
    }


def rest_days_for_team(team: str, fixture_date: str, results: pd.DataFrame) -> int:
    """Calculate rest days automatically from the team's most recent match."""
    if results is None or results.empty:
        return DEFAULT_REST_DAYS

    fd = pd.to_datetime(fixture_date, errors="coerce")
    if pd.isna(fd):
        return DEFAULT_REST_DAYS

    df = results.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] < fd) & ((df["home"] == team) | (df["away"] == team))]
    if df.empty:
        return DEFAULT_REST_DAYS

    last_date = df["date"].max().normalize()
    days = int((fd.normalize() - last_date).days)
    return max(1, min(21, days))


def rest_strength_points(rest_days: int) -> float:
    """Convert turnaround length to a conservative scoreboard-point adjustment."""
    if rest_days <= 4:
        return -2.0
    if rest_days == 5:
        return -1.2
    if rest_days == 6:
        return -0.5
    if 7 <= rest_days <= 9:
        return 0.0
    if 10 <= rest_days <= 13:
        return 0.4
    return 0.6


def clamp_probability(p: float, low: float = 0.06, high: float = 0.94) -> float:
    return min(high, max(low, p))


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
    player_ratings = load_player_ratings()
    team_changes = load_team_changes()
    form_stats = build_recent_form_stats(results, teams)
    home_away_form = build_home_away_form_stats(results, teams)
    team_volatility = build_team_volatility(results, teams)
    team_home_edge = build_team_home_ground_edges(results, teams)
    season_record = build_current_season_record(results, 2026)
    ladder_stats = build_ladder_stats(results, teams, 2026)

    if ad_model is not None:
        ad_model["team_home_edge"] = team_home_edge
        ad_model["ladder_stats"] = ladder_stats
        ad_model["form_stats"] = form_stats
        ad_model["home_away_form"] = home_away_form

    print("[debug] ladder snapshot:")
    for team in sorted(ladder_stats.keys()):
        ls = ladder_stats[team]
        if ls["games"] > 0:
            print(
                f"  {team}: games={ls['games']} wins={ls['wins']} "
                f"diff={ls['diff']:.0f} avg_margin={ls['avg_margin']:.2f} win_pct={ls['win_pct']:.3f}"
            )

    rows = []

    for m in fixtures:
        key = (m.date, m.home, m.away)
        o = odds.get(key, {})
        home_odds = float(o.get("home_odds", float("nan")))
        away_odds = float(o.get("away_odds", float("nan")))
        market_home_prob, market_away_prob = fair_probs_from_odds(home_odds, away_odds)

        if ad_model is not None:
            raw_home_prob, exp_margin, exp_total, conf = simulate_match_ad(
                ad_model, m.home, m.away, m.venue, adj
            )
        else:
            raw_home_prob = 0.50
            exp_margin = 0.0
            exp_total = 44.0
            conf = 0.50

        # Pre-season futures priors are intentionally disabled for finals unless
        # USE_OUTRIGHT_PRIOR is explicitly switched back on.
        if USE_OUTRIGHT_PRIOR:
            raw_home_prob = apply_outright_strength_adjustment(raw_home_prob, m.home, m.away)

        raw_home_prob = apply_early_season_matchup_moderation(
            raw_home_prob, m.home, m.away, season_record
        )
        raw_home_prob = compress_prob(raw_home_prob)

        # ------------------------------------------------------------
        # AUTOMATED REST / TURNAROUND LAYER
        # ------------------------------------------------------------
        home_rest = rest_days_for_team(m.home, m.date, results)
        away_rest = rest_days_for_team(m.away, m.date, results)
        home_rest_pts = rest_strength_points(home_rest)
        away_rest_pts = rest_strength_points(away_rest)
        rest_net_home = home_rest_pts - away_rest_pts
        rest_prob_shift = max(
            -REST_PROB_SHIFT_CAP,
            min(REST_PROB_SHIFT_CAP, rest_net_home * REST_PROB_PER_POINT),
        )
        raw_home_prob = clamp_probability(raw_home_prob + rest_prob_shift)
        exp_margin += rest_net_home
        print(
            f"[rest] {m.home}={home_rest}d ({home_rest_pts:+.1f}) "
            f"{m.away}={away_rest}d ({away_rest_pts:+.1f}) net_home={rest_net_home:+.1f}"
        )

        # ------------------------------------------------------------
        # PLAYER / TEAM AVAILABILITY LAYER
        # ------------------------------------------------------------
        home_avail = team_availability_adjustment(m.home, m.date, team_changes, player_ratings)
        away_avail = team_availability_adjustment(m.away, m.date, team_changes, player_ratings)
        availability_net_home = float(home_avail["points"]) - float(away_avail["points"])
        availability_prob_shift = max(
            -AVAILABILITY_PROB_SHIFT_CAP,
            min(AVAILABILITY_PROB_SHIFT_CAP, availability_net_home * AVAILABILITY_PROB_PER_POINT),
        )
        raw_home_prob = clamp_probability(raw_home_prob + availability_prob_shift)
        exp_margin += availability_net_home

        if home_avail["change_count"] or away_avail["change_count"]:
            print(
                f"[availability] {m.home}={home_avail['points']:+.1f} "
                f"{m.away}={away_avail['points']:+.1f} net_home={availability_net_home:+.1f}"
            )
            if home_avail["summary"]:
                print(f"[availability] {m.home}: {home_avail['summary']}")
            if away_avail["summary"]:
                print(f"[availability] {m.away}: {away_avail['summary']}")

        market_weight = market_weight_from_prices(home_odds, away_odds)
        final_home_prob = anchor_to_market(raw_home_prob, market_home_prob, market_weight)

        # Rivalry volatility nudges an extreme probability slightly back toward 50/50.
        rivalry_matchups = {
            frozenset(["Sea Eagles", "Eels"]),
            frozenset(["Broncos", "Bulldogs"]),
            frozenset(["Rabbitohs", "Roosters"]),
        }
        if frozenset([m.home, m.away]) in rivalry_matchups:
            if final_home_prob >= 0.5:
                final_home_prob -= 0.015
            else:
                final_home_prob += 0.015
            final_home_prob = clamp_probability(final_home_prob)

        # Retain existing manual injury adjustments as a fallback. If player-change
        # data is present, the larger of the two signals is used for upset context
        # rather than simply stacking both injury systems together.
        legacy_home_injury = team_injury_impact(m.home, adj)
        legacy_away_injury = team_injury_impact(m.away, adj)
        home_injury = max(legacy_home_injury, float(home_avail["missing_impact"]))
        away_injury = max(legacy_away_injury, float(away_avail["missing_impact"]))

        h2h_margin_home = recent_h2h_margin(results, m.home, m.away)

        if not math.isnan(market_home_prob):
            favourite_team = m.home if market_home_prob >= 0.50 else m.away
        else:
            favourite_team = m.home if raw_home_prob >= 0.50 else m.away
        underdog_team = m.away if favourite_team == m.home else m.home

        auto_upset = compute_auto_upset_signal(
            home=m.home,
            away=m.away,
            home_model_prob=final_home_prob,
            home_market_prob=market_home_prob,
            home_odds=home_odds,
            away_odds=away_odds,
            form_stats=form_stats,
            h2h_margin_home=h2h_margin_home,
        )

        manual_upset = manual_upsets.get(key, {})
        manual_upset_team = str(manual_upset.get("upset_team", "")).strip()
        manual_upset_score = float(manual_upset.get("manual_upset_score", 0.0) or 0.0)
        upset_team = manual_upset_team if manual_upset_team in {m.home, m.away} else ""
        final_upset_score = float(auto_upset["auto_upset_score"]) + manual_upset_score

        # Short rest matters most when the favourite is the tired side.
        favourite_rest = home_rest if favourite_team == m.home else away_rest
        if favourite_rest <= 4:
            final_upset_score += 1.2
        elif favourite_rest == 5:
            final_upset_score += 0.7
        elif favourite_rest == 6:
            final_upset_score += 0.3

        # Team-sheet shock. Only boost upset risk if the favourite is materially
        # more depleted than the underdog.
        fav_missing = home_injury if favourite_team == m.home else away_injury
        dog_missing = away_injury if favourite_team == m.home else home_injury
        missing_gap = fav_missing - dog_missing
        if missing_gap >= 1.5:
            final_upset_score += 0.8
            if not upset_team:
                upset_team = underdog_team
        if missing_gap >= 3.0:
            final_upset_score += 0.7

        # Volatility from recent match margins. Original >1.2 threshold caused
        # nearly every NRL side to trigger; 10+ is a meaningful volatility level.
        home_vol = float(team_volatility.get(m.home, 0.0))
        away_vol = float(team_volatility.get(m.away, 0.0))
        if max(home_vol, away_vol) >= 12.0:
            final_upset_score += 0.8
        elif max(home_vol, away_vol) >= 10.0:
            final_upset_score += 0.4

        if not math.isnan(market_home_prob) and abs(final_home_prob - market_home_prob) < 0.05:
            final_upset_score += 0.3

        final_upset_score = min(final_upset_score, 3.0)
        print(
            f"[debug] {m.home} vs {m.away} | upset_score={final_upset_score:.2f} "
            f"| fav={favourite_team}"
        )

        if final_upset_score >= UPSET_FLAG_THRESHOLD:
            if not upset_team:
                upset_team = underdog_team
            final_home_prob = apply_upset_probability_adjustment(
                final_home_prob, upset_team, final_upset_score, m.home, m.away
            )

        final_upset_flag = 1 if final_upset_score >= UPSET_FLAG_THRESHOLD else 0

        # Keep the existing extra upset push, but slightly reduced because team
        # availability and rest now directly alter the probability beforehand.
        if final_upset_score >= 2.5:
            final_home_prob += 0.02 if underdog_team == m.home else -0.02
        elif final_upset_score >= 2.0:
            final_home_prob += 0.01 if underdog_team == m.home else -0.01

        final_home_prob = clamp_probability(final_home_prob)
        final_away_prob = 1.0 - final_home_prob

        if final_home_prob >= final_away_prob:
            predicted_winner = m.home
            pick_prob = final_home_prob
            pick_odds = home_odds
        else:
            predicted_winner = m.away
            pick_prob = final_away_prob
            pick_odds = away_odds

        fragile_favourite = 1 if (
            final_upset_flag == 1 and predicted_winner == favourite_team
        ) else 0

        market_pick_prob = market_home_prob if predicted_winner == m.home else market_away_prob
        pick_edge = value_edge(pick_prob, pick_odds)
        market_gap = abs(pick_prob - market_pick_prob) if not math.isnan(market_pick_prob) else 0.0

        rows.append(
            {
                "run_id": run_id,
                "run_utc": run_utc,
                "date": m.date,
                "kickoff_local": m.kickoff_local,
                "home": m.home,
                "away": m.away,
                "venue": m.venue,
                "predicted_winner": predicted_winner,
                "winner_confidence_band": confidence_band(pick_prob, conf, abs(exp_margin)),
                "home_win_probability_raw": round(raw_home_prob, 3),
                "away_win_probability_raw": round(1.0 - raw_home_prob, 3),
                "market_home_win_prob": round(market_home_prob, 3) if not math.isnan(market_home_prob) else np.nan,
                "final_home_win_prob": round(final_home_prob, 3),
                "final_away_win_prob": round(final_away_prob, 3),
                "win_probability": round(pick_prob, 3),
                "exp_margin_home": round(exp_margin, 1),
                "predicted_margin": round(exp_margin if predicted_winner == m.home else -exp_margin, 1),
                "exp_total": round(exp_total, 1),
                "confidence": round(conf, 2),
                "home_odds": home_odds,
                "away_odds": away_odds,
                "predicted_winner_odds": pick_odds,
                "edge": round(pick_edge, 3) if not math.isnan(pick_edge) else np.nan,
                "market_weight": round(market_weight, 3),
                "market_gap": round(market_gap, 3),
                "home_rest_days": home_rest,
                "away_rest_days": away_rest,
                "rest_net_home_pts": round(rest_net_home, 2),
                "home_availability_adjustment": round(float(home_avail["points"]), 2),
                "away_availability_adjustment": round(float(away_avail["points"]), 2),
                "availability_net_home": round(availability_net_home, 2),
                "home_team_changes": home_avail["summary"],
                "away_team_changes": away_avail["summary"],
                "home_spine_out": int(home_avail["spine_out"]),
                "away_spine_out": int(away_avail["spine_out"]),
                "home_injury_impact": round(home_injury, 2),
                "away_injury_impact": round(away_injury, 2),
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
                "fragile_favourite": fragile_favourite,

                # Legacy compatibility fields. Betting has been retired, but
                # keeping these prevents older archive/report scripts from failing.
                "required_edge": np.nan,
                "value_flag": 0,
                "bet_grade": "",
                "pick": "",
                "stake": 0.0,
                "stake_units": 0.0,
                "stake_dollars": 0.0,
                "recommended_bet": "",
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(["date", "kickoff_local"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        return df

    round_start = df["date"].min()
    round_end = round_start + pd.Timedelta(days=4)
    df = df[(df["date"] >= round_start) & (df["date"] <= round_end)].copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    avg_edge_series = pd.to_numeric(df.get("edge", np.nan), errors="coerce").dropna()
    avg_edge = float(avg_edge_series.mean()) if not avg_edge_series.empty else 0.0
    upset_count = int(pd.to_numeric(df.get("upset_flag", 0), errors="coerce").fillna(0).sum())
    availability_games = int(
        ((pd.to_numeric(df.get("home_availability_adjustment", 0), errors="coerce").fillna(0).abs() > 0)
         | (pd.to_numeric(df.get("away_availability_adjustment", 0), errors="coerce").fillna(0).abs() > 0)).sum()
    )

    round_label = f"Round window {round_start.strftime('%Y-%m-%d')} to {round_end.strftime('%Y-%m-%d')}"
    print(f"[predict] {round_label}")
    print(
        f"[predict] fixtures={len(df)} tips={len(df)} upset_flags={upset_count} "
        f"availability_adjusted_games={availability_games} avg_market_edge={avg_edge:.3f}"
    )

    return df


if __name__ == "__main__":
    df = build_predictions()
    df.to_csv("predictions.csv", index=False)
    print(f"[predict] rows={len(df)} tips={len(df)}")

