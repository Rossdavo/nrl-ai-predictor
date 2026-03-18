print("[predict] predict.py loaded")

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
# MODEL / BLEND SETTINGS
# ----------------------------
MIN_EDGE = 0.035
MIN_CONF = 0.54

# Market is the stronger guide, especially on short favourites
MODEL_BLEND = 0.45
MARKET_BLEND = 0.55

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


def norm_team(name: str) -> str:
    name = str(name).strip()
    return TEAM_NAME_NORMALISE.get(name, name)


@dataclass
class Match:
    date: str
    kickoff_local: str
    home: str
    away: str
    venue: str


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
    if len(dates) == 0:
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

    if {"Date", "Home Team", "Away Team", "Result"}.issubset(cols):
        def extract_scores_2(x: object) -> Tuple[float, float]:
            s = str(x)
            m = re.search(r"(\d+)\s*[-–]\s*(\d+)", s)
            if not m:
                return (np.nan, np.nan)
            return (float(m.group(1)), float(m.group(2)))

        scores = df["Result"].apply(extract_scores_2)
        return pd.DataFrame({
            "date": pd.to_datetime(df["Date"], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d"),
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
    """
    Team-specific home-ground edge from 2025 + 2026.
    Uses weighted home win% minus away win% and converts it to a modest points bonus.
    """
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
        pts = delta * 6.0
        out[team] = float(max(-1.5, min(2.5, pts)))

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
    conf = min(0.75, 0.50 + abs(win_prob - 0.5) * 0.70)

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
    if p >= 0.67:
        return 0.67 + (p - 0.67) * 0.78
    if p <= 0.33:
        return 0.33 + (p - 0.33) * 0.78
    return p


def dynamic_required_edge(decimal_odds: float) -> float:
    if decimal_odds < 1.50:
        return 0.055
    elif decimal_odds < 1.70:
        return 0.050
    elif decimal_odds < 2.00:
        return 0.045
    elif decimal_odds < 2.80:
        return 0.040
    return 0.050


def single_bet_cap_by_odds(decimal_odds: float) -> float:
    if decimal_odds < 1.55:
        return 50.0
    if decimal_odds < 1.70:
        return 50.0
    if decimal_odds < 1.90:
        return 40.0
    if decimal_odds < 2.60:
        return 30.0
    return min(40.0, MAX_SINGLE_BET)


def stake_band_dollars(
    decimal_odds: float,
    edge: float,
    confidence: float,
    fragile_favourite: int,
    upset_penalty_factor: float,
    sample_scale: float,
    volatility_penalty: float,
    exp_margin: float,
) -> float:
    if decimal_odds < 2.0:
        if edge >= 0.16:
            stake = 50.0
        elif edge >= 0.12:
            stake = 40.0
        elif edge >= 0.08:
            stake = 30.0
        elif edge >= 0.05:
            stake = 20.0
        else:
            stake = 0.0
    else:
        if edge >= 0.14:
            stake = 40.0
        elif edge >= 0.10:
            stake = 30.0
        elif edge >= 0.07:
            stake = 20.0
        elif edge >= 0.04:
            stake = 10.0
        else:
            stake = 0.0

    if stake <= 0:
        return 0.0

    if decimal_odds < 1.50 and edge >= 0.10 and confidence >= 0.64 and exp_margin >= 10.0:
        stake = max(stake, 40.0)
    if decimal_odds < 1.45 and edge >= 0.14 and confidence >= 0.66 and exp_margin >= 12.0:
        stake = max(stake, 50.0)

    multiplier = 1.0

    if confidence >= 0.68:
        multiplier *= 1.15
    elif confidence < 0.58:
        multiplier *= 0.90

    if fragile_favourite:
        multiplier *= 0.75

    multiplier *= upset_penalty_factor
    multiplier *= sample_scale
    multiplier *= volatility_penalty

    stake *= multiplier

    floor = MIN_BET_SHORT if decimal_odds < 2.0 else MIN_BET_DOG
    if stake > 0:
        stake = max(floor, stake)

    stake = min(stake, single_bet_cap_by_odds(decimal_odds), MAX_SINGLE_BET)
    return round(stake, 2)


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


def build_predictions() -> pd.DataFrame:
    fixtures: List[Match] = []

    if MODE == "AUTO":
        fixtures = fixtures_from_odds_csv("odds.csv")
        if not fixtures:
            fixtures = fetch_upcoming_fixtures(days_ahead=21)
        if not fixtures:
            raise SystemExit("[stop] No upcoming fixtures found from odds.csv or the fixture feed. Not showing trial games.")

        fixtures = _dedupe_fixtures(fixtures)
        fixtures = _filter_current_round_fixtures(fixtures)

        print(f"[debug] fixtures after dedupe/current-round filter: {len(fixtures)}")
        for m in fixtures:
            print(f"[debug] current-round fixture: {m.date} {m.home} v {m.away}")
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

    if ad_model is not None:
        ad_model["team_home_edge"] = team_home_edge

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
            print(" ", k)
        raise SystemExit("Stopping because odds are missing. Update odds.csv then rerun.")

    rows = []

    for m in fixtures:
        if ad_model:
            model_home_prob, exp_margin, exp_total, conf = simulate_match_ad(ad_model, m.home, m.away, m.venue, adj)
            rating_mode = "ATTACK_DEFENCE"
        else:
            model_home_prob, exp_margin, exp_total, conf = 0.50, 0.0, 40.0, 0.45
            rating_mode = "FALLBACK"

        key = (m.date, m.home, m.away)
        o = odds.get(key, {})
        home_odds = o.get("home_odds", float("nan"))
        away_odds = o.get("away_odds", float("nan"))

        market_home_prob, market_away_prob = fair_probs_from_odds(home_odds, away_odds)
        if math.isnan(market_home_prob):
            blended_home_prob = model_home_prob
        else:
            blended_home_prob = (MODEL_BLEND * model_home_prob) + (MARKET_BLEND * market_home_prob)

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

        home_edge = value_edge(final_home_prob, home_odds)
        away_edge = value_edge(1.0 - final_home_prob, away_odds)

        value_flag = ""
        pick = ""
        edge = float("nan")
        stake_units = 0.0
        stake_dollars = 0.0
        market_agreement = 1.0
        req_edge = MIN_EDGE
        fragile_favourite = 0

        favourite_team = auto_upset["favourite_team"]
        underdog_team = auto_upset["underdog_team"]

        if rating_mode != "ATTACK_DEFENCE":
            value_flag = "MODEL OFF (FALLBACK)"
        else:
            if not math.isnan(home_edge) and home_edge >= MIN_EDGE:
                value_flag = f"HOME VALUE +{home_edge:.0%}"
            elif not math.isnan(away_edge) and away_edge >= MIN_EDGE:
                value_flag = f"AWAY VALUE +{away_edge:.0%}"

            best_side = ""
            best_edge = float("-inf")
            best_prob = 0.0
            best_odds = float("nan")
            side_team = ""

            if not math.isnan(home_edge) and home_edge > best_edge:
                best_side = "HOME"
                best_edge = home_edge
                best_prob = final_home_prob
                best_odds = home_odds
                side_team = m.home

            if not math.isnan(away_edge) and away_edge > best_edge:
                best_side = "AWAY"
                best_edge = away_edge
                best_prob = 1.0 - final_home_prob
                best_odds = away_odds
                side_team = m.away

            if best_side:
                model_side_prob = model_home_prob if best_side == "HOME" else (1.0 - model_home_prob)
                market_side_prob = market_home_prob if best_side == "HOME" else market_away_prob

                if not math.isnan(market_side_prob):
                    prob_gap = abs(model_side_prob - market_side_prob)
                    market_agreement = max(0.78, 1.03 - (prob_gap * 3.5))
                else:
                    market_agreement = 0.93

                req_edge = dynamic_required_edge(best_odds)

            qualifies = bool(
                best_side and
                best_edge >= req_edge and
                conf >= MIN_CONF
            )

            if qualifies and best_odds > 3.50:
                qualifies = False
            if qualifies and best_odds < 1.30:
                qualifies = False

            upset_penalty_factor = 1.0

            fragile_score = 0
            if qualifies and side_team == favourite_team:
                fav_market_prob = market_home_prob if side_team == m.home else market_away_prob

                if best_odds < 1.70 and not math.isnan(fav_market_prob) and best_prob < (fav_market_prob - 0.05):
                    fragile_score += 1
                if final_upset_score >= 1.5:
                    fragile_score += 1
                if exp_margin < 5.0 and best_odds < 1.70:
                    fragile_score += 1
                if team_volatility.get(side_team, 0.0) >= 10.5:
                    fragile_score += 1

                fragile_favourite = 1 if fragile_score >= 2 else 0

            if qualifies and fragile_favourite:
                upset_penalty_factor *= 0.75
                if best_odds < 1.70 and best_edge < max(req_edge, 0.055):
                    qualifies = False

            if qualifies and side_team == underdog_team and final_upset_score >= UPSET_FLAG_THRESHOLD:
                upset_penalty_factor *= min(1.10, 1.0 + (0.04 * final_upset_score))

            # manual upset / outsider value override
            if qualifies and manual_upset_team in {m.home, m.away}:
                if side_team == manual_upset_team and side_team == underdog_team:
                    best_edge += 0.01
                    upset_penalty_factor *= 1.08

                if side_team == favourite_team and manual_upset_team == underdog_team:
                    if best_edge < 0.07:
                        qualifies = False
                    else:
                        upset_penalty_factor *= 0.85

            home_games = float(form_stats.get(m.home, {}).get("games", 0.0))
            away_games = float(form_stats.get(m.away, {}).get("games", 0.0))
            min_games = min(home_games, away_games)

            sample_scale = 1.0
            if min_games < 3:
                sample_scale = 0.75
            elif min_games < 5:
                sample_scale = 0.88

            side_vol = float(team_volatility.get(side_team, 0.0))
            volatility_penalty = 1.0
            if side_vol >= 12.0:
                volatility_penalty = 0.72
            elif side_vol >= 10.0:
                volatility_penalty = 0.82
            elif side_vol >= 8.5:
                volatility_penalty = 0.90

            if qualifies:
                stake_dollars = stake_band_dollars(
                    decimal_odds=best_odds,
                    edge=best_edge,
                    confidence=conf,
                    fragile_favourite=fragile_favourite,
                    upset_penalty_factor=upset_penalty_factor,
                    sample_scale=sample_scale,
                    volatility_penalty=volatility_penalty,
                    exp_margin=exp_margin,
                )

                if stake_dollars > 0:
                    pick = best_side
                    edge = best_edge
                    stake_units = round(stake_dollars / UNIT_SIZE, 2) if UNIT_SIZE > 0 else 0.0

        rows.append({
            "mode": MODE,
            "rating_mode": rating_mode,
            "date": m.date,
            "kickoff_local": m.kickoff_local,
            "venue": m.venue,
            "home": m.home,
            "away": m.away,
            "model_home_win_prob": round(model_home_prob, 3),
            "market_home_win_prob": round(market_home_prob, 3) if not math.isnan(market_home_prob) else np.nan,
            "final_home_win_prob": round(final_home_prob, 3),
            "exp_margin_home": round(exp_margin, 1),
            "exp_total": round(exp_total, 1),
            "confidence": round(conf, 2),
            "home_odds": home_odds,
            "away_odds": away_odds,
            "favourite_team": favourite_team,
            "underdog_team": underdog_team,
            "auto_upset_score": round(float(auto_upset["auto_upset_score"]), 2),
            "auto_upset_reasons": auto_upset["auto_upset_reasons"],
            "manual_upset_team": manual_upset_team,
            "manual_upset_score": round(manual_upset_score, 2),
            "manual_upset_notes": manual_upset.get("notes", ""),
            "upset_team": upset_team,
            "final_upset_score": round(final_upset_score, 2),
            "upset_flag": final_upset_flag,
            "fragile_favourite": int(fragile_favourite),
            "required_edge": round(req_edge, 3),
            "value_flag": value_flag,
            "pick": pick,
            "edge": round(edge, 3) if not math.isnan(edge) else 0.0,
            "stake": float(stake_units),
            "stake_units": float(stake_units),
            "stake_dollars": float(stake_dollars),
            "recommended_bet": (
                f"${stake_dollars:.2f} {m.home if pick == 'HOME' else m.away}"
                if stake_dollars > 0 and pick in {"HOME", "AWAY"}
                else "No Bet"
            ),
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        })

    df = pd.DataFrame(rows).sort_values(["date", "kickoff_local"]).reset_index(drop=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    round_start = df["date"].min()
    round_end = round_start + pd.Timedelta(days=3)
    df = df[(df["date"] >= round_start) & (df["date"] <= round_end)].copy()

    if "stake_dollars" in df.columns:
        df["stake_dollars"] = pd.to_numeric(df["stake_dollars"], errors="coerce").fillna(0.0)
        df["stake_units"] = pd.to_numeric(df["stake_units"], errors="coerce").fillna(0.0)
        df["stake"] = pd.to_numeric(df["stake"], errors="coerce").fillna(0.0)
        df["edge"] = pd.to_numeric(df["edge"], errors="coerce").fillna(0.0)

        bet_mask = df["stake_dollars"] > 0
        bet_df = df[bet_mask].copy()
        bet_df = bet_df.sort_values(
            ["edge", "confidence", "final_upset_score", "date", "kickoff_local"],
            ascending=[False, False, True, True, True]
        ).reset_index()

        running_exposure = 0.0
        short_fav_exposure = 0.0
        keep_original_idx = []

        for _, row in bet_df.iterrows():
            stake_amt = float(row["stake_dollars"])
            odds_amt = float(row["home_odds"]) if row["pick"] == "HOME" else float(row["away_odds"])
            is_short_fav = odds_amt < 1.65

            if running_exposure + stake_amt > MAX_ROUND_EXPOSURE:
                continue
            if is_short_fav and (short_fav_exposure + stake_amt) > (MAX_ROUND_EXPOSURE * 0.60):
                continue

            keep_original_idx.append(int(row["index"]))
            running_exposure += stake_amt
            if is_short_fav:
                short_fav_exposure += stake_amt

        excluded_mask = bet_mask & (~df.index.isin(keep_original_idx))
        df.loc[excluded_mask, "pick"] = ""
        df.loc[excluded_mask, "value_flag"] = ""
        df.loc[excluded_mask, "edge"] = 0.0
        df.loc[excluded_mask, "stake"] = 0.0
        df.loc[excluded_mask, "stake_units"] = 0.0
        df.loc[excluded_mask, "stake_dollars"] = 0.0
        df.loc[excluded_mask, "recommended_bet"] = "No Bet"

        print(f"[predict] exposure cap applied: ${running_exposure:.2f} / ${MAX_ROUND_EXPOSURE:.2f}")

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    round_label = f"Round window {round_start.strftime('%Y-%m-%d')} to {round_end.strftime('%Y-%m-%d')}"
    bet_count = int((pd.to_numeric(df.get("stake_units", 0), errors="coerce").fillna(0) > 0).sum())
    exposure = float(pd.to_numeric(df.get("stake_dollars", 0), errors="coerce").fillna(0).sum())
    avg_edge_series = pd.to_numeric(df.get("edge", 0), errors="coerce").fillna(0).replace(0, pd.NA).dropna()
    avg_edge = float(avg_edge_series.mean()) if not avg_edge_series.empty else 0.0

    print(f"[predict] {round_label}")
    print(f"[predict] current round fixtures={len(df)} bets={bet_count} exposure=${exposure:.2f} avg_edge={avg_edge:.3f}")

    return df


def load_results_csv(path: str) -> pd.DataFrame:
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
