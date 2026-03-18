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

BANKROLL = 200.0
UNIT_PCT = 0.05
UNIT_SIZE = round(BANKROLL * UNIT_PCT, 2)
MAX_ROUND_EXPOSURE_PCT = 0.40
MAX_ROUND_EXPOSURE = BANKROLL * MAX_ROUND_EXPOSURE_PCT

print(f"[predict] bankroll=${BANKROLL} | unit=${UNIT_SIZE}")

# ----------------------------
# RUN MODE
# ----------------------------
MODE = "AUTO"

# ----------------------------
# Optional try scorer fallback control
# ----------------------------
FORCE_TRY_FALLBACK = False
TRYSCORERS_CSV_PATH = "try_scorers.csv"

# ----------------------------
# Results source for ratings
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
    # short -> short
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
    "Dolphins": "Dolphins",
}


def norm_team(name: str) -> str:
    name = str(name).strip()
    return TEAM_NAME_NORMALISE.get(name, name)


# ----------------------------
# Regions
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


# ----------------------------
# RESULTS INGEST
# ----------------------------
def fetch_completed_results() -> pd.DataFrame:
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
    seed: int = 7,
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
# TRY SCORERS
# ----------------------------
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


def load_bookmaker_try_scorers(path: str = TRYSCORERS_CSV_PATH) -> Dict[Tuple[str, str, str], Dict[str, List[Tuple[str, float]]]]:
    """
    Reads:
      date,home,away,team,player,odds,rank

    Returns:
      {
        (date, home, away): {
          "home": [(player, odds), ...],
          "away": [(player, odds), ...],
        }
      }
    """
    if not os.path.exists(path):
        print(f"[warn] try scorers file not found: {path}")
        return {}

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[warn] could not read {path}: {e}")
        return {}

    required = {"date", "home", "away", "team", "player", "odds"}
    if not required.issubset(set(df.columns)):
        print(f"[warn] {path} missing required columns. Need {sorted(required)}")
        return {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["home"] = df["home"].astype(str).apply(norm_team)
    df["away"] = df["away"].astype(str).apply(norm_team)
    df["team"] = df["team"].astype(str).apply(norm_team)
    df["player"] = df["player"].astype(str).str.strip()
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df = df.dropna(subset=["date", "home", "away", "team", "player", "odds"])

    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    else:
        df["rank"] = np.nan

    out: Dict[Tuple[str, str, str], Dict[str, List[Tuple[str, float]]]] = {}

    for (date, home, away), g in df.groupby(["date", "home", "away"], dropna=False):
        g = g.copy()

        if g["rank"].notna().any():
            g = g.sort_values(["team", "rank", "odds", "player"], ascending=[True, True, True, True])
        else:
            g = g.sort_values(["team", "odds", "player"], ascending=[True, True, True])

        home_rows = g[g["team"] == home].head(3)
        away_rows = g[g["team"] == away].head(3)

        out[(date, home, away)] = {
            "home": [(str(r["player"]), float(r["odds"])) for _, r in home_rows.iterrows()],
            "away": [(str(r["player"]), float(r["odds"])) for _, r in away_rows.iterrows()],
        }

    print(f"[info] Loaded bookmaker try scorers: {path} ({len(out)} fixtures)")
    return out


def _format_try_scorers_bookmaker(rows: List[Tuple[str, float]]) -> str:
    if not rows:
        return ""
    return " | ".join([f"{name} {odds:.2f}" for name, odds in rows[:3]])


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


def kelly_stake_dollars(model_prob: float, decimal_odds: float, bankroll: float, confidence: float) -> float:
    if decimal_odds <= 1.0 or model_prob <= 0.0 or model_prob >= 1.0:
        return 0.0

    b = decimal_odds - 1.0
    p = model_prob
    q = 1.0 - p
    raw_kelly = ((b * p) - q) / b
    if raw_kelly <= 0:
        return 0.0

    adj_kelly = raw_kelly * 0.25
    conf_scale = max(0.5, min(1.0, confidence / 0.80))
    adj_kelly *= conf_scale
    adj_kelly = min(adj_kelly, 0.10)

    return round(bankroll * adj_kelly, 2)


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
            venue="",
        ))

    fixtures.sort(key=lambda m: (m.date, m.kickoff_local))
    return fixtures


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
    print(f"[debug] combined results rows={len(results)} (deduped)")

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
    bookmaker_try_scorers = load_bookmaker_try_scorers(TRYSCORERS_CSV_PATH)

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

        key = (m.date, m.home, m.away)
        bm_try = bookmaker_try_scorers.get(key, {})

        if FORCE_TRY_FALLBACK:
            home_named = _try_profiles_fallback(exp_home_pts)
            away_named = _try_profiles_fallback(exp_away_pts)
            home_top_try_str = " | ".join([f"{n} {p:.0%}" for n, p in home_named])
            away_top_try_str = " | ".join([f"{n} {p:.0%}" for n, p in away_named])
            try_scorer_source = "forced fallback"
        else:
            home_bm = bm_try.get("home", [])
            away_bm = bm_try.get("away", [])

            if home_bm and away_bm:
                home_top_try_str = _format_try_scorers_bookmaker(home_bm)
                away_top_try_str = _format_try_scorers_bookmaker(away_bm)
                try_scorer_source = "bookmaker try scorers"
            else:
                home_named = _try_profiles_fallback(exp_home_pts)
                away_named = _try_profiles_fallback(exp_away_pts)
                home_top_try_str = " | ".join([f"{n} {p:.0%}" for n, p in home_named])
                away_top_try_str = " | ".join([f"{n} {p:.0%}" for n, p in away_named])
                try_scorer_source = "fallback (no bookmaker try scorers)"

        o = odds.get(key, {})
        home_odds = o.get("home_odds", float("nan"))
        away_odds = o.get("away_odds", float("nan"))

        home_edge = float("nan")
        away_edge = float("nan")
        value_flag = ""
        pick = ""
        edge = float("nan")
        stake_units = 0.0
        stake_dollars = 0.0

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

                if best_side == "HOME":
                    stake_dollars = kelly_stake_dollars(win_prob, home_odds, BANKROLL, conf)
                else:
                    stake_dollars = kelly_stake_dollars(1 - win_prob, away_odds, BANKROLL, conf)

                stake_units = round(stake_dollars / UNIT_SIZE, 2) if UNIT_SIZE > 0 else 0.0

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
            "stake": float(stake_units),
            "stake_units": float(stake_units),
            "stake_dollars": float(stake_dollars),
            "recommended_bet": (
                f"${stake_dollars:.2f} {m.home if pick == 'HOME' else m.away}"
                if stake_dollars > 0 and pick in {"HOME", "AWAY"}
                else "No Bet"
            ),
            "home_top_try": home_top_try_str,
            "away_top_try": away_top_try_str,
            "try_scorer_source": try_scorer_source,
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
        bet_df = bet_df.sort_values(["edge", "date", "kickoff_local"], ascending=[False, True, True]).reset_index()

        running_exposure = 0.0
        keep_original_idx = []

        for _, row in bet_df.iterrows():
            stake_amt = float(row["stake_dollars"])
            if running_exposure + stake_amt <= MAX_ROUND_EXPOSURE:
                keep_original_idx.append(int(row["index"]))
                running_exposure += stake_amt

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
