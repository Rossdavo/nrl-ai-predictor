import math
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo

# ----------------------------
# RUN MODE
# "TRIALS" = use hardcoded fixtures
# "AUTO"   = pull upcoming fixtures automatically
# ----------------------------
MODE = "TRIALS"

# ----------------------------
# Team lists (trials page) – optional
# ----------------------------
TEAMLIST_URL = "https://www.nrl.com/news/2026/02/10/witzer-pre-season-challenge-team-lists-round-2/"

# ----------------------------
# Results source for ratings (Attack/Defence)
# ----------------------------
RESULTS_URL = "https://fixturedownload.com/results/nrl-2026"

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
    Returns dataframe with columns: home, away, home_pts, away_pts
    Uses FixtureDownload results table.
    """
    r = requests.get(RESULTS_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    # First table on the page is the fixture/results table
    tables = pd.read_html(r.text)
    df = tables[0].copy()

    # Expected columns (site may vary slightly): Home Team, Away Team, Result
    # Normalise column names
    df.columns = [str(c).strip() for c in df.columns]
    home_col = next((c for c in df.columns if "Home" in c), None)
    away_col = next((c for c in df.columns if "Away" in c), None)
    res_col  = next((c for c in df.columns if "Result" in c), None)

    if not home_col or not away_col or not res_col:
        return pd.DataFrame(columns=["home", "away", "home_pts", "away_pts"])

    out_rows = []
    for _, row in df.iterrows():
        home = str(row.get(home_col, "")).strip()
        away = str(row.get(away_col, "")).strip()
        res  = str(row.get(res_col, "")).strip()

        # Completed matches look like "20 - 18" (future matches are "-" or empty)
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", res)
        if not m:
            continue

        out_rows.append({
            "home": home,
            "away": away,
            "home_pts": int(m.group(1)),
            "away_pts": int(m.group(2)),
        })

    return pd.DataFrame(out_rows)

def fit_attack_defence(results: pd.DataFrame, teams: List[str]) -> Optional[Dict[str, object]]:
    """
    Least squares fit:
      HomePts = mu + home_adv + atk_home - def_away
      AwayPts = mu          + atk_away - def_home

    Returns dict with mu, home_adv, atk (dict), dfn (dict)
    If insufficient results, returns None.
    """
    if results is None or results.empty or len(results) < 8:
        return None

    team_to_i = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    # Parameters: [mu, home_adv, atk_0..atk_{T-1}, def_0..def_{T-1}]
    p = 2 + 2 * n_teams
    X = []
    y = []

    for _, r in results.iterrows():
        h = r["home"]; a = r["away"]
        if h not in team_to_i or a not in team_to_i:
            continue
        hi = team_to_i[h]; ai = team_to_i[a]

        # Home score row
        row = np.zeros(p)
        row[0] = 1.0                 # mu
        row[1] = 1.0                 # home_adv
        row[2 + hi] = 1.0            # atk_home
        row[2 + n_teams + ai] = -1.0 # -def_away
        X.append(row); y.append(float(r["home_pts"]))

        # Away score row
        row = np.zeros(p)
        row[0] = 1.0                 # mu
        row[1] = 0.0                 # no home_adv
        row[2 + ai] = 1.0            # atk_away
        row[2 + n_teams + hi] = -1.0 # -def_home
        X.append(row); y.append(float(r["away_pts"]))

    if len(y) < 16:
        return None

    X = np.vstack(X)
    y = np.array(y)

    # Ridge to stabilise early-season fits
    ridge = 1.0
    XtX = X.T @ X + ridge * np.eye(p)
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)

    mu = float(beta[0])
    home_adv = float(beta[1])

    atk = beta[2:2+n_teams].copy()
    dfn = beta[2+n_teams:2+2*n_teams].copy()

    # Centre them (identifiability)
    atk -= atk.mean()
    dfn -= dfn.mean()

    atk_map = {t: float(atk[team_to_i[t]]) for t in teams}
    dfn_map = {t: float(dfn[team_to_i[t]]) for t in teams}

    return {"mu": mu, "home_adv": home_adv, "atk": atk_map, "dfn": dfn_map}

def expected_points(model: Dict[str, object], home: str, away: str) -> Tuple[float, float]:
    mu = model["mu"]
    ha = model["home_adv"]
    atk = model["atk"]
    dfn = model["dfn"]
    home_pts = mu + ha + atk.get(home, 0.0) - dfn.get(away, 0.0)
    away_pts = mu +      atk.get(away, 0.0) - dfn.get(home, 0.0)
    # clamp to sensible range
    return (max(4.0, min(40.0, home_pts)), max(4.0, min(40.0, away_pts)))

def simulate_match_ad(model: Dict[str, object], home: str, away: str, n: int = 20000, seed: int = 7) -> Tuple[float, float, float, float]:
    random.seed(seed)
    hw = 0
    margins = []
    totals = []

    exp_home, exp_away = expected_points(model, home, away)

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

# ----------------------------
# BUILD OUTPUT
# ----------------------------
def build_predictions() -> pd.DataFrame:
    if MODE == "AUTO":
        fixtures = fetch_upcoming_fixtures(days_ahead=7)
    else:
        fixtures = FIXTURES

    # Fit Attack/Defence model when there are enough completed results
    teams = sorted(list({m.home for m in fixtures} | {m.away for m in fixtures}))
    results = fetch_completed_results()
    ad_model = fit_attack_defence(results, teams)

    starters_by_team = fetch_starters_by_team(TEAMLIST_URL)
    rows = []

    for m in fixtures:
        if ad_model:
            win_prob, exp_margin, exp_total, conf = simulate_match_ad(ad_model, m.home, m.away)
            exp_home_pts = (exp_total + exp_margin) / 2.0
            exp_away_pts = (exp_total - exp_margin) / 2.0
            rating_mode = "ATTACK_DEFENCE"
        else:
            # fallback early season/trials: simple priors
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
