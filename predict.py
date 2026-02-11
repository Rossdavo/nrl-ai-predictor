import math
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple

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
# Team lists (trials page)
# ----------------------------
TEAMLIST_URL = "https://www.nrl.com/news/2026/02/10/witzer-pre-season-challenge-team-lists-round-2/"

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
    """
    Pulls the next `days_ahead` days of fixtures from FixtureDownload JSON feed.
    """
    now = datetime.now(SYDNEY_TZ)
    end = now + pd.Timedelta(days=days_ahead)

    r = requests.get(
        FIXTURE_FEED_URL,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    r.raise_for_status()
    data = r.json()

    matches: List[Match] = []

    for item in data:
        dt_str = item.get("date") or item.get("Date") or item.get("startDate") or item.get("StartDate")
        if not dt_str:
            continue

        try:
            dt = pd.to_datetime(dt_str, utc=True)
        except Exception:
            continue

        dt = dt.tz_convert(SYDNEY_TZ)

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
# Simple priors (small because trials are volatile)
# ----------------------------
TEAM_RATING: Dict[str, float] = {
    "Storm": 0.35,
    "Panthers": 0.35,
    "Roosters": 0.25,
    "Sharks": 0.20,
    "Bulldogs": 0.15,
    "Sea Eagles": 0.10,
    "Raiders": 0.05,
    "Cowboys": 0.05,
    "Warriors": 0.05,
    "Dolphins": 0.05,
    "Rabbitohs": 0.05,
    "Titans": 0.00,
    "Eels": 0.00,
    "Dragons": -0.05,
    "Knights": -0.05,
    "Wests Tigers": -0.10,
}

HOME_ADV = 0.10
TRIAL_NOISE_SD = 0.55
BASE_POINTS = 20.0

# ----------------------------
# TEAM LIST SCRAPE (starters only 1–13)
# ----------------------------
def _strip_html_to_text(html: str) -> str:
    html = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_starters_by_team(url: str) -> Dict[str, Dict[int, str]]:
    """
    Extracts: { "Dolphins": {1:"Jake Averillo", 2:"Jamayne Isaako", ...}, ... }
    Starters only (1–13).
    """
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
                name
            ).strip()

            if not name:
                continue

            starters.setdefault(team, {})
            starters[team].setdefault(num, name)

        return starters
    except Exception:
        return {}

# ----------------------------
# MODEL
# ----------------------------
def simulate_match(home: str, away: str, n: int = 20000, seed: int = 7) -> Tuple[float, float, float, float]:
    random.seed(seed)
    hr = TEAM_RATING.get(home, 0.0) + HOME_ADV
    ar = TEAM_RATING.get(away, 0.0)

    home_wins = 0
    margins: List[int] = []
    totals: List[int] = []

    for _ in range(n):
        h_eff = hr + random.gauss(0, TRIAL_NOISE_SD)
        a_eff = ar + random.gauss(0, TRIAL_NOISE_SD)

        exp_home = BASE_POINTS + 6.0 * (h_eff - a_eff)
        exp_away = BASE_POINTS + 6.0 * (a_eff - h_eff)

        exp_home = max(6.0, min(36.0, exp_home))
        exp_away = max(6.0, min(36.0, exp_away))

        h_score = max(0, int(round(random.gauss(exp_home, 8.0) / 2.0) * 2))
        a_score = max(0, int(round(random.gauss(exp_away, 8.0) / 2.0) * 2))

        if h_score > a_score:
            home_wins += 1
        margins.append(h_score - a_score)
        totals.append(h_score + a_score)

    win_prob = home_wins / n
    exp_margin = sum(margins) / n
    exp_total = sum(totals) / n
    conf = min(0.70, 0.45 + abs(win_prob - 0.5) * 0.9)

    return win_prob, exp_margin, exp_total, conf

# ----------------------------
# TRY SCORERS
# ----------------------------
def _try_probs_named(starters: Dict[int, str], team_exp_points: float) -> List[Tuple[str, float]]:
    exp_tries = max(1.5, team_exp_points / 4.2)

    weights_by_num = {
        2: 0.24, 5: 0.24,
        1: 0.14,
        3: 0.12, 4: 0.12,
        11: 0.08, 12: 0.08,
    }

    if not starters or len(starters) < 7:
        return []

    out: List[Tuple[str, float]] = []
    remaining_share = 1.0 - sum(weights_by_num.values())
    other_nums = [n for n in range(1, 14) if n not in weights_by_num]
    per_other = max(0.0, remaining_share / len(other_nums))

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
    exp_tries = max(1.5, team_exp_points / 4.2)
    buckets = [
        ("Winger", 0.44),
        ("Centre", 0.28),
        ("Fullback", 0.12),
        ("Edge", 0.10),
        ("Other", 0.06),
    ]
    out: List[Tuple[str, float]] = []
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

    starters_by_team = fetch_starters_by_team(TEAMLIST_URL)
    rows = []

    for m in fixtures:
        win_prob, exp_margin, exp_total, conf = simulate_match(m.home, m.away)

        exp_home_pts = (exp_total + exp_margin) / 2.0
        exp_away_pts = (exp_total - exp_margin) / 2.0

        home_named = _try_probs_named(starters_by_team.get(m.home, {}), exp_home_pts)
        away_named = _try_probs_named(starters_by_team.get(m.away, {}), exp_away_pts)

        if not home_named:
            home_named = _try_profiles_fallback(exp_home_pts)
        if not away_named:
            away_named = _try_profiles_fallback(exp_away_pts)

        rows.append({
            "mode": MODE,
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
