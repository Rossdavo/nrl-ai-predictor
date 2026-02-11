import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict

import pandas as pd

# ----------------------------
# NOTE:
# This is a "trial-friendly" starter model:
# - Simple team strength priors (editable)
# - Adds randomness for trials (rotations)
# - Produces win prob, expected margin, totals, and try-scorer style probabilities
# Later we can upgrade to real stats ingestion.
# ----------------------------

@dataclass
class Match:
    date: str  # YYYY-MM-DD
    kickoff_local: str  # HH:MM
    home: str
    away: str
    venue: str

# This week's NRL pre-season challenge fixtures (Sydney timezone dates)
# If anything changes, you can edit these lines easily.
FIXTURES: List[Match] = [
    Match("2026-02-12", "19:00", "Dolphins", "Titans", "Kayo Stadium",),
    Match("2026-02-13", "18:00", "Raiders", "Storm", "GIO Stadium",),
    Match("2026-02-13", "20:00", "Cowboys", "Panthers", "Queensland Country Bank Stadium",),
    Match("2026-02-14", "15:00", "Warriors", "Sea Eagles", "Go Media Stadium",),
    Match("2026-02-14", "17:30", "Wests Tigers", "Roosters", "Leichhardt Oval",),
    Match("2026-02-14", "19:30", "Knights", "Bulldogs", "McDonald Jones Stadium",),
    Match("2026-02-14", "20:00", "Dragons", "Rabbitohs", "Netstrata Jubilee Stadium",),
    Match("2026-02-15", "16:00", "Sharks", "Eels", "PointsBet Stadium",),
]

# Simple team strength priors (0 = average). You can edit over time.
# For trials, keep these small because squads rotate heavily.
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

# Home advantage (reduced for trials)
HOME_ADV = 0.10

# Trials: higher noise, lower confidence
TRIAL_NOISE_SD = 0.55

# Baseline points for a trial game (lower than regular season)
BASE_POINTS = 20.0

def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def simulate_match(home: str, away: str, n: int = 20000, seed: int = 7):
    random.seed(seed)
    hr = TEAM_RATING.get(home, 0.0) + HOME_ADV
    ar = TEAM_RATING.get(away, 0.0)

    home_wins = 0
    margins = []
    totals = []

    # Try model: expected tries roughly proportional to expected points / 4.2
    for _ in range(n):
        # Add trial uncertainty
        h_eff = hr + random.gauss(0, TRIAL_NOISE_SD)
        a_eff = ar + random.gauss(0, TRIAL_NOISE_SD)

        # Expected points
        # Difference drives margin, sum drives totals
        exp_home = BASE_POINTS + 6.0 * (h_eff - a_eff)
        exp_away = BASE_POINTS + 6.0 * (a_eff - h_eff)

        # Clamp to realistic trial ranges
        exp_home = max(6.0, min(36.0, exp_home))
        exp_away = max(6.0, min(36.0, exp_away))

        # Sample score via normal approx then round to rugby-ish
        h_score = max(0, int(round(random.gauss(exp_home, 8.0) / 2.0) * 2))
        a_score = max(0, int(round(random.gauss(exp_away, 8.0) / 2.0) * 2))

        if h_score > a_score:
            home_wins += 1
        margins.append(h_score - a_score)
        totals.append(h_score + a_score)

    win_prob = home_wins / n
    exp_margin = sum(margins) / n
    exp_total = sum(totals) / n

    # Confidence heuristic (lower for close matchups + trials)
    conf = min(0.70, 0.45 + abs(win_prob - 0.5) * 0.9)
    return win_prob, exp_margin, exp_total, conf

def try_scorer_probs(team_exp_points: float):
    """
    Rough try-probabilities by role (no player names yet).
    We'll upgrade this later to actual named players from team lists.
    """
    exp_tries = max(1.5, team_exp_points / 4.2)
    # Probability a try goes to each role bucket
    # (wingers highest, then centres, then fullback, then edge backrow)
    buckets = [
        ("Winger 1", 0.22),
        ("Winger 2", 0.22),
        ("Centre 1", 0.14),
        ("Centre 2", 0.14),
        ("Fullback", 0.12),
        ("Edge backrower", 0.10),
        ("Other", 0.06),
    ]
    # Convert to "at least one try by role" approximation
    # p(role scores >=1) ≈ 1 - exp(-lambda_role)
    out = []
    for name, share in buckets:
        lam = exp_tries * share
        p = 1 - math.exp(-lam)
        out.append((name, p))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:3]

def build_predictions():
    rows = []
    for m in FIXTURES:
        win_prob, exp_margin, exp_total, conf = simulate_match(m.home, m.away)
        # derive expected team points from total and margin
        exp_home_pts = (exp_total + exp_margin) / 2.0
        exp_away_pts = (exp_total - exp_margin) / 2.0

        home_try = try_scorer_probs(exp_home_pts)
        away_try = try_scorer_probs(exp_away_pts)

        rows.append({
            "date": m.date,
            "kickoff_local": m.kickoff_local,
            "venue": m.venue,
            "home": m.home,
            "away": m.away,
            "home_win_prob": round(win_prob, 3),
            "exp_margin_home": round(exp_margin, 1),
            "exp_total": round(exp_total, 1),
            "confidence": round(conf, 2),
            "home_top_try_profiles": " | ".join([f"{n} {p:.0%}" for n, p in home_try]),
            "away_top_try_profiles": " | ".join([f"{n} {p:.0%}" for n, p in away_try]),
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        })

    df = pd.DataFrame(rows).sort_values(["date", "kickoff_local"])
    return df

if __name__ == "__main__":
    df = build_predictions()
    df.to_csv("predictions.csv", index=False)
    print(df.to_string(index=False))
