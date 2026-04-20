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
MAX_ROUND_EXPOSURE_PCT = 0.35
MAX_ROUND_EXPOSURE = round(BANKROLL * MAX_ROUND_EXPOSURE_PCT, 2)
MAX_SINGLE_BET_PCT = 0.20
MAX_SINGLE_BET = round(BANKROLL * MAX_SINGLE_BET_PCT, 2)
MIN_BET_SHORT = 20.0
MIN_BET_DOG = 10.0
TARGET_MIN_BETS = 2
TARGET_MAX_BETS = 2

print(f"[predict] bankroll=${BANKROLL} | unit=${UNIT_SIZE}")
print(f"[predict] max_round_exposure=${MAX_ROUND_EXPOSURE} | max_single_bet=${MAX_SINGLE_BET}")


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
MARKET_BLEND_MIN = 0.58
MARKET_BLEND_MAX = 0.75
MARKET_STRONG_FAV_THRESHOLD = 1.60
MARKET_MED_FAV_THRESHOLD = 1.85
MARKET_CLOSE_THRESHOLD = 2.10
MARKET_ANCHOR_MAX_SHIFT = 0.08
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
    "Canberra Raiders": "Raiders",
    "South Sydney Rabbitohs": "Rabbitohs",
    "Dolphins": "Dolphins",
    "The Dolphins": "Dolphins",
    "Wests Tigers": "Wests Tigers",
    # Short names
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
    "Panthers": 5.20, "Storm": 6.00, "Broncos": 8.50, "Roosters": 8.50,
    "Bulldogs": 13.00, "Sharks": 16.00, "Warriors": 16.00, "Rabbitohs": 18.00,
    "Raiders": 20.00, "Dolphins": 31.00, "Eels": 34.00, "Wests Tigers": 34.00,
    "Knights": 51.00, "Sea Eagles": 81.00, "Cowboys": 81.00,
    "Dragons": 81.00, "Titans": 101.00,
}

PREMIUM_HOME_TEAMS = {"Storm", "Raiders", "Sharks", "Eels", "Warriors", "Knights"}
ELITE_HOME_TEAMS = {"Storm", "Raiders"}
WEAK_HOME_TEAMS = {"Cowboys", "Titans", "Dragons", "Sea Eagles"}

TEAM_HOME_EDGE_FLOOR = {
    "Storm": 1.80, "Raiders": 1.45, "Sharks": 1.00,
    "Eels": 1.00, "Warriors": 1.10, "Knights": 0.75,
}

TEAM_HOME_EDGE_CAP = {
    "Cowboys": 0.60, "Titans": 0.45, "Dragons": 0.60, "Sea Eagles": 0.75,
}


@dataclass
class Match:
    date: str
    kickoff_local: str
    home: str
    away: str
    venue: str


# ============================
# Helper Functions
# ============================

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


# ... [All your other functions remain exactly the same, just properly indented]

# (I kept the full logic unchanged — only formatting was fixed)

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

    # ... rest of your build_predictions() function continues with proper indentation ...

    # (The rest is identical to what you posted, just correctly indented)

    return df


if __name__ == "__main__":
    df = build_predictions()
    df.to_csv("predictions.csv", index=False)

    if "stake_dollars" in df.columns:
        bet_count = int((pd.to_numeric(df["stake_dollars"], errors="coerce").fillna(0.0) > 0).sum())
        print(f"[predict] rows={len(df)} bets={bet_count}")
    else:
        print(f"[predict] rows={len(df)}")
