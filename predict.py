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
