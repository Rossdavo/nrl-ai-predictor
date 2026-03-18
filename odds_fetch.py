import os
import json
import requests
import pandas as pd
from datetime import datetime, timezone

API_KEY = os.getenv("ODDS_API_KEY")

SPORT = "rugbyleague_nrl"
REGIONS = "au"
H2H_MARKETS = "h2h"
TRY_MARKETS = "player_try_scorer"
BOOKMAKERS = "sportsbet,tab,pointsbetau"

BASE_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
EVENT_ODDS_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events"

ODDS_OUT = "odds.csv"
TRY_OUT = "try_scorers.csv"

# Prefer one bookmaker for try scorers so prices are consistent.
TRY_BOOK_PRIORITY = ["sportsbet", "tab", "pointsbetau"]

TEAM_ALIASES = {
    "Wests Tigers": ["Wests Tigers", "Wests Tigers NRL", "Wests Tigers (NRL)"],
    "Sea Eagles": ["Manly", "Manly Sea Eagles", "Manly-Warringah Sea Eagles", "Sea Eagles"],
    "Rabbitohs": ["Souths", "South Sydney", "South Sydney Rabbitohs", "Rabbitohs"],
    "Roosters": ["Sydney Roosters", "Roosters"],
    "Bulldogs": ["Canterbury", "Canterbury Bulldogs", "Canterbury-Bankstown Bulldogs", "Bulldogs"],
    "Eels": ["Parramatta", "Parramatta Eels", "Eels"],
    "Knights": ["Newcastle", "Newcastle Knights", "Knights"],
    "Dragons": ["St George", "St George Illawarra", "St George Illawarra Dragons", "Dragons"],
    "Sharks": ["Cronulla", "Cronulla Sharks", "Cronulla-Sutherland Sharks", "Cronulla Sutherland Sharks", "Sharks"],
    "Storm": ["Melbourne Storm", "Storm"],
    "Raiders": ["Canberra Raiders", "Raiders"],
    "Warriors": ["NZ Warriors", "New Zealand Warriors", "Warriors"],
    "Panthers": ["Penrith Panthers", "Panthers"],
    "Cowboys": ["North Queensland Cowboys", "Cowboys"],
    "Titans": ["Gold Coast Titans", "Titans"],
    "Dolphins": ["The Dolphins", "Dolphins"],
    "Broncos": ["Brisbane Broncos", "Broncos"],
}

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
    "Wests Tigers": "Wests Tigers",
    "Dolphins": "Dolphins",
}


def norm_team(name: str) -> str:
    name = str(name or "").strip()
    return TEAM_NAME_NORMALISE.get(name, name)


def norm(s: str) -> str:
    return (s or "").strip().lower()


def matches_team(raw_name: str, team_name: str) -> bool:
    raw = norm(raw_name)
    team = norm(team_name)
    if raw == team:
        return True
    for alias in TEAM_ALIASES.get(team_name, []):
        if norm(alias) == raw:
            return True
    return False


def canonical_team_from_text(text: str, home: str, away: str) -> str:
    txt = norm(text)
    if not txt:
        return ""

    if matches_team(txt, home):
        return home
    if matches_team(txt, away):
        return away

    for team in [home, away]:
        for alias in TEAM_ALIASES.get(team, []):
            if norm(alias) in txt:
                return team

    return ""


def best_h2h_prices(bookmakers, home, away):
    home_best = None
    away_best = None

    for book in bookmakers or []:
        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("name")
                price = outcome.get("price")
                if price is None:
                    continue
                if matches_team(name, home):
                    home_best = price if home_best is None else max(home_best, price)
                if matches_team(name, away):
                    away_best = price if away_best is None else max(away_best, price)

    return home_best, away_best


def bookmaker_sort_key(book_key: str) -> int:
    try:
        return TRY_BOOK_PRIORITY.index(book_key)
    except ValueError:
        return 999


def infer_team_from_outcome(outcome: dict, home: str, away: str) -> str:
    """
    Best-effort team inference from available outcome fields.
    """
    candidate_fields = [
        outcome.get("team"),
        outcome.get("description"),
        outcome.get("participant"),
        outcome.get("group"),
        outcome.get("title"),
        outcome.get("label"),
        outcome.get("header"),
        outcome.get("side"),
    ]

    for value in candidate_fields:
        team = canonical_team_from_text(str(value or ""), home, away)
        if team:
            return team

    blob = json.dumps(outcome, ensure_ascii=False)
    team = canonical_team_from_text(blob, home, away)
    if team:
        return team

    return ""


def looks_like_player_name(name: str) -> bool:
    if not name:
        return False
    name = str(name).strip()
    if len(name) < 4:
        return False
    if "over" in name.lower() or "under" in name.lower():
        return False
    parts = [p for p in name.split() if p]
    return len(parts) >= 2


def fetch_event_try_scorers(event_id: str, home: str, away: str) -> list:
    """
    Fetch anytime try scorer market for one event and return
    rows with team attribution when available.
    """
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": TRY_MARKETS,
        "bookmakers": BOOKMAKERS,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    url = f"{EVENT_ODDS_URL}/{event_id}/odds"
    r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    data = r.json()

    bookmakers = data.get("bookmakers", [])
    bookmakers = sorted(bookmakers, key=lambda b: bookmaker_sort_key(b.get("key", "")))

    rows = []
    for book in bookmakers:
        book_key = book.get("key", "")
        for market in book.get("markets", []):
            if market.get("key") != TRY_MARKETS:
                continue

            market_rows = []
            for outcome in market.get("outcomes", []):
                player = str(outcome.get("name", "")).strip()
                odds = outcome.get("price")
                if odds is None or not looks_like_player_name(player):
                    continue

                team = infer_team_from_outcome(outcome, home, away)
                if not team:
                    continue

                market_rows.append({
                    "bookmaker": book_key,
                    "team": team,
                    "player": player,
                    "odds": float(odds),
                })

            # If this bookmaker has usable, team-attributed rows, prefer it and stop.
            if market_rows:
                rows.extend(market_rows)
                return rows

    return rows


def write_empty_outputs():
    pd.DataFrame(
        columns=["date", "home", "away", "home_odds", "away_odds", "captured_at_utc"]
    ).to_csv(ODDS_OUT, index=False)

    pd.DataFrame(
        columns=["date", "home", "away", "team", "player", "odds", "rank", "bookmaker", "event_id", "captured_at_utc"]
    ).to_csv(TRY_OUT, index=False)


def main():
    if not API_KEY:
        print("No ODDS_API_KEY provided")
        write_empty_outputs()
        return

    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": H2H_MARKETS,
        "bookmakers": BOOKMAKERS,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    r = requests.get(BASE_URL, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    data = r.json()

    odds_rows = []
    try_rows = []
    priced = 0
    try_events_with_rows = 0

    captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    for game in data:
        event_id = game.get("id")
        home = norm_team(game.get("home_team"))
        away = norm_team(game.get("away_team"))
        commence = game.get("commence_time")

        if not home or not away or not commence:
            continue

        date = str(commence)[:10]
        home_odds, away_odds = best_h2h_prices(game.get("bookmakers", []), home, away)

        if home_odds is not None or away_odds is not None:
            priced += 1

        odds_rows.append({
            "date": date,
            "home": home,
            "away": away,
            "home_odds": home_odds,
            "away_odds": away_odds,
            "captured_at_utc": captured_at,
        })

        if not event_id:
            continue

        try:
            event_try_rows = fetch_event_try_scorers(event_id, home, away)
        except Exception as e:
            print(f"[warn] try scorer fetch failed for {home} v {away}: {e}")
            event_try_rows = []

        if not event_try_rows:
            print(f"[warn] no usable try scorer rows for {home} v {away}")
            continue

        event_df = pd.DataFrame(event_try_rows)

        # Keep shortest odds per player within each team
        event_df = (
            event_df.sort_values(["team", "odds", "player"])
            .drop_duplicates(subset=["team", "player"], keep="first")
        )

        home_df = event_df[event_df["team"] == home].sort_values(["odds", "player"]).head(3).copy()
        away_df = event_df[event_df["team"] == away].sort_values(["odds", "player"]).head(3).copy()

        if home_df.empty or away_df.empty:
            print(f"[warn] incomplete try scorer split for {home} v {away}")
            continue

        home_df["rank"] = range(1, len(home_df) + 1)
        away_df["rank"] = range(1, len(away_df) + 1)

        final_df = pd.concat([home_df, away_df], ignore_index=True)
        final_df["date"] = date
        final_df["home"] = home
        final_df["away"] = away
        final_df["event_id"] = event_id
        final_df["captured_at_utc"] = captured_at

        try_rows.extend(
            final_df[["date", "home", "away", "team", "player", "odds", "rank", "bookmaker", "event_id", "captured_at_utc"]]
            .to_dict("records")
        )
        try_events_with_rows += 1

    odds_df = pd.DataFrame(odds_rows).drop_duplicates(subset=["date", "home", "away"])
    if odds_df.empty:
        odds_df = pd.DataFrame(columns=["date", "home", "away", "home_odds", "away_odds", "captured_at_utc"])
    odds_df.to_csv(ODDS_OUT, index=False)

    try_df = pd.DataFrame(try_rows)
    if try_df.empty:
        try_df = pd.DataFrame(columns=["date", "home", "away", "team", "player", "odds", "rank", "bookmaker", "event_id", "captured_at_utc"])
    else:
        try_df = try_df.drop_duplicates(subset=["date", "home", "away", "team", "player"], keep="first")
        try_df = try_df.sort_values(["date", "home", "away", "team", "rank", "odds", "player"]).reset_index(drop=True)

    try_df.to_csv(TRY_OUT, index=False)

    print(f"odds.csv updated ({len(odds_df)} rows). Priced events: {priced}")
    print(f"try_scorers.csv updated ({len(try_df)} rows). Fixtures with top-3 per team: {try_events_with_rows}")


if __name__ == "__main__":
    main()
