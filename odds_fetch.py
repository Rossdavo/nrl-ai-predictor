import os
import requests
import pandas as pd
from datetime import datetime, timezone

API_KEY = os.getenv("ODDS_API_KEY")

SPORT = "rugbyleague_nrl"
REGIONS = "au"
MARKETS = "h2h"
BOOKMAKERS = "sportsbet,tab,pointsbetau"

URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"

# Map API naming quirks -> your naming (and vice versa)
TEAM_ALIASES = {
    "Wests Tigers": ["Wests Tigers", "Wests Tigers NRL", "Wests Tigers (NRL)"],
    "Sea Eagles": ["Manly", "Manly Sea Eagles", "Manly-Warringah Sea Eagles", "Sea Eagles"],
    "Rabbitohs": ["Souths", "South Sydney", "South Sydney Rabbitohs", "Rabbitohs"],
    "Roosters": ["Sydney Roosters", "Roosters"],
    "Bulldogs": ["Canterbury", "Canterbury Bulldogs", "Bulldogs"],
    "Eels": ["Parramatta", "Parramatta Eels", "Eels"],
    "Knights": ["Newcastle", "Newcastle Knights", "Knights"],
    "Dragons": ["St George", "St George Illawarra", "St George Illawarra Dragons", "Dragons"],
    "Sharks": ["Cronulla", "Cronulla Sharks", "Sharks"],
    "Storm": ["Melbourne Storm", "Storm"],
    "Raiders": ["Canberra Raiders", "Raiders"],
    "Warriors": ["NZ Warriors", "New Zealand Warriors", "Warriors"],
    "Panthers": ["Penrith Panthers", "Panthers"],
    "Cowboys": ["North Queensland Cowboys", "Cowboys"],
    "Titans": ["Gold Coast Titans", "Titans"],
    "Dolphins": ["The Dolphins", "Dolphins"],
    "Broncos": ["Brisbane Broncos", "Broncos"],
}

def norm(s: str) -> str:
    return (s or "").strip().lower()

def matches_team(outcome_name: str, team_name: str) -> bool:
    """Return True if outcome_name matches team_name (including aliases)."""
    o = norm(outcome_name)
    t = norm(team_name)
    if o == t:
        return True
    for alias in TEAM_ALIASES.get(team_name, []):
        if norm(alias) == o:
            return True
    return False

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

def main():
    if not API_KEY:
        print("No ODDS_API_KEY provided")
        pd.DataFrame(columns=["date", "home", "away", "home_odds", "away_odds", "captured_at_utc"]).to_csv("odds.csv", index=False)
        return

    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "bookmakers": BOOKMAKERS,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    r = requests.get(URL, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    data = r.json()

    rows = []
    priced = 0

    for game in data:
        home = game.get("home_team")
        away = game.get("away_team")
        commence = game.get("commence_time")

        if not home or not away or not commence:
            continue

        date = str(commence)[:10]
        home_odds, away_odds = best_h2h_prices(game.get("bookmakers", []), home, away)

        if home_odds is not None or away_odds is not None:
            priced += 1

        rows.append({
            "date": date,
            "home": home,
            "away": away,
            "home_odds": home_odds,
            "away_odds": away_odds,
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "home", "away"])
    df["captured_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv("odds.csv", index=False)

    print(f"odds.csv updated ({len(df)} rows). Priced events: {priced}")

if __name__ == "__main__":
    main()
