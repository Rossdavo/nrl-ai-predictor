import os
import requests
import pandas as pd

API_KEY = os.getenv("ODDS_API_KEY")

SPORT = "rugbyleague_nrl"
REGIONS = "au"
MARKETS = "h2h"
BOOKMAKERS = "sportsbet,tab,pointsbetau"

URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"

def main():
    if not API_KEY:
        print("No ODDS_API_KEY provided")
        return

    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "bookmakers": BOOKMAKERS,
        "oddsFormat": "decimal"
    }

    r = requests.get(URL, params=params)
    data = r.json()

    rows = []

    for game in data:
        home = game["home_team"]
        away = [t for t in game["teams"] if t != home][0]
        date = game["commence_time"][:10]

        home_odds = None
        away_odds = None

        for book in game["bookmakers"]:
            for market in book["markets"]:
                for outcome in market["outcomes"]:
                    if outcome["name"] == home:
                        home_odds = outcome["price"]
                    if outcome["name"] == away:
                        away_odds = outcome["price"]

        rows.append({
            "date": date,
            "home": home,
            "away": away,
            "home_odds": home_odds,
            "away_odds": away_odds
        })

    df = pd.DataFrame(rows)
    df.to_csv("odds.csv", index=False)
    print("odds.csv updated")

if __name__ == "__main__":
    main()
