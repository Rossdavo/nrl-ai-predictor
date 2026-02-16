import os
import requests
import pandas as pd

API_KEY = os.getenv("ODDS_API_KEY")

SPORT = "rugbyleague_nrl"
REGIONS = "au"
MARKETS = "h2h"
BOOKMAKERS = "sportsbet,tab,pointsbetau"

URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"


def best_h2h_prices(bookmakers, home, away):
    """
    Returns (home_price, away_price) using BEST price found across bookmakers.
    """
    home_best = None
    away_best = None

    for book in bookmakers or []:
        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("name")
                price = outcome.get("price")
                if name == home and price is not None:
                    home_best = price if home_best is None else max(home_best, price)
                if name == away and price is not None:
                    away_best = price if away_best is None else max(away_best, price)

    return home_best, away_best


def main():
    if not API_KEY:
        print("No ODDS_API_KEY provided")
        # still write an empty odds.csv so downstream doesn't crash
        pd.DataFrame(columns=["date", "home", "away", "home_odds", "away_odds"]).to_csv("odds.csv", index=False)
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
    for game in data:
        home = game.get("home_team")
        away = game.get("away_team")
        commence = game.get("commence_time")

        if not home or not away or not commence:
            continue

        date = str(commence)[:10]

        home_odds, away_odds = best_h2h_prices(game.get("bookmakers", []), home, away)

        rows.append({
            "date": date,
            "home": home,
            "away": away,
            "home_odds": home_odds,
            "away_odds": away_odds,
        })

    from datetime import datetime, timezone

# ... after you build df ...
df["captured_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
df.to_csv("odds.csv", index=False)

    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "home", "away"])
    df.to_csv("odds.csv", index=False)

    print(f"odds.csv updated ({len(df)} rows)")


if __name__ == "__main__":
    main()
