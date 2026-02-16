import pandas as pd
import os

BET_LOG = "bet_log.csv"
CLOSING_ODDS = "closing_odds.csv"
CLV_FILE = "clv_results.csv"

def main():

    if not os.path.exists(BET_LOG) or not os.path.exists(CLOSING_ODDS):
        print("Missing inputs — no CLV calculated")
        return

    bets = pd.read_csv(BET_LOG)
    closing = pd.read_csv(CLOSING_ODDS)

    merged = bets.merge(
        closing,
        on=["date", "home", "away"],
        how="left",
        suffixes=("", "_close")
    )

    merged["clv"] = (
        (1 / merged["home_odds_close"]) -
        (1 / merged["home_odds"])
    )

    merged.to_csv(CLV_FILE, index=False)

    print("CLV updated")

if __name__ == "__main__":
    main()
