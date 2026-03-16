import os
import pandas as pd

PRED_HISTORY = "predictions_history.csv"
RESULTS = "results_cache.csv"

BET_HISTORY_OUT = "bet_history.csv"
BET_SUMMARY_OUT = "bet_summary.csv"

START_BANKROLL = 1000


def norm(x):
    return str(x).strip().upper()


def load_predictions():
    df = pd.read_csv(PRED_HISTORY)

    df["home"] = df["home"].map(norm)
    df["away"] = df["away"].map(norm)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["stake"] = pd.to_numeric(df["stake"], errors="coerce").fillna(0)
    df["stake_dollars"] = pd.to_numeric(df["stake_dollars"], errors="coerce").fillna(0)

    df["home_odds"] = pd.to_numeric(df["home_odds"], errors="coerce")
    df["away_odds"] = pd.to_numeric(df["away_odds"], errors="coerce")

    return df


def load_results():
    df = pd.read_csv(RESULTS)

    df["home"] = df["home"].map(norm)
    df["away"] = df["away"].map(norm)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")

    return df


def settle():

    pred = load_predictions()
    res = load_results()

    merged = pred.merge(
        res,
        on=["date", "home", "away"],
        how="left"
    )

    bets = merged[merged["stake"] > 0].copy()

    if bets.empty:
        print("No bets found")
        return

    def actual_winner(r):

        if pd.isna(r.home_pts):
            return "PENDING"

        if r.home_pts > r.away_pts:
            return "HOME"

        if r.away_pts > r.home_pts:
            return "AWAY"

        return "DRAW"


    bets["actual"] = bets.apply(actual_winner, axis=1)

    def bet_profit(r):

        if r.actual == "PENDING":
            return 0

        if r.actual == "DRAW":
            return -r.stake

        if r.pick == r.actual:

            odds = r.home_odds if r.pick == "HOME" else r.away_odds

            return r.stake * (odds - 1)

        return -r.stake


    bets["profit_units"] = bets.apply(bet_profit, axis=1)

    bankroll = START_BANKROLL
    bankrolls = []

    for p in bets["profit_units"]:
        bankroll += p
        bankrolls.append(bankroll)

    bets["bankroll"] = bankrolls

    bets.to_csv(BET_HISTORY_OUT, index=False)

    summary = {
        "bets": len(bets),
        "wins": (bets["profit_units"] > 0).sum(),
        "losses": (bets["profit_units"] < 0).sum(),
        "units_profit": bets["profit_units"].sum(),
        "roi": bets["profit_units"].sum() / bets["stake"].sum(),
        "closing_bankroll": bankroll
    }

    pd.DataFrame([summary]).to_csv(BET_SUMMARY_OUT, index=False)

    print("Bets:", summary["bets"])
    print("Profit Units:", round(summary["units_profit"], 2))
    print("ROI:", round(summary["roi"] * 100, 2), "%")
    print("Bankroll:", round(bankroll, 2))


if __name__ == "__main__":
    settle()
