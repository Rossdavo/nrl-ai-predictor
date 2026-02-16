import pandas as pd
import os

BANKROLL = float(os.getenv("BANKROLL", "1000"))   # default bankroll
KELLY_FRACTION = 0.5   # half-Kelly (safe)


def kelly_fraction(prob, odds):
    """
    Returns Kelly bet fraction.
    """
    if odds <= 1:
        return 0.0

    b = odds - 1
    edge = (prob * odds) - 1
    k = edge / b
    return max(0.0, k)


def main():

    if not os.path.exists("predictions.csv") or not os.path.exists("odds.csv"):
        print("Missing predictions or odds file")
        return

    preds = pd.read_csv("predictions.csv")
    odds = pd.read_csv("odds.csv")

    df = preds.merge(odds, on=["date", "home", "away"], how="left")

    bets = []

    for _, r in df.iterrows():

        prob = r["home_win_prob"]
        odds_home = r["home_odds"]

        if pd.isna(odds_home):
            continue

        k = kelly_fraction(prob, odds_home)

        if k <= 0:
            continue

        stake = BANKROLL * k * KELLY_FRACTION

        bets.append({
            "date": r["date"],
            "home": r["home"],
            "away": r["away"],
            "bet_side": "HOME",
            "prob": prob,
            "odds": odds_home,
            "kelly_fraction": k,
            "stake": round(stake,2)
        })

    out = pd.DataFrame(bets)
    out.to_csv("bet_log.csv", index=False)

    print(f"{len(out)} bets generated")
    

if __name__ == "__main__":
    main()
