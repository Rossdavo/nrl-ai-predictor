import pandas as pd
import os

def main():
    # Required inputs
    if not os.path.exists("bet_log.csv") or not os.path.exists("results_cache.csv"):
        # Write empty file WITH headers so other scripts never fail
        pd.DataFrame(columns=[
            "date",
            "home",
            "away",
            "bet_side",
            "odds",
            "stake",
            "profit"
        ]).to_csv("performance.csv", index=False)
        print("performance.csv created (no settled bets yet)")
        return

    bets = pd.read_csv("bet_log.csv")
    results = pd.read_csv("results_cache.csv")

    if bets.empty or results.empty:
        pd.DataFrame(columns=[
            "date",
            "home",
            "away",
            "bet_side",
            "odds",
            "stake",
            "profit"
        ]).to_csv("performance.csv", index=False)
        print("performance.csv created (no settled bets yet)")
        return

    merged = bets.merge(results, on=["date", "home", "away"], how="inner")

    rows = []

    for _, r in merged.iterrows():
        winner = r.get("winner")
        bet_side = r.get("bet_side")
        stake = float(r.get("stake", 0))
        odds = float(r.get("odds", 0))

        if winner == bet_side:
            profit = stake * (odds - 1)
        else:
            profit = -stake

        rows.append({
            "date": r["date"],
            "home": r["home"],
            "away": r["away"],
            "bet_side": bet_side,
            "odds": odds,
            "stake": stake,
            "profit": profit,
        })

    out = pd.DataFrame(rows)

    if out.empty:
        out = pd.DataFrame(columns=[
            "date",
            "home",
            "away",
            "bet_side",
            "odds",
            "stake",
            "profit"
        ])

    out.to_csv("performance.csv", index=False)
    print(f"performance.csv updated ({len(out)} settled bets)")

if __name__ == "__main__":
    main()
