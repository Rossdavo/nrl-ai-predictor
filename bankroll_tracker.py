import pandas as pd
import os

START_BANKROLL = float(os.getenv("BANKROLL", "200"))

def main():
    if not os.path.exists("performance.csv"):
        print("No performance.csv yet")
        return

    perf = pd.read_csv("performance.csv")

    if perf.empty or "profit" not in perf.columns:
        print("No settled bets yet")
        return

    total_profit = perf["profit"].sum()

    current_bankroll = START_BANKROLL + total_profit
    peak_bankroll = max(START_BANKROLL, perf["bankroll_after"].max() if "bankroll_after" in perf.columns else current_bankroll)

    drawdown = 0
    if peak_bankroll > 0:
        drawdown = (peak_bankroll - current_bankroll) / peak_bankroll

    roi = total_profit / START_BANKROLL if START_BANKROLL > 0 else 0

    out = pd.DataFrame([{
        "start_bankroll": START_BANKROLL,
        "current_bankroll": round(current_bankroll, 2),
        "peak_bankroll": round(peak_bankroll, 2),
        "drawdown_pct": round(drawdown * 100, 2),
        "roi_pct": round(roi * 100, 2),
    }])

    out.to_csv("bankroll_status.csv", index=False)
    print("bankroll_status.csv updated")

if __name__ == "__main__":
    main()
