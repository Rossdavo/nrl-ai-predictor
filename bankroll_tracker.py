import os
import pandas as pd
from pandas.errors import EmptyDataError

START_BANKROLL = float(os.getenv("BANKROLL", "200"))

def _write_default():
    out = {
        "start_bankroll": START_BANKROLL,
        "current_bankroll": START_BANKROLL,
        "peak_bankroll": START_BANKROLL,
        "drawdown_pct": 0.0,
        "roi_pct": 0.0,
        "settled_bets": 0,
        "total_profit": 0.0,
    }
    pd.DataFrame([out]).to_csv("bankroll_status.csv", index=False)
    print("bankroll_status.csv updated (no settled bets yet)")

def main():
    # If no file at all, write default and exit
    if not os.path.exists("performance.csv"):
        _write_default()
        return

    # If file exists but is empty (0 bytes), write default and exit
    if os.path.getsize("performance.csv") == 0:
        _write_default()
        return

    # Try reading performance.csv safely
    try:
        perf = pd.read_csv("performance.csv")
    except EmptyDataError:
        _write_default()
        return
    except Exception:
        _write_default()
        return

    # If no usable profit column, write default
    if perf.empty or "profit" not in perf.columns:
        _write_default()
        return

    profits = pd.to_numeric(perf["profit"], errors="coerce").dropna()
    if profits.empty:
        _write_default()
        return

    cum_profit = profits.cumsum()
    bankroll_series = START_BANKROLL + cum_profit

    current = float(bankroll_series.iloc[-1])
    peak = float(bankroll_series.max())
    drawdown = 0.0 if peak <= 0 else (peak - current) / peak
    total_profit = float(cum_profit.iloc[-1])
    roi = 0.0 if START_BANKROLL <= 0 else total_profit / START_BANKROLL

    out = {
        "start_bankroll": round(START_BANKROLL, 2),
        "current_bankroll": round(current, 2),
        "peak_bankroll": round(peak, 2),
        "drawdown_pct": round(drawdown * 100, 2),
        "roi_pct": round(roi * 100, 2),
        "settled_bets": int(len(profits)),
        "total_profit": round(total_profit, 2),
    }

    pd.DataFrame([out]).to_csv("bankroll_status.csv", index=False)
    print("bankroll_status.csv updated")

if __name__ == "__main__":
    main()
