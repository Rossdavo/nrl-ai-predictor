import os
import pandas as pd
from pandas.errors import EmptyDataError

BET_SUMMARY_PATH = "bet_summary.csv"
PERFORMANCE_PATH = "performance.csv"

START_BANKROLL = float(os.getenv("BANKROLL", "1000"))


def _write_default():
    out = {
        "start_bankroll": round(START_BANKROLL, 2),
        "current_bankroll": round(START_BANKROLL, 2),
        "peak_bankroll": round(START_BANKROLL, 2),
        "drawdown_pct": 0.0,
        "roi_pct": 0.0,
        "settled_bets": 0,
        "total_profit": 0.0,
    }
    pd.DataFrame([out]).to_csv("bankroll_status.csv", index=False)
    print("bankroll_status.csv updated (no settled bets yet)")


def _load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    if os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _from_bet_summary() -> bool:
    df = _load_csv_safe(BET_SUMMARY_PATH)
    if df.empty:
        return False

    needed = {"start_bankroll", "closing_bankroll", "bets_settled", "units_profit", "roi"}
    if not needed.issubset(df.columns):
        return False

    row = df.iloc[-1]

    start_bankroll = pd.to_numeric(pd.Series([row["start_bankroll"]]), errors="coerce").iloc[0]
    closing_bankroll = pd.to_numeric(pd.Series([row["closing_bankroll"]]), errors="coerce").iloc[0]
    settled_bets = pd.to_numeric(pd.Series([row["bets_settled"]]), errors="coerce").iloc[0]
    total_profit = pd.to_numeric(pd.Series([row["units_profit"]]), errors="coerce").iloc[0]
    roi = pd.to_numeric(pd.Series([row["roi"]]), errors="coerce").iloc[0]

    if pd.isna(start_bankroll):
        start_bankroll = START_BANKROLL
    if pd.isna(closing_bankroll):
        closing_bankroll = start_bankroll
    if pd.isna(settled_bets):
        settled_bets = 0
    if pd.isna(total_profit):
        total_profit = 0.0
    if pd.isna(roi):
        roi = 0.0

    peak_bankroll = max(float(start_bankroll), float(closing_bankroll))
    drawdown = 0.0 if peak_bankroll <= 0 else (peak_bankroll - float(closing_bankroll)) / peak_bankroll

    out = {
        "start_bankroll": round(float(start_bankroll), 2),
        "current_bankroll": round(float(closing_bankroll), 2),
        "peak_bankroll": round(float(peak_bankroll), 2),
        "drawdown_pct": round(drawdown * 100, 2),
        "roi_pct": round(float(roi) * 100, 2),
        "settled_bets": int(settled_bets),
        "total_profit": round(float(total_profit), 2),
    }

    pd.DataFrame([out]).to_csv("bankroll_status.csv", index=False)
    print("bankroll_status.csv updated (from bet_summary.csv)")
    return True


def _from_performance() -> bool:
    perf = _load_csv_safe(PERFORMANCE_PATH)
    if perf.empty or "profit" not in perf.columns:
        return False

    profits = pd.to_numeric(perf["profit"], errors="coerce").dropna()
    if profits.empty:
        return False

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
    print("bankroll_status.csv updated (from performance.csv)")
    return True


def main():
    if _from_bet_summary():
        return

    if _from_performance():
        return

    _write_default()


if __name__ == "__main__":
    main()
