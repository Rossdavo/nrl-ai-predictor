import os
import pandas as pd

def _load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def main():
    bets = _load_csv_safe("bet_log.csv")
    closing = _load_csv_safe("closing_odds.csv")

    if bets.empty:
        print("No bet_log.csv or no bets yet — skipping CLV.")
        pd.DataFrame(columns=[
            "date","home","away","side","odds_taken","odds_close","clv_implied_prob","clv_pct"
        ]).to_csv("clv_results.csv", index=False)
        return

    if closing.empty:
        print("No closing odds yet — skipping CLV.")
        pd.DataFrame(columns=[
            "date","home","away","side","odds_taken","odds_close","clv_implied_prob","clv_pct"
        ]).to_csv("clv_results.csv", index=False)
        return

    # Expected bet_log columns: date, home, away, side, odds_taken
    # side should be HOME or AWAY
    need = {"date","home","away","side","odds_taken"}
    if not need.issubset(set(bets.columns)):
        print("bet_log.csv missing required columns for CLV.")
        pd.DataFrame(columns=[
            "date","home","away","side","odds_taken","odds_close","clv_implied_prob","clv_pct"
        ]).to_csv("clv_results.csv", index=False)
        return

    merged = bets.merge(closing, on=["date","home","away"], how="left")

    def pick_close(row):
        side = str(row.get("side","")).upper().strip()
        if side == "HOME":
            return row.get("home_odds_close")
        if side == "AWAY":
            return row.get("away_odds_close")
        return None

    merged["odds_close"] = merged.apply(pick_close, axis=1)

    merged["odds_taken"] = pd.to_numeric(merged["odds_taken"], errors="coerce")
    merged["odds_close"] = pd.to_numeric(merged["odds_close"], errors="coerce")

    # CLV implied probability = (1/close) - (1/taken)
    merged["clv_implied_prob"] = (1.0 / merged["odds_close"]) - (1.0 / merged["odds_taken"])
    merged["clv_pct"] = merged["clv_implied_prob"]

    out = merged[["date","home","away","side","odds_taken","odds_close","clv_implied_prob","clv_pct"]].copy()
    out.to_csv("clv_results.csv", index=False)
    print(f"clv_results.csv updated ({len(out)} rows)")

if __name__ == "__main__":
    main()
