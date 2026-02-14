import os
import math
import pandas as pd

PRED_HIST = "predictions_history.csv"
ODDS_HIST = "odds_history.csv"
ACC = "accuracy.csv"

OUT = "performance.csv"

def implied_prob(decimal_odds: float) -> float:
    if decimal_odds is None or math.isnan(decimal_odds) or decimal_odds <= 1.0:
        return float("nan")
    return 1.0 / decimal_odds

def main():
    if not (os.path.exists(PRED_HIST) and os.path.exists(ODDS_HIST) and os.path.exists(ACC)):
        pd.DataFrame().to_csv(OUT, index=False)
        print("Missing history/accuracy files; nothing to report yet.")
        return

    pred = pd.read_csv(PRED_HIST)
    odds = pd.read_csv(ODDS_HIST)
    acc = pd.read_csv(ACC)

    if pred.empty or odds.empty or acc.empty:
        pd.DataFrame().to_csv(OUT, index=False)
        print("Empty inputs; nothing to report yet.")
        return

    # normalize keys
    for df in (pred, odds, acc):
        if "date" in df.columns:
            df["date"] = df["date"].astype(str).str.slice(0, 10).str.strip()
        if "home" in df.columns: df["home"] = df["home"].astype(str).str.strip()
        if "away" in df.columns: df["away"] = df["away"].astype(str).str.strip()

    # For each match, pick:
    # - earliest recorded odds (our "bet" time proxy)
    # - latest recorded odds before results exist (closing proxy)
    odds["run_utc"] = pd.to_datetime(odds.get("run_utc", ""), errors="coerce")
    odds = odds.dropna(subset=["run_utc"])

    # earliest odds per match
    o_open = odds.sort_values("run_utc").groupby(["date","home","away"], as_index=False).first()
    o_close = odds.sort_values("run_utc").groupby(["date","home","away"], as_index=False).last()

    # scored matches + outcome
    acc_key = acc[["date","home","away","winner_correct","actual_winner"]].copy()

    # join everything
    j = pred.sort_values("run_utc").groupby(["date","home","away"], as_index=False).last()
    j = j.merge(o_open[["date","home","away","home_odds","away_odds"]], on=["date","home","away"], how="left", suffixes=("", "_open"))
    j = j.merge(o_close[["date","home","away","home_odds","away_odds"]], on=["date","home","away"], how="left", suffixes=("", "_close"))
    j = j.merge(acc_key, on=["date","home","away"], how="left")

    # derive value bet side using your existing value_flag
    j["value_side"] = ""
    if "value_flag" in j.columns:
        vf = j["value_flag"].fillna("").astype(str)
        j.loc[vf.str.contains("HOME VALUE", na=False), "value_side"] = "HOME"
        j.loc[vf.str.contains("AWAY VALUE", na=False), "value_side"] = "AWAY"

    # Use OPEN odds as our taken price
    j["taken_odds"] = float("nan")
    j.loc[j["value_side"]=="HOME", "taken_odds"] = pd.to_numeric(j["home_odds"], errors="coerce")
    j.loc[j["value_side"]=="AWAY", "taken_odds"] = pd.to_numeric(j["away_odds"], errors="coerce")

    # Closing odds proxy (latest recorded)
    j["close_odds"] = float("nan")
    j.loc[j["value_side"]=="HOME", "close_odds"] = pd.to_numeric(j["home_odds_close"], errors="coerce")
    j.loc[j["value_side"]=="AWAY", "close_odds"] = pd.to_numeric(j["away_odds_close"], errors="coerce")

    # CLV as implied probability improvement (positive = good)
    j["taken_imp"] = j["taken_odds"].apply(lambda x: implied_prob(float(x)) if pd.notna(x) else float("nan"))
    j["close_imp"] = j["close_odds"].apply(lambda x: implied_prob(float(x)) if pd.notna(x) else float("nan"))
    j["clv"] = j["close_imp"] - j["taken_imp"]  # >0 means market moved toward your side

    # ROI on 1 unit stakes (only when result is known and value_side chosen)
    j["staked"] = ((j["value_side"]!="") & j["taken_odds"].notna()).astype(int)
    # determine if bet won
    j["bet_won"] = 0
    j.loc[(j["value_side"]=="HOME") & (j["actual_winner"]==j["home"]), "bet_won"] = 1
    j.loc[(j["value_side"]=="AWAY") & (j["actual_winner"]==j["away"]), "bet_won"] = 1

    j["profit"] = 0.0
    j.loc[(j["staked"]==1) & (j["bet_won"]==1), "profit"] = j["taken_odds"] - 1.0
    j.loc[(j["staked"]==1) & (j["bet_won"]==0) & (j["actual_winner"].notna()), "profit"] = -1.0

    # summary row
    bets = j[(j["staked"]==1) & (j["actual_winner"].notna())].copy()
    total_bets = len(bets)
    roi = (bets["profit"].sum() / total_bets) if total_bets else float("nan")
    avg_clv = bets["clv"].mean() if total_bets else float("nan")

    summary = pd.DataFrame([{
        "bets_scored": total_bets,
        "total_profit_units": round(float(bets["profit"].sum()), 3) if total_bets else 0.0,
        "roi_per_bet": round(float(roi), 4) if roi == roi else "",
        "avg_clv": round(float(avg_clv), 4) if avg_clv == avg_clv else "",
    }])

    summary.to_csv(OUT, index=False)
    print("Wrote performance.csv")

if __name__ == "__main__":
    main()
