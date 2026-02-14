import os
import pandas as pd
import numpy as np

BET_LOG_PATH = "bet_log.csv"
RESULTS_PATH = "results_cache.csv"


def _norm(s: str) -> str:
    return str(s).strip()


def main():
    if not (os.path.exists(BET_LOG_PATH) and os.path.exists(RESULTS_PATH)):
        print("Missing bet_log.csv or results_cache.csv; nothing to settle.")
        return

    bets = pd.read_csv(BET_LOG_PATH)
    res = pd.read_csv(RESULTS_PATH)

    if bets.empty or res.empty:
        print("Empty bets or results; nothing to settle.")
        return

    # Ensure columns exist in bet log
    for col in ["settled", "actual_winner", "profit_units"]:
        if col not in bets.columns:
            bets[col] = ""

    # Normalize keys
    bets["date"] = bets["date"].astype(str).str.slice(0, 10).str.strip()
    bets["home"] = bets["home"].map(_norm)
    bets["away"] = bets["away"].map(_norm)

    res["date"] = res["date"].astype(str).str.slice(0, 10).str.strip()
    res["home"] = res["home"].map(_norm)
    res["away"] = res["away"].map(_norm)
    res["home_pts"] = pd.to_numeric(res["home_pts"], errors="coerce")
    res["away_pts"] = pd.to_numeric(res["away_pts"], errors="coerce")
    res = res.dropna(subset=["home_pts", "away_pts"])

    # Compute winner
    res["actual_winner"] = np.where(res["home_pts"] > res["away_pts"], res["home"], res["away"])

    # Join results onto bets
    j = bets.merge(res[["date", "home", "away", "actual_winner"]], on=["date", "home", "away"], how="left", suffixes=("", "_r"))

    # Only settle bets with a result
    has_result = j["actual_winner_r"].notna()

    # If already settled, don’t change
    already = (j["settled"].astype(str).str.upper() == "Y")
    to_settle = has_result & (~already)

    if to_settle.sum() == 0:
        print("No new bets to settle.")
        return

    # Win if side matches winner
    win = (
        ((j["value_side"] == "HOME") & (j["actual_winner_r"] == j["home"])) |
        ((j["value_side"] == "AWAY") & (j["actual_winner_r"] == j["away"]))
    )

    j.loc[to_settle, "actual_winner"] = j.loc[to_settle, "actual_winner_r"]

    # Profit in units: win = (odds-1)*stake ; loss = -stake
    j["taken_odds"] = pd.to_numeric(j["taken_odds"], errors="coerce")
    j["stake_units"] = pd.to_numeric(j["stake_units"], errors="coerce").fillna(1.0)

    profit = np.where(win, (j["taken_odds"] - 1.0) * j["stake_units"], -1.0 * j["stake_units"])
    j.loc[to_settle, "profit_units"] = profit[to_settle]
    j.loc[to_settle, "settled"] = "Y"

    # Drop helper column, write back
    j = j.drop(columns=["actual_winner_r"], errors="ignore")
    j.to_csv(BET_LOG_PATH, index=False)
    print(f"Settled {int(to_settle.sum())} bets -> {BET_LOG_PATH}")


if __name__ == "__main__":
    main()
