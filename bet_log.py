import os
import pandas as pd
from datetime import datetime, timezone

PRED_PATH = "predictions.csv"
RESULTS_PATH = "results_cache.csv"     # created by predict.py (your results fetch)
BET_LOG_PATH = "bet_log.csv"


def _norm(s: str) -> str:
    return str(s).strip()


def _value_side(value_flag: str) -> str:
    v = (value_flag or "").strip().upper()
    if "HOME VALUE" in v:
        return "HOME"
    if "AWAY VALUE" in v:
        return "AWAY"
    return ""


def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def main():
    if not os.path.exists(PRED_PATH):
        print("No predictions.csv found, skipping bet log.")
        return

    pred = pd.read_csv(PRED_PATH)

    required = {"date", "home", "away", "value_flag", "home_odds", "away_odds"}
    if not required.issubset(set(pred.columns)):
        print("predictions.csv missing required columns for bet logging.")
        return

    pred["date"] = pred["date"].astype(str).str.slice(0, 10).str.strip()
    pred["home"] = pred["home"].map(_norm)
    pred["away"] = pred["away"].map(_norm)
    pred["value_flag"] = pred["value_flag"].fillna("").astype(str)

    pred["value_side"] = pred["value_flag"].map(_value_side)
    pred = pred[pred["value_side"] != ""].copy()

    if pred.empty:
        print("No value bets flagged this run.")
        return

    # Choose odds based on side
    pred["taken_odds"] = None
    pred.loc[pred["value_side"] == "HOME", "taken_odds"] = pd.to_numeric(pred["home_odds"], errors="coerce")
    pred.loc[pred["value_side"] == "AWAY", "taken_odds"] = pd.to_numeric(pred["away_odds"], errors="coerce")

    pred = pred.dropna(subset=["taken_odds"])
    if pred.empty:
        print("Value bets had no odds (NaN).")
        return

    # Flat staking
    pred["stake_units"] = 1.0
    pred["logged_utc"] = _now_utc_str()

    # Create bet_id that’s stable across re-runs
    pred["bet_id"] = (
        pred["date"].astype(str) + "|" +
        pred["home"].astype(str) + "|" +
        pred["away"].astype(str) + "|" +
        pred["value_side"].astype(str)
    )

    new_bets = pred[[
        "bet_id", "logged_utc", "date", "home", "away",
        "value_side", "taken_odds", "stake_units", "value_flag"
    ]].copy()

    # Load existing log and append only new bet_id
    if os.path.exists(BET_LOG_PATH):
        old = pd.read_csv(BET_LOG_PATH)
        if "bet_id" in old.columns:
            existing = set(old["bet_id"].astype(str).tolist())
        else:
            existing = set()
        new_bets = new_bets[~new_bets["bet_id"].astype(str).isin(existing)].copy()
        if new_bets.empty:
            print("No new bets to add (already logged).")
            return
        out = pd.concat([old, new_bets], ignore_index=True)
    else:
        out = new_bets

    out.to_csv(BET_LOG_PATH, index=False)
    print(f"Logged {len(new_bets)} new bets -> {BET_LOG_PATH}")


if __name__ == "__main__":
    main()
