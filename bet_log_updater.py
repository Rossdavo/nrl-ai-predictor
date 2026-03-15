import os
import pandas as pd

PREDICTIONS_PATH = "predictions.csv"
BET_LOG_PATH = "bet_log.csv"


def main():
    if not os.path.exists(PREDICTIONS_PATH):
        pd.DataFrame(columns=[
            "date", "home", "away", "pick", "selection",
            "odds", "stake_dollars", "edge", "confidence"
        ]).to_csv(BET_LOG_PATH, index=False)
        print("bet_log.csv created (no predictions.csv found)")
        return

    try:
        df = pd.read_csv(PREDICTIONS_PATH)
    except Exception as e:
        print(f"[warn] Could not read {PREDICTIONS_PATH}: {e}")
        pd.DataFrame(columns=[
            "date", "home", "away", "pick", "selection",
            "odds", "stake_dollars", "edge", "confidence"
        ]).to_csv(BET_LOG_PATH, index=False)
        print("bet_log.csv created empty due to read error")
        return

    if df.empty:
        pd.DataFrame(columns=[
            "date", "home", "away", "pick", "selection",
            "odds", "stake_dollars", "edge", "confidence"
        ]).to_csv(BET_LOG_PATH, index=False)
        print("bet_log.csv created (predictions empty)")
        return

    needed = {"date", "home", "away", "pick", "recommended_bet"}
    missing = needed - set(df.columns)
    if missing:
        print(f"[warn] predictions.csv missing columns: {sorted(missing)}")
        pd.DataFrame(columns=[
            "date", "home", "away", "pick", "selection",
            "odds", "stake_dollars", "edge", "confidence"
        ]).to_csv(BET_LOG_PATH, index=False)
        print("bet_log.csv created empty due to missing columns")
        return

    work = df.copy()

    # Keep only actual bets
    work["recommended_bet"] = work["recommended_bet"].astype(str).str.strip()
    work = work[work["recommended_bet"].ne("")]
    work = work[work["recommended_bet"].str.lower().ne("no bet")]

    if work.empty:
        pd.DataFrame(columns=[
            "date", "home", "away", "pick", "selection",
            "odds", "stake_dollars", "edge", "confidence"
        ]).to_csv(BET_LOG_PATH, index=False)
        print("bet_log.csv updated (no bets for this round)")
        return

    # Build selection from pick
    def selection_from_pick(row):
        pick = str(row.get("pick", "")).strip().upper()
        if pick == "HOME":
            return str(row.get("home", "")).strip()
        if pick == "AWAY":
            return str(row.get("away", "")).strip()
        return ""

    work["selection"] = work.apply(selection_from_pick, axis=1)

    # Pick correct odds side
    def odds_from_pick(row):
        pick = str(row.get("pick", "")).strip().upper()
        if pick == "HOME":
            return row.get("home_odds", pd.NA)
        if pick == "AWAY":
            return row.get("away_odds", pd.NA)
        return pd.NA

    work["odds"] = work.apply(odds_from_pick, axis=1)

    out_cols = [
        "date", "home", "away", "pick", "selection",
        "odds", "stake_dollars", "edge", "confidence"
    ]

    for col in out_cols:
        if col not in work.columns:
            work[col] = pd.NA

    out = work[out_cols].copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["home"] = out["home"].astype(str).str.strip()
    out["away"] = out["away"].astype(str).str.strip()
    out["pick"] = out["pick"].astype(str).str.strip().str.upper()
    out["selection"] = out["selection"].astype(str).str.strip()
    out["odds"] = pd.to_numeric(out["odds"], errors="coerce")
    out["stake_dollars"] = pd.to_numeric(out["stake_dollars"], errors="coerce")
    out["edge"] = pd.to_numeric(out["edge"], errors="coerce")
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")

    out = out.dropna(subset=["date", "home", "away", "pick", "selection", "odds", "stake_dollars"])
    out = out.sort_values(["date", "home", "away"]).reset_index(drop=True)

    out.to_csv(BET_LOG_PATH, index=False)
    print(f"bet_log.csv updated ({len(out)} bets)")


if __name__ == "__main__":
    main()
