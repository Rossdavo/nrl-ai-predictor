import os
import pandas as pd
from zoneinfo import ZoneInfo

SYDNEY_TZ = ZoneInfo("Australia/Sydney")

def _load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def main():
    odds_hist = _load_csv_safe("odds_history.csv")
    if odds_hist.empty:
        print("No odds_history.csv yet — cannot build closing odds.")
        # still write an empty file with headers to avoid future crashes
        pd.DataFrame(columns=[
            "date","home","away","home_odds_close","away_odds_close","close_captured_at_utc"
        ]).to_csv("closing_odds.csv", index=False)
        return

    # We need kickoff times. Prefer predictions_history, else predictions.csv.
    preds = _load_csv_safe("predictions_history.csv")
    if preds.empty:
        preds = _load_csv_safe("predictions.csv")

    if preds.empty:
        print("No predictions file found — cannot build closing odds.")
        pd.DataFrame(columns=[
            "date","home","away","home_odds_close","away_odds_close","close_captured_at_utc"
        ]).to_csv("closing_odds.csv", index=False)
        return

    # Normalize required columns
    for c in ["date", "home", "away"]:
        if c not in odds_hist.columns or c not in preds.columns:
            print("Missing required columns in odds_history/predictions.")
            pd.DataFrame(columns=[
                "date","home","away","home_odds_close","away_odds_close","close_captured_at_utc"
            ]).to_csv("closing_odds.csv", index=False)
            return

    # Must have timestamp column from odds_fetch.py
    ts_col = None
    for cand in ["captured_at_utc", "generated_at", "pulled_at_utc", "timestamp_utc"]:
        if cand in odds_hist.columns:
            ts_col = cand
            break

    if ts_col is None:
        print("No timestamp column found in odds_history.csv (need captured_at_utc).")
        pd.DataFrame(columns=[
            "date","home","away","home_odds_close","away_odds_close","close_captured_at_utc"
        ]).to_csv("closing_odds.csv", index=False)
        return

    # Parse timestamps
    odds_hist[ts_col] = pd.to_datetime(odds_hist[ts_col], errors="coerce", utc=True)
    odds_hist = odds_hist.dropna(subset=[ts_col])

    # Kickoff datetime (Sydney local) -> UTC
    if "kickoff_local" not in preds.columns:
        print("Predictions missing kickoff_local — cannot build closing odds.")
        pd.DataFrame(columns=[
            "date","home","away","home_odds_close","away_odds_close","close_captured_at_utc"
        ]).to_csv("closing_odds.csv", index=False)
        return

    preds["kickoff_dt"] = pd.to_datetime(
        preds["date"].astype(str) + " " + preds["kickoff_local"].astype(str),
        errors="coerce"
    )
    preds = preds.dropna(subset=["kickoff_dt"])
    preds["kickoff_dt_utc"] = preds["kickoff_dt"].dt.tz_localize(SYDNEY_TZ, nonexistent="shift_forward").dt.tz_convert("UTC")

    games = preds[["date", "home", "away", "kickoff_dt_utc"]].drop_duplicates()

    out_rows = []
    # Build closing odds per game: last snapshot before kickoff
    for _, g in games.iterrows():
        date, home, away, ko_utc = g["date"], g["home"], g["away"], g["kickoff_dt_utc"]

        mask = (
            (odds_hist["date"].astype(str) == str(date)) &
            (odds_hist["home"].astype(str) == str(home)) &
            (odds_hist["away"].astype(str) == str(away)) &
            (odds_hist[ts_col] <= ko_utc)
        )
        subset = odds_hist.loc[mask].copy()
        if subset.empty:
            continue

        subset = subset.sort_values(ts_col)
        last = subset.iloc[-1]

        home_close = last.get("home_odds")
        away_close = last.get("away_odds")
        cap = last.get(ts_col)

        out_rows.append({
            "date": date,
            "home": home,
            "away": away,
            "home_odds_close": home_close,
            "away_odds_close": away_close,
            "close_captured_at_utc": cap.strftime("%Y-%m-%d %H:%M:%S") if hasattr(cap, "strftime") else str(cap)
        })

    out = pd.DataFrame(out_rows)
    if out.empty:
        out = pd.DataFrame(columns=[
            "date","home","away","home_odds_close","away_odds_close","close_captured_at_utc"
        ])

    out.to_csv("closing_odds.csv", index=False)
    print(f"closing_odds.csv updated ({len(out)} rows)")

if __name__ == "__main__":
    main()
