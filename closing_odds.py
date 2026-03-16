import os
import pandas as pd
from zoneinfo import ZoneInfo

SYDNEY_TZ = ZoneInfo("Australia/Sydney")

OUT_COLS = [
    "date",
    "home",
    "away",
    "home_odds_close",
    "away_odds_close",
    "close_captured_at_utc",
]


def _load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Could not read {path}: {e}")
        return pd.DataFrame()


def _norm(s: str) -> str:
    return " ".join(str(s).strip().upper().split())


def _write_empty(msg: str) -> None:
    pd.DataFrame(columns=OUT_COLS).to_csv("closing_odds.csv", index=False)
    print(msg)


def main():
    odds_hist = _load_csv_safe("odds_history.csv")
    if odds_hist.empty:
        _write_empty("No odds_history.csv yet — cannot build closing odds.")
        return

    preds = _load_csv_safe("predictions_history.csv")
    if preds.empty:
        preds = _load_csv_safe("predictions.csv")

    if preds.empty:
        _write_empty("No predictions file found — cannot build closing odds.")
        return

    for c in ["date", "home", "away"]:
        if c not in odds_hist.columns or c not in preds.columns:
            _write_empty("Missing required columns in odds_history/predictions.")
            return

    ts_col = None
    for cand in ["captured_at_utc", "generated_at", "pulled_at_utc", "timestamp_utc"]:
        if cand in odds_hist.columns:
            ts_col = cand
            break

    if ts_col is None:
        _write_empty("No timestamp column found in odds_history.csv (need captured_at_utc or equivalent).")
        return

    odds_hist = odds_hist.copy()
    preds = preds.copy()

    odds_hist["date"] = pd.to_datetime(odds_hist["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    preds["date"] = pd.to_datetime(preds["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    odds_hist["home"] = odds_hist["home"].map(_norm)
    odds_hist["away"] = odds_hist["away"].map(_norm)
    preds["home"] = preds["home"].map(_norm)
    preds["away"] = preds["away"].map(_norm)

    odds_hist[ts_col] = pd.to_datetime(odds_hist[ts_col], errors="coerce", utc=True)
    odds_hist["home_odds"] = pd.to_numeric(odds_hist.get("home_odds"), errors="coerce")
    odds_hist["away_odds"] = pd.to_numeric(odds_hist.get("away_odds"), errors="coerce")
    odds_hist = odds_hist.dropna(subset=[ts_col, "date", "home", "away"]).copy()

    if "kickoff_local" not in preds.columns:
        _write_empty("Predictions missing kickoff_local — cannot build closing odds.")
        return

    # Prefer latest saved prediction row per game if generated_at exists
    if "generated_at" in preds.columns:
        preds["generated_at"] = pd.to_datetime(preds["generated_at"], errors="coerce", utc=True)
        preds = preds.sort_values(["date", "home", "away", "generated_at"])
        preds = preds.drop_duplicates(subset=["date", "home", "away"], keep="last").copy()
    else:
        preds = preds.drop_duplicates(subset=["date", "home", "away"], keep="last").copy()

    # kickoff_local may be either "19:00" or a fuller datetime string
    kickoff_raw = preds["kickoff_local"].astype(str).str.strip()

    # Try date + time first
    preds["kickoff_dt"] = pd.to_datetime(
        preds["date"].astype(str) + " " + kickoff_raw,
        errors="coerce"
    )

    # Fallback: kickoff_local itself may already be a full datetime
    missing_mask = preds["kickoff_dt"].isna()
    if missing_mask.any():
        preds.loc[missing_mask, "kickoff_dt"] = pd.to_datetime(
            kickoff_raw[missing_mask],
            errors="coerce"
        )

    preds = preds.dropna(subset=["kickoff_dt"]).copy()

    preds["kickoff_dt_utc"] = (
        preds["kickoff_dt"]
        .dt.tz_localize(SYDNEY_TZ, nonexistent="shift_forward", ambiguous="NaT")
        .dt.tz_convert("UTC")
    )
    preds = preds.dropna(subset=["kickoff_dt_utc"]).copy()

    games = preds[["date", "home", "away", "kickoff_dt_utc"]].drop_duplicates()

    out_rows = []

    for _, g in games.iterrows():
        date = g["date"]
        home = g["home"]
        away = g["away"]
        ko_utc = g["kickoff_dt_utc"]

        subset = odds_hist.loc[
            (odds_hist["date"] == date) &
            (odds_hist["home"] == home) &
            (odds_hist["away"] == away) &
            (odds_hist[ts_col] <= ko_utc)
        ].copy()

        if subset.empty:
            continue

        subset = subset.sort_values(ts_col)
        last = subset.iloc[-1]

        cap = last[ts_col]

        out_rows.append({
            "date": date,
            "home": home,
            "away": away,
            "home_odds_close": last.get("home_odds"),
            "away_odds_close": last.get("away_odds"),
            "close_captured_at_utc": cap.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(cap) else "",
        })

    out = pd.DataFrame(out_rows)
    if out.empty:
        out = pd.DataFrame(columns=OUT_COLS)
    else:
        out = out[OUT_COLS].sort_values(["date", "home", "away"]).reset_index(drop=True)

    out.to_csv("closing_odds.csv", index=False)
    print(f"closing_odds.csv updated ({len(out)} rows)")


if __name__ == "__main__":
    main()
