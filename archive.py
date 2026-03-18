import os
import csv
import pandas as pd
from datetime import datetime, timezone

PRED = "predictions.csv"
ODDS = "odds.csv"

PRED_HIST = "predictions_history.csv"
ODDS_HIST = "odds_history.csv"

PRED_HISTORY_COLS = [
    "date",
    "kickoff_local",
    "home",
    "away",
    "home_win_prob",
    "exp_margin_home",
    "home_odds",
    "away_odds",
    "pick",
    "edge",
    "stake",
    "stake_dollars",
    "recommended_bet",
    "home_top_try",
    "away_top_try",
    "generated_at",
]

ODDS_HISTORY_COLS = [
    "date",
    "home",
    "away",
    "home_odds",
    "away_odds",
]


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Could not read {path}: {e}")
        return pd.DataFrame()


def _prepare_prediction_history(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()

    for col in PRED_HISTORY_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[PRED_HISTORY_COLS].copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["kickoff_local"] = out["kickoff_local"].astype(str).replace("nan", "").str.strip()
    out["home"] = out["home"].astype(str).str.strip()
    out["away"] = out["away"].astype(str).str.strip()
    out["home_win_prob"] = pd.to_numeric(out["home_win_prob"], errors="coerce")
    out["exp_margin_home"] = pd.to_numeric(out["exp_margin_home"], errors="coerce")
    out["home_odds"] = pd.to_numeric(out["home_odds"], errors="coerce")
    out["away_odds"] = pd.to_numeric(out["away_odds"], errors="coerce")
    out["edge"] = pd.to_numeric(out["edge"], errors="coerce")
    out["stake"] = pd.to_numeric(out["stake"], errors="coerce")
    out["stake_dollars"] = pd.to_numeric(out["stake_dollars"], errors="coerce")
    out["pick"] = out["pick"].astype(str).replace("nan", "").str.strip()
    out["recommended_bet"] = out["recommended_bet"].astype(str).replace("nan", "").str.strip()
    out["home_top_try"] = out["home_top_try"].astype(str).replace("nan", "").str.strip()
    out["away_top_try"] = out["away_top_try"].astype(str).replace("nan", "").str.strip()
    out["generated_at"] = out["generated_at"].astype(str).replace("nan", "").str.strip()

    out = out.dropna(subset=["date", "home", "away", "home_win_prob"]).copy()

    run_utc = _utc_now_str()
    out.insert(0, "run_utc", run_utc)
    out.insert(0, "run_id", run_id)

    return out


def _prepare_odds_history(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()

    for col in ODDS_HISTORY_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[ODDS_HISTORY_COLS].copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["home"] = out["home"].astype(str).str.strip()
    out["away"] = out["away"].astype(str).str.strip()
    out["home_odds"] = pd.to_numeric(out["home_odds"], errors="coerce")
    out["away_odds"] = pd.to_numeric(out["away_odds"], errors="coerce")

    out = out.dropna(subset=["date", "home", "away"]).copy()

    run_utc = _utc_now_str()
    out.insert(0, "run_utc", run_utc)
    out.insert(0, "run_id", run_id)

    return out


def _append_history(df: pd.DataFrame, dst: str) -> None:
    if df.empty:
        return

    if os.path.exists(dst):
        df.to_csv(dst, mode="a", header=False, index=False, quoting=csv.QUOTE_ALL)
    else:
        df.to_csv(dst, index=False, quoting=csv.QUOTE_ALL)


def main():
    run_id = os.environ.get("GITHUB_RUN_ID", "")

    pred_df = _load_csv_safe(PRED)
    pred_hist_df = _prepare_prediction_history(pred_df, run_id)
    _append_history(pred_hist_df, PRED_HIST)

    odds_df = _load_csv_safe(ODDS)
    odds_hist_df = _prepare_odds_history(odds_df, run_id)
    _append_history(odds_hist_df, ODDS_HIST)

    print("Archived predictions + odds.")


if __name__ == "__main__":
    main()
