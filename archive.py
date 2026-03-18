import os
import csv
import pandas as pd
from datetime import datetime, timezone

PRED = "predictions.csv"
ODDS = "odds.csv"

PRED_HIST = "predictions_history.csv"
ODDS_HIST = "odds_history.csv"

PRED_HISTORY_COLS = [
    "run_id",
    "run_utc",
    "mode",
    "rating_mode",
    "date",
    "kickoff_local",
    "venue",
    "home",
    "away",
    "model_home_win_prob",
    "market_home_win_prob",
    "final_home_win_prob",
    "exp_margin_home",
    "exp_total",
    "confidence",
    "home_odds",
    "away_odds",
    "favourite_team",
    "underdog_team",
    "auto_upset_score",
    "auto_upset_reasons",
    "manual_upset_team",
    "manual_upset_score",
    "manual_upset_notes",
    "upset_team",
    "final_upset_score",
    "upset_flag",
    "fragile_favourite",
    "required_edge",
    "value_flag",
    "pick",
    "edge",
    "stake",
    "stake_units",
    "stake_dollars",
    "recommended_bet",
    "generated_at",
]

ODDS_HISTORY_COLS = [
    "run_id",
    "run_utc",
    "date",
    "home",
    "away",
    "home_odds",
    "away_odds",
]

LEGACY_ALIASES = {
    "home_win_prob": "model_home_win_prob",
}

NUMERIC_COLS = [
    "model_home_win_prob",
    "market_home_win_prob",
    "final_home_win_prob",
    "exp_margin_home",
    "exp_total",
    "confidence",
    "home_odds",
    "away_odds",
    "auto_upset_score",
    "manual_upset_score",
    "final_upset_score",
    "upset_flag",
    "fragile_favourite",
    "required_edge",
    "edge",
    "stake",
    "stake_units",
    "stake_dollars",
]

TEXT_COLS = [
    "run_id",
    "run_utc",
    "mode",
    "rating_mode",
    "date",
    "kickoff_local",
    "venue",
    "home",
    "away",
    "favourite_team",
    "underdog_team",
    "auto_upset_reasons",
    "manual_upset_team",
    "manual_upset_notes",
    "upset_team",
    "value_flag",
    "pick",
    "recommended_bet",
    "generated_at",
]


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"[warn] Could not read {path}: {e}")
        return pd.DataFrame()


def _normalise_text(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).replace({"nan": "", "NaT": ""}).str.strip()
    return df


def _normalise_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _prepare_prediction_history(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=PRED_HISTORY_COLS)

    out = df.copy()

    # map legacy names into new schema if needed
    for old_name, new_name in LEGACY_ALIASES.items():
        if old_name in out.columns and new_name not in out.columns:
            out = out.rename(columns={old_name: new_name})

    # ensure all expected cols exist
    for col in PRED_HISTORY_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    # clean types
    out = _normalise_text(out, TEXT_COLS)
    out = _normalise_numeric(out, NUMERIC_COLS)

    # rebuild stake_units if missing but stake exists or dollars exists
    if "stake_units" in out.columns:
        missing_units = out["stake_units"].isna()
        if "stake" in out.columns:
            out.loc[missing_units, "stake_units"] = out.loc[missing_units, "stake"]
        missing_units = out["stake_units"].isna()
        if "stake_dollars" in out.columns:
            out.loc[missing_units, "stake_units"] = (out.loc[missing_units, "stake_dollars"] / 10.0).round(2)

    # normalise dates
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["generated_at"] = out["generated_at"].replace("", pd.NA)
    missing_generated = out["generated_at"].isna()
    out.loc[missing_generated, "generated_at"] = _utc_now_str() + " UTC"

    # insert run metadata
    run_utc = _utc_now_str()
    out["run_id"] = str(run_id).strip()
    out["run_utc"] = run_utc

    # keep only usable prediction rows
    out = out.dropna(subset=["date", "home", "away"]).copy()

    # preserve exact output ordering
    out = out[PRED_HISTORY_COLS].copy()
    return out


def _prepare_odds_history(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=ODDS_HISTORY_COLS)

    out = df.copy()

    for col in ODDS_HISTORY_COLS:
        if col not in out.columns:
            out[col] = pd.NA

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["home"] = out["home"].fillna("").astype(str).str.strip()
    out["away"] = out["away"].fillna("").astype(str).str.strip()
    out["home_odds"] = pd.to_numeric(out["home_odds"], errors="coerce")
    out["away_odds"] = pd.to_numeric(out["away_odds"], errors="coerce")

    run_utc = _utc_now_str()
    out["run_id"] = str(run_id).strip()
    out["run_utc"] = run_utc

    out = out.dropna(subset=["date", "home", "away"]).copy()
    out = out[ODDS_HISTORY_COLS].copy()
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
