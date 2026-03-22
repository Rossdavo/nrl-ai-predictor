import os
import pandas as pd

PRED_HISTORY_PATH = "predictions_history.csv"
RESULTS_PATH = "results_cache.csv"

CURRENT_OUT = "predictions.csv"
COMPLETED_OUT = "latest_completed_round.csv"

EXPECTED_COLS = [
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

RESULT_COLS = ["date", "home", "away", "home_pts", "away_pts"]


def empty_csv(path: str, cols: list[str]) -> None:
    pd.DataFrame(columns=cols).to_csv(path, index=False)


def load_prediction_history() -> pd.DataFrame:
    if not os.path.exists(PRED_HISTORY_PATH):
        print(f"[warn] Missing {PRED_HISTORY_PATH}")
        return pd.DataFrame(columns=EXPECTED_COLS)

    try:
        df = pd.read_csv(PRED_HISTORY_PATH, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"[warn] Could not read {PRED_HISTORY_PATH}: {e}")
        return pd.DataFrame(columns=EXPECTED_COLS)

    if df.empty:
        return pd.DataFrame(columns=EXPECTED_COLS)

    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = ""

    df = df[EXPECTED_COLS].copy()
    df["generated_at"] = pd.to_datetime(df["generated_at"], errors="coerce", utc=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["generated_at"]).copy()

    return df


def load_results() -> pd.DataFrame:
    if not os.path.exists(RESULTS_PATH):
        print(f"[warn] Missing {RESULTS_PATH}")
        return pd.DataFrame(columns=RESULT_COLS)

    try:
        df = pd.read_csv(RESULTS_PATH)
    except Exception as e:
        print(f"[warn] Could not read {RESULTS_PATH}: {e}")
        return pd.DataFrame(columns=RESULT_COLS)

    needed = set(RESULT_COLS)
    if not needed.issubset(df.columns):
        print(f"[warn] {RESULTS_PATH} missing required columns")
        return pd.DataFrame(columns=RESULT_COLS)

    df = df[RESULT_COLS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["home"] = df["home"].astype(str).str.strip()
    df["away"] = df["away"].astype(str).str.strip()
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")
    df = df.dropna(subset=["date", "home", "away", "home_pts", "away_pts"]).copy()

    return df


def write_latest_run(df: pd.DataFrame) -> None:
    if df.empty:
        print("[warn] No valid history rows found")
        empty_csv(CURRENT_OUT, EXPECTED_COLS)
        return

    latest_run_id = (
        df.groupby("run_id", dropna=False)["generated_at"]
        .max()
        .reset_index()
        .sort_values("generated_at")
        .iloc[-1]["run_id"]
    )

    out = df[df["run_id"] == latest_run_id].copy()
    out = out.sort_values(["date", "kickoff_local", "home"])
    out.to_csv(CURRENT_OUT, index=False)
    print(f"[info] Wrote {len(out)} rows for latest run_id={latest_run_id}")


def write_latest_completed_round(pred_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    if pred_df.empty or results_df.empty:
        print("[warn] Cannot build completed round file")
        empty_csv(COMPLETED_OUT, EXPECTED_COLS + ["home_pts", "away_pts", "winner", "result_found"])
        return

    merged = pred_df.merge(
        results_df,
        how="left",
        on=["date", "home", "away"]
    )

    merged["result_found"] = merged["home_pts"].notna() & merged["away_pts"].notna()
    merged["winner"] = ""
    merged.loc[merged["result_found"] & (merged["home_pts"] > merged["away_pts"]), "winner"] = merged["home"]
    merged.loc[merged["result_found"] & (merged["away_pts"] > merged["home_pts"]), "winner"] = merged["away"]
    merged.loc[merged["result_found"] & (merged["home_pts"] == merged["away_pts"]), "winner"] = "DRAW"

    # Find latest run_id where all matches in that run have results
    run_summary = (
        merged.groupby("run_id", dropna=False)
        .agg(
            generated_at=("generated_at", "max"),
            games=("home", "count"),
            results_found=("result_found", "sum"),
        )
        .reset_index()
    )

    completed_runs = run_summary[run_summary["games"] == run_summary["results_found"]].copy()

    if completed_runs.empty:
        print("[warn] No fully completed round found yet")
        empty_csv(COMPLETED_OUT, EXPECTED_COLS + ["home_pts", "away_pts", "winner", "result_found"])
        return

    latest_completed_run_id = (
        completed_runs.sort_values("generated_at").iloc[-1]["run_id"]
    )

    out = merged[merged["run_id"] == latest_completed_run_id].copy()
    out = out.sort_values(["date", "kickoff_local", "home"])
    out.to_csv(COMPLETED_OUT, index=False)

    print(f"[info] Wrote {len(out)} rows for latest completed run_id={latest_completed_run_id}")


def main():
    pred_df = load_prediction_history()
    results_df = load_results()

    if pred_df.empty:
        empty_csv(CURRENT_OUT, EXPECTED_COLS)
        empty_csv(COMPLETED_OUT, EXPECTED_COLS + ["home_pts", "away_pts", "winner", "result_found"])
        return

    write_latest_run(pred_df)
    write_latest_completed_round(pred_df, results_df)


if __name__ == "__main__":
    main()
