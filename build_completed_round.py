import os
import pandas as pd

PRED_HISTORY_PATH = "predictions_history.csv"
RESULTS_PATH = "results_cache.csv"
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
    "predicted_winner",
    "win_probability",
    "confidence_band",
    "bet_grade",
    "model_home_win_prob",
    "market_home_win_prob",
    "final_home_win_prob",
    "final_away_win_prob",
    "exp_margin_home",
    "exp_total",
    "confidence",
    "home_odds",
    "away_odds",
    "predicted_winner_odds",
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


def norm_text(v) -> str:
    return str(v or "").strip()


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
    df["home"] = df["home"].map(norm_text)
    df["away"] = df["away"].map(norm_text)
    df = df.dropna(subset=["generated_at", "date"]).copy()
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
    df["home"] = df["home"].map(norm_text)
    df["away"] = df["away"].map(norm_text)
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")
    df = df.dropna(subset=["date", "home", "away", "home_pts", "away_pts"]).copy()
    df = df.drop_duplicates(subset=["date", "home", "away"], keep="last").copy()
    return df


def main():
    pred_df = load_prediction_history()
    results_df = load_results()

    out_cols = EXPECTED_COLS + ["home_pts", "away_pts", "result_found", "winner"]

    if pred_df.empty or results_df.empty:
        print("[warn] Cannot build completed round file")
        empty_csv(COMPLETED_OUT, out_cols)
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

    run_summary = (
        merged.groupby("run_id", dropna=False)
        .agg(
            generated_at=("generated_at", "max"),
            games=("home", "count"),
            results_found=("result_found", "sum"),
            min_date=("date", "min"),
            max_date=("date", "max"),
        )
        .reset_index()
        .sort_values("generated_at")
    )

    print("[debug] run completion summary:")
    for _, r in run_summary.iterrows():
        print(
            f"  run_id={r['run_id']} "
            f"games={int(r['games'])} "
            f"results_found={int(r['results_found'])} "
            f"window={r['min_date']} to {r['max_date']}"
        )

    completed_runs = run_summary[
        (run_summary["games"] >= 8) &
        (run_summary["games"] == run_summary["results_found"])
    ].copy()

    if completed_runs.empty:
        print("[warn] No fully completed round found yet")
        empty_csv(COMPLETED_OUT, out_cols)
        return

    latest_completed_run_id = completed_runs.sort_values("generated_at").iloc[-1]["run_id"]

    out = merged[merged["run_id"] == latest_completed_run_id].copy()
    out = out.sort_values(["date", "kickoff_local", "home"]).copy()
    out.to_csv(COMPLETED_OUT, index=False)

    print(f"[info] Wrote {len(out)} rows for latest completed run_id={latest_completed_run_id} -> {COMPLETED_OUT}")

    missing = out[~out["result_found"]]
    if not missing.empty:
        print("[warn] Rows still missing results in selected run:")
        print(missing[["date", "home", "away"]].to_string(index=False))


if __name__ == "__main__":
    main()
