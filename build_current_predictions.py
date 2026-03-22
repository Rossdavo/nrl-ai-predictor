import os
import pandas as pd

PRED_HISTORY_PATH = "predictions_history.csv"
CURRENT_OUT = "current_predictions.csv"

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


def empty_csv(path: str, cols: list[str]) -> None:
    pd.DataFrame(columns=cols).to_csv(path, index=False)


def main():
    if not os.path.exists(PRED_HISTORY_PATH):
        print(f"[warn] Missing {PRED_HISTORY_PATH}")
        empty_csv(CURRENT_OUT, EXPECTED_COLS)
        return

    try:
        df = pd.read_csv(PRED_HISTORY_PATH, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"[warn] Could not read {PRED_HISTORY_PATH}: {e}")
        empty_csv(CURRENT_OUT, EXPECTED_COLS)
        return

    if df.empty:
        print("[warn] No valid history rows found")
        empty_csv(CURRENT_OUT, EXPECTED_COLS)
        return

    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = ""

    df = df[EXPECTED_COLS].copy()
    df["generated_at"] = pd.to_datetime(df["generated_at"], errors="coerce", utc=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    df = df.dropna(subset=["generated_at"])
    if df.empty:
        print("[warn] No rows with valid generated_at")
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

    print(f"[info] Wrote {len(out)} rows for latest run_id={latest_run_id} -> {CURRENT_OUT}")


if __name__ == "__main__":
    main()
