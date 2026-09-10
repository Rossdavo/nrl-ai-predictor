import os
import math
import pandas as pd

PRED_HISTORY_PATH = "predictions_history.csv"
RESULTS_CACHE_PATH = "results_cache.csv"
OUT_PATH = "performance.csv"


OUTPUT_COLUMNS = [
    "date",
    "home",
    "away",
    "run_id",
    "round_key",
    "predicted_winner",
    "actual_winner",
    "correct",
    "home_win_probability",
    "winner_probability",
    "exp_margin_home",
    "actual_margin_home",
    "margin_error",
    "brier",
]


def _load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[performance] Warning: could not read {path}: {e}")
        return pd.DataFrame()


def _norm_team(value: object) -> str:
    return " ".join(str(value).strip().upper().split())


def _pick_numeric_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")

    return pd.Series([float("nan")] * len(df), index=df.index, dtype=float)


def _latest_prediction_per_match(pred: pd.DataFrame) -> pd.DataFrame:
    """
    Multiple snapshots of the same finals match are now deliberately archived.
    For performance scoring, use the latest available prediction snapshot for
    each date/home/away combination.
    """
    work = pred.copy()

    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    work = work.dropna(subset=["date", "home", "away"]).copy()

    work["_home_key"] = work["home"].map(_norm_team)
    work["_away_key"] = work["away"].map(_norm_team)

    if "run_utc" in work.columns:
        work["_run_sort"] = pd.to_datetime(work["run_utc"], errors="coerce", utc=True)
    else:
        work["_run_sort"] = pd.NaT

    if "run_id" not in work.columns:
        work["run_id"] = ""
    work["run_id"] = work["run_id"].fillna("").astype(str)

    # run_id is YYYYMMDDHHMMSS, so it is a useful fallback sort key.
    work = work.sort_values(
        ["date", "_home_key", "_away_key", "_run_sort", "run_id"],
        na_position="first",
    )

    work = work.drop_duplicates(
        subset=["date", "_home_key", "_away_key"],
        keep="last",
    ).reset_index(drop=True)

    return work


def build_performance() -> pd.DataFrame:
    pred = _load_csv_safe(PRED_HISTORY_PATH)
    results = _load_csv_safe(RESULTS_CACHE_PATH)

    if pred.empty:
        print(f"[performance] No {PRED_HISTORY_PATH} found.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if results.empty:
        print(f"[performance] No {RESULTS_CACHE_PATH} found.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    pred_required = {"date", "home", "away", "predicted_winner"}
    result_required = {"date", "home", "away", "home_pts", "away_pts"}

    pred_missing = pred_required - set(pred.columns)
    result_missing = result_required - set(results.columns)

    if pred_missing:
        print(f"[performance] {PRED_HISTORY_PATH} missing: {sorted(pred_missing)}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if result_missing:
        print(f"[performance] {RESULTS_CACHE_PATH} missing: {sorted(result_missing)}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    pred = _latest_prediction_per_match(pred)

    results = results.copy()
    results["date"] = pd.to_datetime(results["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    results["home_pts"] = pd.to_numeric(results["home_pts"], errors="coerce")
    results["away_pts"] = pd.to_numeric(results["away_pts"], errors="coerce")
    results = results.dropna(subset=["date", "home", "away", "home_pts", "away_pts"]).copy()

    results["_home_key"] = results["home"].map(_norm_team)
    results["_away_key"] = results["away"].map(_norm_team)

    # Guard against duplicate copies of the same completed result.
    results = results.drop_duplicates(
        subset=["date", "_home_key", "_away_key"],
        keep="last",
    )

    merged = pred.merge(
        results[
            [
                "date",
                "_home_key",
                "_away_key",
                "home_pts",
                "away_pts",
            ]
        ],
        on=["date", "_home_key", "_away_key"],
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Flexible support for the probability column names used across earlier
    # versions of predict.py.
    merged["home_win_probability"] = _pick_numeric_column(
        merged,
        [
            "home_win_probability",
            "final_home_win_probability",
            "home_prob",
            "home_probability",
        ],
    )

    # Old archives may only contain winner probability.
    winner_prob_source = _pick_numeric_column(
        merged,
        [
            "win_probability",
            "winner_probability",
            "predicted_winner_probability",
        ],
    )

    exp_margin_source = _pick_numeric_column(
        merged,
        [
            "exp_margin_home",
            "expected_margin_home",
            "exp_margin",
            "predicted_margin_home",
        ],
    )

    rows = []

    for _, r in merged.iterrows():
        home = str(r["home"]).strip()
        away = str(r["away"]).strip()
        predicted_winner = str(r.get("predicted_winner", "")).strip()

        home_pts = float(r["home_pts"])
        away_pts = float(r["away_pts"])

        if home_pts > away_pts:
            actual_winner = home
            actual_home_result = 1.0
        elif away_pts > home_pts:
            actual_winner = away
            actual_home_result = 0.0
        else:
            actual_winner = "DRAW"
            actual_home_result = 0.5

        if actual_winner == "DRAW":
            correct = float("nan")
        else:
            correct = 1 if _norm_team(predicted_winner) == _norm_team(actual_winner) else 0

        home_prob = r["home_win_probability"]

        # If home probability was absent in an old row, reconstruct it from
        # winner probability when possible.
        winner_prob = winner_prob_source.loc[r.name]
        if pd.isna(home_prob) and pd.notna(winner_prob):
            if _norm_team(predicted_winner) == _norm_team(home):
                home_prob = float(winner_prob)
            elif _norm_team(predicted_winner) == _norm_team(away):
                home_prob = 1.0 - float(winner_prob)

        if pd.notna(home_prob):
            home_prob = min(1.0, max(0.0, float(home_prob)))
            brier = (home_prob - actual_home_result) ** 2

            if _norm_team(predicted_winner) == _norm_team(home):
                winner_probability = home_prob
            elif _norm_team(predicted_winner) == _norm_team(away):
                winner_probability = 1.0 - home_prob
            else:
                winner_probability = float("nan")
        else:
            brier = float("nan")
            winner_probability = float(winner_prob) if pd.notna(winner_prob) else float("nan")

        exp_margin_home = exp_margin_source.loc[r.name]
        actual_margin_home = home_pts - away_pts

        if pd.notna(exp_margin_home):
            exp_margin_home = float(exp_margin_home)
            margin_error = abs(exp_margin_home - actual_margin_home)
        else:
            margin_error = float("nan")

        rows.append(
            {
                "date": r["date"],
                "home": home,
                "away": away,
                "run_id": str(r.get("run_id", "")).strip(),
                "round_key": str(r.get("round_key", "")).strip(),
                "predicted_winner": predicted_winner,
                "actual_winner": actual_winner,
                "correct": correct,
                "home_win_probability": round(home_prob, 4) if pd.notna(home_prob) else float("nan"),
                "winner_probability": round(winner_probability, 4) if pd.notna(winner_probability) else float("nan"),
                "exp_margin_home": round(exp_margin_home, 2) if pd.notna(exp_margin_home) else float("nan"),
                "actual_margin_home": round(actual_margin_home, 2),
                "margin_error": round(margin_error, 2) if pd.notna(margin_error) else float("nan"),
                "brier": round(brier, 4) if pd.notna(brier) else float("nan"),
            }
        )

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return out.sort_values(["date", "home", "away"]).reset_index(drop=True)


def main():
    out = build_performance()
    out.to_csv(OUT_PATH, index=False)

    if out.empty:
        print("[performance] performance.csv updated (0 scored matches)")
        return

    scored = out["correct"].dropna()
    accuracy = float(scored.mean()) if not scored.empty else float("nan")

    brier_series = pd.to_numeric(out["brier"], errors="coerce").dropna()
    margin_series = pd.to_numeric(out["margin_error"], errors="coerce").dropna()

    brier = float(brier_series.mean()) if not brier_series.empty else float("nan")
    margin_mae = float(margin_series.mean()) if not margin_series.empty else float("nan")
    draws = int((out["actual_winner"] == "DRAW").sum())

    accuracy_text = f"{accuracy:.1%}" if not math.isnan(accuracy) else "n/a"
    brier_text = f"{brier:.3f}" if not math.isnan(brier) else "n/a"
    margin_text = f"{margin_mae:.2f}" if not math.isnan(margin_mae) else "n/a"

    print(
        f"[performance] performance.csv updated "
        f"({len(scored)} scored matches | accuracy={accuracy_text} | "
        f"Brier={brier_text} | Margin MAE={margin_text} | draws={draws})"
    )


if __name__ == "__main__":
    main()
