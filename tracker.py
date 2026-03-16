import os
import pandas as pd

PRED_PATH = "predictions.csv"
PRED_HIST_PATH = "predictions_history.csv"
RESULTS_CACHE_PATH = "results_cache.csv"
OUT_PATH = "accuracy.csv"
SUMMARY_PATH = "accuracy_summary.csv"


def _norm(s: str) -> str:
    return " ".join(str(s).strip().upper().split())


def _empty_accuracy() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "date", "home", "away", "match_type", "generated_at",
        "home_win_prob", "exp_margin_home",
        "home_pts", "away_pts", "actual_margin",
        "pred_winner", "actual_winner", "winner_correct",
        "is_draw", "brier", "abs_margin_error"
    ])


def _write_empty_outputs(message: str) -> None:
    _empty_accuracy().to_csv(OUT_PATH, index=False)
    pd.DataFrame([{
        "scored_matches": 0,
        "winner_accuracy": pd.NA,
        "brier": pd.NA,
        "mae_margin": pd.NA,
        "draws": 0,
        "exact_matches": 0,
        "minus1_matches": 0,
        "plus1_matches": 0
    }]).to_csv(SUMMARY_PATH, index=False)
    print(message)


def _load_prediction_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Could not read {path}: {e}")
        return pd.DataFrame()

    need_pred = {"date", "home", "away", "home_win_prob"}
    if not need_pred.issubset(set(df.columns)):
        print(f"[warn] {path} missing required columns.")
        return pd.DataFrame()

    keep_cols = ["date", "home", "away", "home_win_prob", "exp_margin_home", "generated_at"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[keep_cols].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home"] = df["home"].map(_norm)
    df["away"] = df["away"].map(_norm)
    df["home_win_prob"] = pd.to_numeric(df["home_win_prob"], errors="coerce")
    df["exp_margin_home"] = pd.to_numeric(df["exp_margin_home"], errors="coerce")
    df["generated_at"] = pd.to_datetime(df["generated_at"], errors="coerce", utc=True)

    df = df.dropna(subset=["date", "home", "away", "home_win_prob"]).copy()
    df["date"] = df["date"].dt.normalize()

    print(f"[info] Loaded predictions from {path}: {len(df)} rows")
    return df


def _load_results_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Could not read {path}: {e}")
        return pd.DataFrame()

    need_res = {"date", "home", "away", "home_pts", "away_pts"}
    if not need_res.issubset(set(df.columns)):
        print(f"[warn] {path} missing required columns.")
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home"] = df["home"].map(_norm)
    df["away"] = df["away"].map(_norm)
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")

    df = df.dropna(subset=["date", "home", "away", "home_pts", "away_pts"]).copy()
    df["date"] = df["date"].dt.normalize()

    df = (
        df.sort_values(["date", "home", "away"])
          .drop_duplicates(subset=["date", "home", "away"], keep="last")
          .reset_index(drop=True)
    )

    print(f"[info] Loaded results from {path}: {len(df)} rows")
    return df


def _match_predictions_to_results(pred: pd.DataFrame, res: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Match results to predictions using:
      1) exact date match
      2) prediction date - 1 day
      3) prediction date + 1 day

    Team orientation must remain the same.
    """
    pred = pred.copy()
    res = res.copy()

    pred["match_key"] = pred["home"] + "||" + pred["away"]
    res["match_key"] = res["home"] + "||" + res["away"]
    pred["pred_row_id"] = range(len(pred))

    matched_parts = []
    matched_pred_ids = set()

    exact_count = 0
    minus1_count = 0
    plus1_count = 0

    for delta, label in [(0, "exact"), (-1, "minus1"), (1, "plus1")]:
        still_unmatched = pred.loc[~pred["pred_row_id"].isin(matched_pred_ids)].copy()
        if still_unmatched.empty:
            continue

        still_unmatched["match_date"] = still_unmatched["date"] + pd.to_timedelta(delta, unit="D")
        res_tmp = res.rename(columns={"date": "match_date"}).copy()

        j = still_unmatched.merge(
            res_tmp,
            on=["match_date", "home", "away", "match_key"],
            how="inner",
            suffixes=("_pred", "_res")
        )

        if j.empty:
            continue

        j = (
            j.sort_values(["pred_row_id"])
             .drop_duplicates(subset=["pred_row_id"], keep="first")
             .copy()
        )
        j["match_type"] = label

        matched_parts.append(j)
        matched_pred_ids.update(set(j["pred_row_id"].tolist()))

        if label == "exact":
            exact_count += len(j)
        elif label == "minus1":
            minus1_count += len(j)
        elif label == "plus1":
            plus1_count += len(j)

    out = pd.concat(matched_parts, ignore_index=True) if matched_parts else pd.DataFrame()

    stats = {
        "exact_matches": exact_count,
        "minus1_matches": minus1_count,
        "plus1_matches": plus1_count,
    }

    print(f"[info] Matches found — exact: {exact_count}, -1 day: {minus1_count}, +1 day: {plus1_count}")
    return out, stats


def main():
    # Prefer history first, then fall back to current predictions only if needed
    hist_pred = _load_prediction_file(PRED_HIST_PATH)
    current_pred = _load_prediction_file(PRED_PATH) if hist_pred.empty else pd.DataFrame()

    pred_frames = []
    if not hist_pred.empty:
        pred_frames.append(hist_pred)
    if not current_pred.empty:
        pred_frames.append(current_pred)

    if not pred_frames:
        _write_empty_outputs("No usable prediction files found.")
        return

    pred = pd.concat(pred_frames, ignore_index=True)

    pred = pred.sort_values(["date", "home", "away", "generated_at"])
    pred = pred.drop_duplicates(subset=["date", "home", "away"], keep="last").reset_index(drop=True)

    if not os.path.exists(RESULTS_CACHE_PATH):
        _write_empty_outputs("No results_cache.csv found yet — nothing to score.")
        return

    res = _load_results_file(RESULTS_CACHE_PATH)
    if res.empty:
        _write_empty_outputs("No usable results found.")
        return

    j, match_stats = _match_predictions_to_results(pred, res)

    if j.empty:
        _write_empty_outputs("No matching completed matches to score yet.")
        return

    j["actual_margin"] = j["home_pts"] - j["away_pts"]
    j["is_draw"] = (j["home_pts"] == j["away_pts"]).astype(int)

    j["pred_winner"] = j.apply(
        lambda r: r["home"] if r["home_win_prob"] >= 0.5 else r["away"],
        axis=1
    )

    def _actual_winner(row):
        if row["home_pts"] > row["away_pts"]:
            return row["home"]
        if row["away_pts"] > row["home_pts"]:
            return row["away"]
        return "DRAW"

    j["actual_winner"] = j.apply(_actual_winner, axis=1)

    j["winner_correct"] = (
        (j["actual_winner"] != "DRAW") &
        (j["pred_winner"] == j["actual_winner"])
    ).astype(int)

    actual_home_win_prob = j.apply(
        lambda r: 1.0 if r["home_pts"] > r["away_pts"] else (0.0 if r["away_pts"] > r["home_pts"] else 0.5),
        axis=1
    )
    j["brier"] = (j["home_win_prob"] - actual_home_win_prob) ** 2

    j["abs_margin_error"] = (
        pd.to_numeric(j["exp_margin_home"], errors="coerce") - j["actual_margin"]
    ).abs()

    j["date"] = pd.to_datetime(j["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    j["generated_at"] = pd.to_datetime(j["generated_at"], errors="coerce", utc=True)

    out_cols = [
        "date", "home", "away", "match_type", "generated_at",
        "home_win_prob", "exp_margin_home",
        "home_pts", "away_pts", "actual_margin",
        "pred_winner", "actual_winner", "winner_correct",
        "is_draw", "brier", "abs_margin_error"
    ]

    out_df = j[out_cols].sort_values(["date", "home", "away"]).reset_index(drop=True)
    out_df.to_csv(OUT_PATH, index=False)

    scored = len(out_df)
    winner_accuracy = out_df["winner_correct"].mean() if scored else pd.NA
    brier = out_df["brier"].mean() if scored else pd.NA
    mae_margin = out_df["abs_margin_error"].mean() if scored else pd.NA
    draws = int(out_df["is_draw"].sum()) if scored else 0

    summary_df = pd.DataFrame([{
        "scored_matches": scored,
        "winner_accuracy": winner_accuracy,
        "brier": brier,
        "mae_margin": mae_margin,
        "draws": draws,
        "exact_matches": match_stats["exact_matches"],
        "minus1_matches": match_stats["minus1_matches"],
        "plus1_matches": match_stats["plus1_matches"],
    }])
    summary_df.to_csv(SUMMARY_PATH, index=False)

    acc_text = f"{winner_accuracy:.0%}" if pd.notna(winner_accuracy) else "NA"
    brier_text = f"{brier:.3f}" if pd.notna(brier) else "NA"
    mae_text = f"{mae_margin:.2f}" if pd.notna(mae_margin) else "NA"

    print(
        f"Scored matches: {scored} | "
        f"Winner accuracy: {acc_text} | "
        f"Brier: {brier_text} | "
        f"Margin MAE: {mae_text} | "
        f"Draws: {draws}"
    )


if __name__ == "__main__":
    main()
