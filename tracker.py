import os
import pandas as pd

PRED_PATH = "predictions.csv"
PRED_HIST_PATH = "predictions_history.csv"
RESULTS_CACHE_PATH = "results_cache.csv"
OUT_PATH = "accuracy.csv"


def _norm(s: str) -> str:
    return str(s).strip()


def _empty_accuracy() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "date", "home", "away", "home_win_prob",
        "home_pts", "away_pts", "actual_margin",
        "pred_winner", "actual_winner", "winner_correct",
        "brier", "abs_margin_error"
    ])


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
    df["generated_at"] = pd.to_datetime(df["generated_at"], errors="coerce")

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

    print(f"[info] Loaded results from {path}: {len(df)} rows")
    return df


def _match_predictions_to_results(pred: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
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
        res_tmp = res.copy().rename(columns={"date": "match_date"})

        j = still_unmatched.merge(
            res_tmp,
            on=["match_date", "home", "away", "match_key"],
            how="inner",
            suffixes=("_pred", "_res")
        )

        if j.empty:
            continue

        j = j.sort_values(["pred_row_id"]).drop_duplicates(subset=["pred_row_id"], keep="first").copy()
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

    print(f"[info] Matches found — exact: {exact_count}, -1 day: {minus1_count}, +1 day: {plus1_count}")
    return out


def main():
    pred_frames = []

    current_pred = _load_prediction_file(PRED_PATH)
    if not current_pred.empty:
        pred_frames.append(current_pred)

    hist_pred = _load_prediction_file(PRED_HIST_PATH)
    if not hist_pred.empty:
        pred_frames.append(hist_pred)

    if not pred_frames:
        _empty_accuracy().to_csv(OUT_PATH, index=False)
        print("No usable prediction files found.")
        return

    pred = pd.concat(pred_frames, ignore_index=True)

    # Keep latest version for each match
    pred = pred.sort_values(["date", "home", "away", "generated_at"])
    pred = pred.drop_duplicates(subset=["date", "home", "away"], keep="last").reset_index(drop=True)

    if not os.path.exists(RESULTS_CACHE_PATH):
        _empty_accuracy().to_csv(OUT_PATH, index=False)
        print("No results_cache.csv found yet — nothing to score.")
        return

    res = _load_results_file(RESULTS_CACHE_PATH)
    if res.empty:
        _empty_accuracy().to_csv(OUT_PATH, index=False)
        print("No usable results found.")
        return

    j = _match_predictions_to_results(pred, res)

    if j.empty:
        _empty_accuracy().to_csv(OUT_PATH, index=False)
        print("No matching completed matches to score yet.")
        return

    j["date"] = j["date_pred"].dt.strftime("%Y-%m-%d")
    j["actual_margin"] = j["home_pts"] - j["away_pts"]

    j["pred_winner"] = j.apply(
        lambda r: r["home"] if r["home_win_prob"] >= 0.5 else r["away"],
        axis=1
    )

    j["actual_winner"] = j.apply(
        lambda r: r["home"] if r["home_pts"] > r["away_pts"] else r["away"],
        axis=1
    )

    j["winner_correct"] = (j["pred_winner"] == j["actual_winner"]).astype(int)

    actual_home_win = (j["home_pts"] > j["away_pts"]).astype(int)
    j["brier"] = (j["home_win_prob"] - actual_home_win) ** 2
    j["abs_margin_error"] = (pd.to_numeric(j["exp_margin_home"], errors="coerce") - j["actual_margin"]).abs()

    out_cols = [
        "date", "home", "away",
        "home_win_prob",
        "home_pts", "away_pts", "actual_margin",
        "pred_winner", "actual_winner", "winner_correct",
        "brier", "abs_margin_error"
    ]

    j[out_cols].sort_values(["date", "home"]).to_csv(OUT_PATH, index=False)

    scored = len(j)
    win_acc = j["winner_correct"].mean()
    brier = j["brier"].mean()
    print(f"Scored matches: {scored} | Winner accuracy: {win_acc:.0%} | Brier: {brier:.3f}")


if __name__ == "__main__":
    main()
