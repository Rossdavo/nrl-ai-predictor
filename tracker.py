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

    need_pred = {"date", "home", "away", "home_win_prob"}

    # Try normal read first
    try:
        df = pd.read_csv(path)
    except Exception:
        # Fallback: tolerate bad lines / mixed-width history rows
        try:
            df = pd.read_csv(path, on_bad_lines="skip", engine="python")
            print(f"[warn] Loaded {path} with bad lines skipped.")
        except Exception as e:
            print(f"[warn] Could not read {path}: {e}")
            return pd.DataFrame()

    if not need_pred.issubset(set(df.columns)):
        print(f"[warn] {path} missing required columns.")
        return pd.DataFrame()

    # Keep only the columns we actually need, ignore schema drift
    keep_cols = ["date", "home", "away", "home_win_prob", "exp_margin_home", "generated_at"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[keep_cols].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["home"] = df["home"].map(_norm)
    df["away"] = df["away"].map(_norm)
    df["home_win_prob"] = pd.to_numeric(df["home_win_prob"], errors="coerce")
    df["exp_margin_home"] = pd.to_numeric(df["exp_margin_home"], errors="coerce")
    df["generated_at"] = df["generated_at"].astype(str).str.strip()

    df = df.dropna(subset=["date", "home", "away", "home_win_prob"])

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

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["home"] = df["home"].map(_norm)
    df["away"] = df["away"].map(_norm)
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")

    df = df.dropna(subset=["date", "home", "away", "home_pts", "away_pts"])

    print(f"[info] Loaded results from {path}: {len(df)} rows")
    return df


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

    # Match completed results to predictions
    j = pred.merge(res, on=["date", "home", "away"], how="inner")

    if j.empty:
        _empty_accuracy().to_csv(OUT_PATH, index=False)
        print("No matching completed matches to score yet.")
        return

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
