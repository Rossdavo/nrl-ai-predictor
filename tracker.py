import os
import pandas as pd

PRED_PATH = "predictions.csv"
RESULTS_CACHE_PATH = "results_cache.csv"
OUT_PATH = "accuracy.csv"


def _norm(s: str) -> str:
    return str(s).strip()


def main():
    if not os.path.exists(PRED_PATH):
        pd.DataFrame().to_csv(OUT_PATH, index=False)
        print("No predictions.csv found.")
        return

    pred = pd.read_csv(PRED_PATH)
    need_pred = {"date", "home", "away", "home_win_prob"}
    if not need_pred.issubset(set(pred.columns)):
        pd.DataFrame().to_csv(OUT_PATH, index=False)
        print("predictions.csv missing required columns.")
        return

    if not os.path.exists(RESULTS_CACHE_PATH):
        # No results yet — still write an empty accuracy file
        pd.DataFrame(columns=[
            "date", "home", "away", "home_win_prob",
            "home_pts", "away_pts", "actual_margin",
            "pred_winner", "actual_winner", "winner_correct",
            "brier", "abs_margin_error"
        ]).to_csv(OUT_PATH, index=False)
        print("No results_cache.csv found yet — nothing to score.")
        return

    res = pd.read_csv(RESULTS_CACHE_PATH)
    need_res = {"date", "home", "away", "home_pts", "away_pts"}
    if not need_res.issubset(set(res.columns)):
        pd.DataFrame().to_csv(OUT_PATH, index=False)
        print("results_cache.csv missing required columns.")
        return

    pred["date"] = pred["date"].astype(str).str.strip()
    pred["home"] = pred["home"].map(_norm)
    pred["away"] = pred["away"].map(_norm)
    pred["home_win_prob"] = pd.to_numeric(pred["home_win_prob"], errors="coerce")

    res["date"] = res["date"].astype(str).str.slice(0, 10).str.strip()
    res["home"] = res["home"].map(_norm)
    res["away"] = res["away"].map(_norm)
    res["home_pts"] = pd.to_numeric(res["home_pts"], errors="coerce")
    res["away_pts"] = pd.to_numeric(res["away_pts"], errors="coerce")

    pred = pred.dropna(subset=["home_win_prob"])
    res = res.dropna(subset=["home_pts", "away_pts"])

    # Join on date + home + away (same orientation)
    j = pred.merge(res, on=["date", "home", "away"], how="inner")

    if j.empty:
        # Still write empty file
        pd.DataFrame(columns=[
            "date", "home", "away", "home_win_prob",
            "home_pts", "away_pts", "actual_margin",
            "pred_winner", "actual_winner", "winner_correct",
            "brier", "abs_margin_error"
        ]).to_csv(OUT_PATH, index=False)
        print("No matching completed matches to score yet.")
        return

    j["actual_margin"] = j["home_pts"] - j["away_pts"]
    j["pred_winner"] = j.apply(lambda r: r["home"] if r["home_win_prob"] >= 0.5 else r["away"], axis=1)
    j["actual_winner"] = j.apply(lambda r: r["home"] if r["home_pts"] > r["away_pts"] else r["away"], axis=1)
    j["winner_correct"] = (j["pred_winner"] == j["actual_winner"]).astype(int)

    actual_home_win = (j["home_pts"] > j["away_pts"]).astype(int)
    j["brier"] = (j["home_win_prob"] - actual_home_win) ** 2

    if "exp_margin_home" in j.columns:
        j["exp_margin_home"] = pd.to_numeric(j["exp_margin_home"], errors="coerce")
        j["abs_margin_error"] = (j["exp_margin_home"] - j["actual_margin"]).abs()
    else:
        j["abs_margin_error"] = pd.NA

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
