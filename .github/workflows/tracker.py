import os
from typing import Tuple
import pandas as pd

PRED_PATH = "predictions.csv"
RESULTS_CACHE_PATH = "results_cache.csv"
OUT_PATH = "accuracy.csv"


def _norm_team(s: str) -> str:
    return str(s).strip()


def _load_predictions() -> pd.DataFrame:
    if not os.path.exists(PRED_PATH):
        return pd.DataFrame()

    df = pd.read_csv(PRED_PATH)
    # required columns
    need = {"date", "home", "away", "home_win_prob"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    df["date"] = df["date"].astype(str).str.strip()
    df["home"] = df["home"].map(_norm_team)
    df["away"] = df["away"].map(_norm_team)
    df["home_win_prob"] = pd.to_numeric(df["home_win_prob"], errors="coerce")
    return df.dropna(subset=["home_win_prob"])


def _load_results() -> pd.DataFrame:
    # Prefer cache file because predict.py updates it when live fetch succeeds.
    if not os.path.exists(RESULTS_CACHE_PATH):
        return pd.DataFrame()

    df = pd.read_csv(RESULTS_CACHE_PATH)
    need = {"date", "home", "away", "home_pts", "away_pts"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    df["date"] = df["date"].astype(str).str.slice(0, 10).str.strip()
    df["home"] = df["home"].map(_norm_team)
    df["away"] = df["away"].map(_norm_team)
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")
    df = df.dropna(subset=["home_pts", "away_pts"])
    return df


def _join(pred: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    if pred.empty or res.empty:
        return pd.DataFrame()

    # Join on exact fixture orientation first
    j = pred.merge(
        res,
        on=["date", "home", "away"],
        how="inner",
        suffixes=("", "_res"),
    )

    if j.empty:
        # Some feeds swap home/away; attempt a reversed join and flip
        rev = pred.merge(
            res.rename(columns={"home": "away", "away": "home", "home_pts": "away_pts", "away_pts": "home_pts"}),
            on=["date", "home", "away"],
            how="inner",
        )
        j = rev

    return j


def _metrics(df: pd.DataFrame) -> Tuple[float, float, float]:
    # winner correctness
    actual_home_win = (df["home_pts"] > df["away_pts"]).astype(int)
    pred_home_win = (df["home_win_prob"] >= 0.5).astype(int)
    acc = float((actual_home_win == pred_home_win).mean()) if len(df) else float("nan")

    # abs margin error (needs exp_margin_home if available)
    if "exp_margin_home" in df.columns:
        df["exp_margin_home"] = pd.to_numeric(df["exp_margin_home"], errors="coerce")
        margin_err = (df["exp_margin_home"] - (df["home_pts"] - df["away_pts"])).abs()
        mae = float(margin_err.mean()) if margin_err.notna().any() else float("nan")
    else:
        mae = float("nan")

    # Brier score for win prob
    brier = ((df["home_win_prob"] - actual_home_win) ** 2).mean()
    brier = float(brier) if len(df) else float("nan")

    return acc, mae, brier


def main():
    pred = _load_predictions()
    res = _load_results()
    j = _join(pred, res)

    if j.empty:
        # still write empty file so report can show "no results yet"
        pd.DataFrame(columns=[
            "date", "home", "away", "home_win_prob",
            "home_pts", "away_pts", "actual_margin",
            "pred_winner", "actual_winner", "winner_correct",
            "brier", "abs_margin_error"
        ]).to_csv(OUT_PATH, index=False)
        print("No completed matches to score yet.")
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

    # Save detailed scoring table
    out_cols = [
        "date", "home", "away",
        "home_win_prob",
        "home_pts", "away_pts", "actual_margin",
        "pred_winner", "actual_winner", "winner_correct",
        "brier", "abs_margin_error"
    ]
    j[out_cols].sort_values(["date", "home"]).to_csv(OUT_PATH, index=False)

    acc, mae, brier = _metrics(j)
    print(f"Scored matches: {len(j)} | Winner acc: {acc:.0%} | Margin MAE: {mae:.2f} | Brier: {brier:.3f}")


if __name__ == "__main__":
    main()
