import os
import pandas as pd

PRED_HISTORY_PATH = "predictions_history.csv"
RESULTS_CACHE_PATH = "results_cache.csv"

BET_HISTORY_OUT = "bet_history.csv"
BET_SUMMARY_OUT = "bet_summary.csv"

START_BANKROLL = float(os.getenv("BANKROLL", "200"))


TEAM_ALIASES = {
    "SEA EAGLES": "MANLY",
    "MANLY SEA EAGLES": "MANLY",
    "WESTS TIGERS": "TIGERS",
    "ST GEORGE ILLAWARRA DRAGONS": "DRAGONS",
    "GOLD COAST TITANS": "TITANS",
    "NORTH QUEENSLAND COWBOYS": "COWBOYS",
    "SYDNEY ROOSTERS": "ROOSTERS",
    "SOUTH SYDNEY RABBITOHS": "RABBITOHS",
    "NEWCASTLE KNIGHTS": "KNIGHTS",
    "CANBERRA RAIDERS": "RAIDERS",
    "CRONULLA SHARKS": "SHARKS",
    "CRONULLA-SUTHERLAND SHARKS": "SHARKS",
    "PARRAMATTA EELS": "EELS",
    "NEW ZEALAND WARRIORS": "WARRIORS",
    "THE DOLPHINS": "DOLPHINS",
    "REDCLIFFE DOLPHINS": "DOLPHINS",
    "BRISBANE BRONCOS": "BRONCOS",
    "MELBOURNE STORM": "STORM",
    "PENRITH PANTHERS": "PANTHERS",
    "CANTERBURY BULLDOGS": "BULLDOGS",
    "CANTERBURY-BANKSTOWN BULLDOGS": "BULLDOGS",
}


def _norm(s: str) -> str:
    s = str(s).strip().upper()
    s = " ".join(s.split())
    return TEAM_ALIASES.get(s, s)


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _resolve_prob_column(df: pd.DataFrame) -> str | None:
    for col in ["final_home_win_prob", "model_home_win_prob", "home_win_prob"]:
        if col in df.columns:
            return col
    return None


def _load_predictions(path: str) -> pd.DataFrame:
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame()

    prob_col = _resolve_prob_column(df)
    if prob_col is None:
        return pd.DataFrame()

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["generated_at"] = pd.to_datetime(df["generated_at"], errors="coerce", utc=True)

    df["home"] = df["home"].map(_norm)
    df["away"] = df["away"].map(_norm)
    df["pick"] = df["pick"].astype(str).str.upper().str.strip()

    df = df.dropna(subset=["date", "home", "away"])

    df = df.rename(columns={prob_col: "home_win_prob"})
    return df


def _load_results(path: str) -> pd.DataFrame:
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["home"] = df["home"].map(_norm)
    df["away"] = df["away"].map(_norm)

    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")

    df = df.dropna(subset=["date", "home", "away", "home_pts", "away_pts"])

    df = df.drop_duplicates(subset=["date", "home", "away"], keep="last")
    return df


def _latest_prediction_per_match(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "home", "away", "generated_at"])
    df = df.drop_duplicates(subset=["date", "home", "away"], keep="last")
    return df


def _match_results(pred: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    merged = pred.merge(
        res,
        how="left",
        on=["date", "home", "away"]
    )

    unmatched = merged["home_pts"].isna().sum()
    print(f"[debug] matched={len(merged)-unmatched} | unmatched={unmatched}")

    return merged


def _actual(row):
    if pd.isna(row["home_pts"]):
        return "PENDING"
    if row["home_pts"] > row["away_pts"]:
        return "HOME"
    if row["away_pts"] > row["home_pts"]:
        return "AWAY"
    return "DRAW"


def _status(row):
    if row["pick"] not in {"HOME", "AWAY"}:
        return "NO_BET"

    actual = _actual(row)

    if actual == "PENDING":
        return "PENDING"
    if actual == "DRAW":
        return "DRAW"
    if row["pick"] == actual:
        return "WIN"
    return "LOSS"


def _profit(row):
    stake = float(row.get("stake", 0) or 0)

    if row["bet_status"] == "WIN":
        odds = float(row["home_odds"] if row["pick"] == "HOME" else row["away_odds"])
        return stake * (odds - 1)
    if row["bet_status"] == "LOSS":
        return -stake
    if row["bet_status"] == "DRAW":
        return -stake

    return 0.0


def main():
    pred = _load_predictions(PRED_HISTORY_PATH)
    res = _load_results(RESULTS_CACHE_PATH)

    if pred.empty or res.empty:
        print("[warn] Missing predictions or results")
        return

    pred = _latest_prediction_per_match(pred)
    merged = _match_results(pred, res)

    merged["actual_result"] = merged.apply(_actual, axis=1)
    merged["bet_status"] = merged.apply(_status, axis=1)
    merged["profit_units"] = merged.apply(_profit, axis=1)

    # ONLY settled bets affect bankroll
    settled = merged[merged["bet_status"].isin(["WIN", "LOSS", "DRAW"])].copy()

    settled = settled.sort_values(["date", "generated_at"])

    bankroll = START_BANKROLL
    bankroll_list = []

    for _, row in settled.iterrows():
        bankroll += row["profit_units"]
        bankroll_list.append(bankroll)

    settled["bankroll_after"] = bankroll_list

    merged = merged.merge(
        settled[["date", "home", "away", "bankroll_after"]],
        on=["date", "home", "away"],
        how="left"
    )

    merged.to_csv(BET_HISTORY_OUT, index=False)

    profit = settled["profit_units"].sum()
    staked = settled["stake"].sum()

    roi = profit / staked if staked > 0 else 0

    summary = pd.DataFrame([{
        "start_bankroll": START_BANKROLL,
        "closing_bankroll": round(START_BANKROLL + profit, 2),
        "bets_settled": len(settled),
        "profit_units": round(profit, 2),
        "roi": round(roi, 4)
    }])

    summary.to_csv(BET_SUMMARY_OUT, index=False)

    print(f"[info] bankroll=${START_BANKROLL + profit:.2f} | profit={profit:.2f}u | ROI={roi:.2%}")


if __name__ == "__main__":
    main()
