import os
import pandas as pd

PRED_HISTORY_PATH = "predictions_history.csv"
RESULTS_CACHE_PATH = "results_cache.csv"

BET_HISTORY_OUT = "bet_history.csv"
BET_SUMMARY_OUT = "bet_summary.csv"

START_BANKROLL = float(os.getenv("BANKROLL", "200"))


def _norm(s: str) -> str:
    return " ".join(str(s).strip().upper().split())


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as e1:
        print(f"[warn] Standard read failed for {path}: {e1}")

    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception as e2:
        print(f"[warn] Fallback read failed for {path}: {e2}")
        return pd.DataFrame()


def _resolve_prob_column(df: pd.DataFrame) -> str | None:
    for col in ["final_home_win_prob", "model_home_win_prob", "home_win_prob"]:
        if col in df.columns:
            return col
    return None


def _load_predictions(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[warn] Missing file: {path}")
        return pd.DataFrame()

    df = _safe_read_csv(path)
    if df.empty:
        print(f"[warn] No usable rows in {path}")
        return pd.DataFrame()

    prob_col = _resolve_prob_column(df)
    if prob_col is None:
        print(f"[warn] predictions history missing probability column")
        return pd.DataFrame()

    required = {
        "run_id", "date", "home", "away", "pick",
        "stake", "stake_dollars", "home_odds", "away_odds", "generated_at"
    }
    missing = required - set(df.columns)
    if missing:
        print(f"[warn] predictions history missing columns: {sorted(missing)}")
        return pd.DataFrame()

    df = df.copy()

    optional_cols = [
        "kickoff_local", "edge", "exp_margin_home", "recommended_bet",
        "stake_units", "mode", "rating_mode", "confidence",
        "favourite_team", "underdog_team", "upset_flag",
        "final_upset_score", "fragile_favourite"
    ]
    for col in optional_cols:
        if col not in df.columns:
            df[col] = pd.NA

    keep_cols = [
        "run_id", "date", "home", "away", "pick",
        "stake", "stake_dollars", "home_odds", "away_odds", "generated_at",
        "kickoff_local", "edge", "exp_margin_home", "recommended_bet",
        "stake_units", "mode", "rating_mode", "confidence",
        "favourite_team", "underdog_team", "upset_flag",
        "final_upset_score", "fragile_favourite", prob_col
    ]

    df = df[keep_cols].copy()
    df = df.rename(columns={prob_col: "home_win_prob"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["generated_at"] = pd.to_datetime(df["generated_at"], errors="coerce", utc=True)
    df["home"] = df["home"].map(_norm)
    df["away"] = df["away"].map(_norm)
    df["pick"] = df["pick"].astype(str).str.strip().str.upper()
    df["kickoff_local"] = df["kickoff_local"].astype(str).replace("nan", "").str.strip()

    for col in [
        "stake", "stake_dollars", "home_odds", "away_odds",
        "edge", "home_win_prob", "exp_margin_home", "stake_units",
        "confidence", "upset_flag", "final_upset_score", "fragile_favourite"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "stake_units" in df.columns:
        missing_units = df["stake_units"].isna()
        df.loc[missing_units, "stake_units"] = df.loc[missing_units, "stake"]

    df = df.dropna(subset=["date", "home", "away"]).copy()
    return df


def _load_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[warn] Missing file: {path}")
        return pd.DataFrame()

    df = _safe_read_csv(path)
    if df.empty:
        print(f"[warn] No usable rows in {path}")
        return pd.DataFrame()

    required = {"date", "home", "away", "home_pts", "away_pts"}
    missing = required - set(df.columns)
    if missing:
        print(f"[warn] results cache missing columns: {sorted(missing)}")
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["home"] = df["home"].map(_norm)
    df["away"] = df["away"].map(_norm)
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")

    df = df.dropna(subset=["date", "home", "away", "home_pts", "away_pts"]).copy()

    df = (
        df.sort_values(["date", "home", "away"])
        .drop_duplicates(subset=["date", "home", "away"], keep="last")
        .reset_index(drop=True)
    )
    return df


def _latest_prediction_per_match(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["date", "home", "away", "generated_at", "run_id"]
    for col in sort_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df.sort_values(sort_cols)
    df = df.drop_duplicates(subset=["date", "home", "away"], keep="last").reset_index(drop=True)
    return df


def _match_results_with_fallback(pred: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """
    Match results by:
      exact date
      prediction date - 1 day
      prediction date + 1 day
    """
    pred = pred.copy()
    res = res.copy()

    pred["pred_row_id"] = range(len(pred))
    pred["match_key"] = pred["home"] + "||" + pred["away"]
    res["match_key"] = res["home"] + "||" + res["away"]

    matched_parts = []
    matched_ids = set()

    for delta, label in [(0, "exact"), (-1, "minus1"), (1, "plus1")]:
        remaining = pred.loc[~pred["pred_row_id"].isin(matched_ids)].copy()
        if remaining.empty:
            continue

        remaining["match_date"] = remaining["date"] + pd.to_timedelta(delta, unit="D")
        res_tmp = res.rename(columns={"date": "match_date"}).copy()

        j = remaining.merge(
            res_tmp,
            on=["match_date", "home", "away", "match_key"],
            how="left",
            suffixes=("", "_res")
        )
        j["match_type"] = label

        matched_parts.append(j)
        matched_ids.update(j["pred_row_id"].tolist())

    if not matched_parts:
        return pd.DataFrame()

    out = pd.concat(matched_parts, ignore_index=True)
    out = out.sort_values(["pred_row_id", "match_type"]).drop_duplicates(subset=["pred_row_id"], keep="first")
    return out.reset_index(drop=True)


def _actual_result(row) -> str:
    if pd.isna(row["home_pts"]) or pd.isna(row["away_pts"]):
        return "PENDING"
    if row["home_pts"] > row["away_pts"]:
        return "HOME"
    if row["away_pts"] > row["home_pts"]:
        return "AWAY"
    return "DRAW"


def _bet_odds(row):
    if row["pick"] == "HOME":
        return row["home_odds"]
    if row["pick"] == "AWAY":
        return row["away_odds"]
    return pd.NA


def _bet_profit_units(row):
    stake_units = float(row["stake"]) if pd.notna(row["stake"]) else 0.0

    if row["bet_status"] == "NO_BET":
        return 0.0
    if row["bet_status"] == "PENDING":
        return 0.0
    if row["bet_status"] == "DRAW":
        return -stake_units
    if row["bet_status"] == "WIN":
        if pd.isna(row["bet_odds"]):
            return 0.0
        return stake_units * (float(row["bet_odds"]) - 1.0)
    if row["bet_status"] == "LOSS":
        return -stake_units
    return 0.0


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame([{
        "start_bankroll": START_BANKROLL,
        "closing_bankroll": START_BANKROLL,
        "bets_total": 0,
        "bets_settled": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "pending": 0,
        "units_staked": 0.0,
        "units_profit": 0.0,
        "roi": 0.0,
        "yield_on_settled": 0.0
    }])


def main():
    pred = _load_predictions(PRED_HISTORY_PATH)
    res = _load_results(RESULTS_CACHE_PATH)

    if pred.empty:
        print("No usable predictions history found.")
        pd.DataFrame().to_csv(BET_HISTORY_OUT, index=False)
        _empty_summary().to_csv(BET_SUMMARY_OUT, index=False)
        return

    pred = _latest_prediction_per_match(pred)
    merged = _match_results_with_fallback(pred, res)

    if merged.empty:
        print("No matches available after merge.")
        pd.DataFrame().to_csv(BET_HISTORY_OUT, index=False)
        _empty_summary().to_csv(BET_SUMMARY_OUT, index=False)
        return

    merged["bet_odds"] = merged.apply(_bet_odds, axis=1)

    def _status(row):
        pick = str(row["pick"]).strip().upper()
        stake = float(row["stake"]) if pd.notna(row["stake"]) else 0.0
        actual = _actual_result(row)

        if pick not in {"HOME", "AWAY"} or stake <= 0:
            return "NO_BET"
        if actual == "PENDING":
            return "PENDING"
        if actual == "DRAW":
            return "DRAW"
        if pick == actual:
            return "WIN"
        return "LOSS"

    merged["actual_result"] = merged.apply(_actual_result, axis=1)
    merged["bet_status"] = merged.apply(_status, axis=1)
    merged["profit_units"] = merged.apply(_bet_profit_units, axis=1)

    bets = merged.copy()

    kickoff_text = bets["kickoff_local"].astype(str).str.strip()
    bets["sort_kickoff"] = pd.to_datetime(
        bets["date"].dt.strftime("%Y-%m-%d") + " " + kickoff_text,
        errors="coerce"
    )

    missing_sort = bets["sort_kickoff"].isna()
    bets.loc[missing_sort, "sort_kickoff"] = pd.to_datetime(
        bets.loc[missing_sort, "date"],
        errors="coerce"
    )

    bets = bets.sort_values(["sort_kickoff", "date", "home", "away"]).reset_index(drop=True)

    bankroll = START_BANKROLL
    bankroll_after = []

    for _, row in bets.iterrows():
        if row["bet_status"] in {"WIN", "LOSS", "DRAW"}:
            bankroll += float(row["profit_units"])
        bankroll_after.append(round(bankroll, 2))

    bets["bankroll_after"] = bankroll_after

    out_cols = [
        "run_id", "date", "kickoff_local", "home", "away",
        "home_win_prob", "exp_margin_home",
        "home_odds", "away_odds", "pick", "bet_odds",
        "edge", "stake", "stake_units", "stake_dollars", "recommended_bet",
        "actual_result", "home_pts", "away_pts",
        "match_type", "bet_status", "profit_units", "bankroll_after",
        "generated_at"
    ]

    for col in out_cols:
        if col not in bets.columns:
            bets[col] = pd.NA

    bets[out_cols].to_csv(BET_HISTORY_OUT, index=False)

    bet_rows = bets[bets["bet_status"] != "NO_BET"].copy()
    settled = bet_rows[bet_rows["bet_status"].isin({"WIN", "LOSS", "DRAW"})].copy()

    units_staked = float(settled["stake"].sum()) if not settled.empty else 0.0
    units_profit = float(settled["profit_units"].sum()) if not settled.empty else 0.0
    roi = (units_profit / units_staked) if units_staked > 0 else 0.0

    closing_bankroll = START_BANKROLL + units_profit

    summary = pd.DataFrame([{
        "start_bankroll": START_BANKROLL,
        "closing_bankroll": round(closing_bankroll, 2),
        "bets_total": int(len(bet_rows)),
        "bets_settled": int(len(settled)),
        "wins": int((settled["bet_status"] == "WIN").sum()),
        "losses": int((settled["bet_status"] == "LOSS").sum()),
        "draws": int((settled["bet_status"] == "DRAW").sum()),
        "pending": int((bet_rows["bet_status"] == "PENDING").sum()),
        "units_staked": round(units_staked, 4),
        "units_profit": round(units_profit, 4),
        "roi": round(roi, 6),
        "yield_on_settled": round(roi, 6)
    }])

    summary.to_csv(BET_SUMMARY_OUT, index=False)

    print(
        f"Bets total: {len(bet_rows)} | "
        f"Settled: {len(settled)} | "
        f"Wins: {(settled['bet_status'] == 'WIN').sum()} | "
        f"Losses: {(settled['bet_status'] == 'LOSS').sum()} | "
        f"Profit: {units_profit:.2f}u | "
        f"ROI: {roi:.2%} | "
        f"Bankroll: ${closing_bankroll:.2f}"
    )


if __name__ == "__main__":
    main()
