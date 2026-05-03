import os
import pandas as pd

PRED_HISTORY_PATH = "predictions_history.csv"
RESULTS_CACHE_PATH = "results_cache.csv"

BET_HISTORY_OUT = "bet_history.csv"
BET_SUMMARY_OUT = "bet_summary.csv"
BET_SUMMARY_BY_ROUND_OUT = "bet_summary_by_round.csv"

START_BANKROLL = float(os.getenv("BANKROLL", "200"))


TEAM_ALIASES = {
    "SEA EAGLES": "MANLY",
    "MANLY SEA EAGLES": "MANLY",
    "WESTS TIGERS": "TIGERS",
    "WEST TIGERS": "TIGERS",
    "ST GEORGE ILLAWARRA DRAGONS": "DRAGONS",
    "ST GEORGE DRAGONS": "DRAGONS",
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
        print("[warn] predictions history missing probability column")
        return pd.DataFrame()

    required = {
        "run_id",
        "date",
        "home",
        "away",
        "pick",
        "stake",
        "stake_dollars",
        "home_odds",
        "away_odds",
        "generated_at",
    }

    missing = required - set(df.columns)
    if missing:
        print(f"[warn] predictions history missing columns: {sorted(missing)}")
        return pd.DataFrame()

    df = df.copy()

    optional_cols = [
        "kickoff_local",
        "edge",
        "exp_margin_home",
        "recommended_bet",
        "stake_units",
        "mode",
        "rating_mode",
        "confidence",
        "favourite_team",
        "underdog_team",
        "upset_flag",
        "final_upset_score",
        "fragile_favourite",
    ]

    for col in optional_cols:
        if col not in df.columns:
            df[col] = pd.NA

    keep_cols = [
        "run_id",
        "date",
        "home",
        "away",
        "pick",
        "stake",
        "stake_dollars",
        "home_odds",
        "away_odds",
        "generated_at",
        "kickoff_local",
        "edge",
        "exp_margin_home",
        "recommended_bet",
        "stake_units",
        "mode",
        "rating_mode",
        "confidence",
        "favourite_team",
        "underdog_team",
        "upset_flag",
        "final_upset_score",
        "fragile_favourite",
        prob_col,
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
        "stake",
        "stake_dollars",
        "home_odds",
        "away_odds",
        "edge",
        "home_win_prob",
        "exp_margin_home",
        "stake_units",
        "confidence",
        "upset_flag",
        "final_upset_score",
        "fragile_favourite",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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


def _match_results_with_fallback(pred: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy().reset_index(drop=True)
    res = res.copy().reset_index(drop=True)

    pred["pred_row_id"] = pred.index
    pred["match_key"] = pred["home"] + "||" + pred["away"]
    res["match_key"] = res["home"] + "||" + res["away"]

    res_small = res[["date", "home", "away", "match_key", "home_pts", "away_pts"]].copy()
    all_matches = []

    for delta in [0, -1, 1, -2, 2]:
        tmp = pred[["pred_row_id", "date", "home", "away", "match_key"]].copy()
        tmp["target_date"] = tmp["date"] + pd.to_timedelta(delta, unit="D")

        joined = tmp.merge(
            res_small,
            left_on=["target_date", "home", "away", "match_key"],
            right_on=["date", "home", "away", "match_key"],
            how="inner",
            suffixes=("", "_res"),
        )

        if not joined.empty:
            joined["match_type"] = (
                "exact"
                if delta == 0
                else f"minus{abs(delta)}"
                if delta < 0
                else f"plus{delta}"
            )
            joined["date_distance"] = abs(delta)
            joined = joined.rename(columns={"date_res": "result_date"})
            all_matches.append(joined)

    if all_matches:
        candidates = pd.concat(all_matches, ignore_index=True)
        candidates = candidates.sort_values(["pred_row_id", "date_distance", "result_date"])
        best = candidates.drop_duplicates(subset=["pred_row_id"], keep="first").copy()
    else:
        best = pd.DataFrame(
            columns=["pred_row_id", "result_date", "home_pts", "away_pts", "match_type", "date_distance"]
        )

    out = pred.merge(
        best[["pred_row_id", "result_date", "home_pts", "away_pts", "match_type", "date_distance"]],
        on="pred_row_id",
        how="left",
    )

    unresolved = out["home_pts"].isna() | out["away_pts"].isna()
    unresolved_count = int(unresolved.sum())

    print(
        f"[debug] result matching: predictions={len(pred)} | "
        f"results={len(res)} | matched={len(pred) - unresolved_count} | "
        f"unmatched={unresolved_count}"
    )

    if unresolved_count > 0:
        print("[debug] Unmatched predictions:")
        for _, row in out.loc[unresolved, ["date", "home", "away"]].head(30).iterrows():
            try:
                date_text = row["date"].date()
            except Exception:
                date_text = row["date"]
            print(f"  - {date_text} {row['home']} vs {row['away']}")

    return out


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
    stake_units = (
        float(row["stake_units"])
        if pd.notna(row["stake_units"])
        else float(row["stake"])
        if pd.notna(row["stake"])
        else 0.0
    )

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
    return pd.DataFrame(
        [
            {
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
                "yield_on_settled": 0.0,
            }
        ]
    )


def main():
    pred = _load_predictions(PRED_HISTORY_PATH)
    res = _load_results(RESULTS_CACHE_PATH)

    if pred.empty:
        print("No usable predictions history found.")
        pd.DataFrame().to_csv(BET_HISTORY_OUT, index=False)
        _empty_summary().to_csv(BET_SUMMARY_OUT, index=False)
        return

    if res.empty:
        print("No usable results found.")
        pd.DataFrame().to_csv(BET_HISTORY_OUT, index=False)
        _empty_summary().to_csv(BET_SUMMARY_OUT, index=False)
        return

    # IMPORTANT:
    # Do NOT collapse to latest prediction per match.
    # We want to settle every historical run_id so round-by-round summaries are accurate.
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
        errors="coerce",
    )

    missing_sort = bets["sort_kickoff"].isna()
    bets.loc[missing_sort, "sort_kickoff"] = pd.to_datetime(
        bets.loc[missing_sort, "date"],
        errors="coerce",
    )

    bets = bets.sort_values(["sort_kickoff", "date", "home", "away", "run_id"]).reset_index(drop=True)

    bankroll = START_BANKROLL
    bankroll_after = []

    for _, row in bets.iterrows():
        if row["bet_status"] in {"WIN", "LOSS", "DRAW"}:
            bankroll += float(row["profit_units"])
        bankroll_after.append(round(bankroll, 2))

    bets["bankroll_after"] = bankroll_after

    out_cols = [
        "run_id",
        "date",
        "kickoff_local",
        "home",
        "away",
        "home_win_prob",
        "exp_margin_home",
        "home_odds",
        "away_odds",
        "pick",
        "bet_odds",
        "edge",
        "stake",
        "stake_units",
        "stake_dollars",
        "recommended_bet",
        "actual_result",
        "home_pts",
        "away_pts",
        "result_date",
        "match_type",
        "date_distance",
        "bet_status",
        "profit_units",
        "bankroll_after",
        "generated_at",
    ]

    for col in out_cols:
        if col not in bets.columns:
            bets[col] = pd.NA

    bets[out_cols].to_csv(BET_HISTORY_OUT, index=False)

    bet_rows = bets[bets["bet_status"] != "NO_BET"].copy()
    settled = bet_rows[bet_rows["bet_status"].isin({"WIN", "LOSS", "DRAW"})].copy()

    units_staked = float(settled["stake_units"].sum()) if not settled.empty else 0.0
    units_profit = float(settled["profit_units"].sum()) if not settled.empty else 0.0
    roi = (units_profit / units_staked) if units_staked > 0 else 0.0

    closing_bankroll = START_BANKROLL + units_profit

    summary = pd.DataFrame(
        [
            {
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
                "yield_on_settled": round(roi, 6),
            }
        ]
    )

    summary.to_csv(BET_SUMMARY_OUT, index=False)

    round_summary = (
        bet_rows.groupby("run_id", dropna=False)
        .agg(
            first_game_date=("date", "min"),
            last_game_date=("date", "max"),
            bets_total=("bet_status", "count"),
            bets_settled=("bet_status", lambda s: s.isin(["WIN", "LOSS", "DRAW"]).sum()),
            wins=("bet_status", lambda s: (s == "WIN").sum()),
            losses=("bet_status", lambda s: (s == "LOSS").sum()),
            draws=("bet_status", lambda s: (s == "DRAW").sum()),
            pending=("bet_status", lambda s: (s == "PENDING").sum()),
            units_staked=("stake_units", "sum"),
            units_profit=("profit_units", "sum"),
            dollars_staked=("stake_dollars", "sum"),
        )
        .reset_index()
    )

    round_summary["roi"] = round_summary.apply(
        lambda r: r["units_profit"] / r["units_staked"] if r["units_staked"] > 0 else 0.0,
        axis=1,
    )

    round_summary["first_game_date"] = pd.to_datetime(
        round_summary["first_game_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    round_summary["last_game_date"] = pd.to_datetime(
        round_summary["last_game_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    round_summary = round_summary.sort_values(["first_game_date", "run_id"]).reset_index(drop=True)
    round_summary.to_csv(BET_SUMMARY_BY_ROUND_OUT, index=False)

    print(
        f"Bets total: {len(bet_rows)} | "
        f"Settled: {len(settled)} | "
        f"Wins: {(settled['bet_status'] == 'WIN').sum()} | "
        f"Losses: {(settled['bet_status'] == 'LOSS').sum()} | "
        f"Draws: {(settled['bet_status'] == 'DRAW').sum()} | "
        f"Pending: {(bet_rows['bet_status'] == 'PENDING').sum()} | "
        f"Profit: {units_profit:.2f}u | "
        f"ROI: {roi:.2%} | "
        f"Bankroll: ${closing_bankroll:.2f}"
    )

    print(f"[info] Wrote {BET_HISTORY_OUT}")
    print(f"[info] Wrote {BET_SUMMARY_OUT}")
    print(f"[info] Wrote {BET_SUMMARY_BY_ROUND_OUT}")


if __name__ == "__main__":
    main()
