import os
import pandas as pd

BET_HISTORY_PATH = "bet_history.csv"
BET_LOG_PATH = "bet_log.csv"
RESULTS_CACHE_PATH = "results_cache.csv"
OUT_PATH = "performance.csv"


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "date",
        "home",
        "away",
        "bet_side",
        "odds",
        "stake",
        "profit",
        "result"
    ])


def _norm(s: str) -> str:
    return " ".join(str(s).strip().upper().split())


def _load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Could not read {path}: {e}")
        return pd.DataFrame()


def _from_bet_history() -> pd.DataFrame:
    df = _load_csv_safe(BET_HISTORY_PATH)
    if df.empty:
        return pd.DataFrame()

    required = {"date", "home", "away", "pick", "bet_odds", "stake", "profit_units", "bet_status"}
    if not required.issubset(df.columns):
        print(f"[warn] {BET_HISTORY_PATH} missing required columns.")
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["home"] = df["home"].map(_norm)
    df["away"] = df["away"].map(_norm)
    df["pick"] = df["pick"].astype(str).str.strip().str.upper()
    df["bet_odds"] = pd.to_numeric(df["bet_odds"], errors="coerce")
    df["stake"] = pd.to_numeric(df["stake"], errors="coerce")
    df["profit_units"] = pd.to_numeric(df["profit_units"], errors="coerce")

    df = df[df["bet_status"].isin(["WIN", "LOSS", "DRAW"])].copy()

    if df.empty:
        return _empty_output()

    df["bet_side"] = df.apply(
        lambda r: r["home"] if r["pick"] == "HOME" else (r["away"] if r["pick"] == "AWAY" else ""),
        axis=1
    )

    df["result"] = df["bet_status"].replace({"DRAW": "LOSS"})
    df["profit"] = df["profit_units"]
    df["odds"] = df["bet_odds"]

    out = df[["date", "home", "away", "bet_side", "odds", "stake", "profit", "result"]].copy()
    out["odds"] = out["odds"].round(2)
    out["stake"] = out["stake"].round(2)
    out["profit"] = out["profit"].round(2)

    return out.sort_values(["date", "home", "away"]).reset_index(drop=True)


def _from_bet_log_and_results() -> pd.DataFrame:
    bets = _load_csv_safe(BET_LOG_PATH)
    results = _load_csv_safe(RESULTS_CACHE_PATH)

    if bets.empty or results.empty:
        return pd.DataFrame()

    need_bets = {"date", "home", "away"}
    need_results = {"date", "home", "away", "home_pts", "away_pts"}

    if not need_bets.issubset(bets.columns):
        print(f"[warn] {BET_LOG_PATH} missing required columns.")
        return pd.DataFrame()

    if not need_results.issubset(results.columns):
        print(f"[warn] {RESULTS_CACHE_PATH} missing required columns.")
        return pd.DataFrame()

    bets = bets.copy()
    results = results.copy()

    bets["date"] = pd.to_datetime(bets["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    results["date"] = pd.to_datetime(results["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    bets["home"] = bets["home"].map(_norm)
    bets["away"] = bets["away"].map(_norm)
    results["home"] = results["home"].map(_norm)
    results["away"] = results["away"].map(_norm)

    if "bet_side" not in bets.columns:
        if "pick" in bets.columns:
            bets["bet_side"] = bets["pick"]
        elif "selection" in bets.columns:
            bets["bet_side"] = bets["selection"]
        else:
            bets["bet_side"] = ""

    if "odds" not in bets.columns:
        if "odds_taken" in bets.columns:
            bets["odds"] = bets["odds_taken"]
        elif "home_odds" in bets.columns and "away_odds" in bets.columns:
            bets["odds"] = bets.apply(
                lambda r: r["home_odds"] if str(r["bet_side"]).strip().upper() in {"HOME", str(r["home"]).strip().upper()} else r.get("away_odds", pd.NA),
                axis=1
            )
        else:
            bets["odds"] = pd.NA

    if "stake" not in bets.columns:
        if "stake_dollars" in bets.columns:
            bets["stake"] = bets["stake_dollars"]
        else:
            bets["stake"] = 0

    bets["bet_side"] = bets["bet_side"].astype(str).str.strip()
    bets["odds"] = pd.to_numeric(bets["odds"], errors="coerce")
    bets["stake"] = pd.to_numeric(bets["stake"], errors="coerce")

    results["home_pts"] = pd.to_numeric(results["home_pts"], errors="coerce")
    results["away_pts"] = pd.to_numeric(results["away_pts"], errors="coerce")

    bets = bets.dropna(subset=["date", "home", "away", "odds", "stake"])
    results = results.dropna(subset=["date", "home", "away", "home_pts", "away_pts"])

    merged = bets.merge(
        results[["date", "home", "away", "home_pts", "away_pts"]],
        on=["date", "home", "away"],
        how="inner"
    )

    rows = []

    for _, r in merged.iterrows():
        home = r["home"]
        away = r["away"]
        bet_side_raw = str(r.get("bet_side", "")).strip()
        bet_side_upper = bet_side_raw.upper()

        if bet_side_upper == "HOME":
            bet_team = home
        elif bet_side_upper == "AWAY":
            bet_team = away
        else:
            bet_team = bet_side_raw.upper()

        stake = float(r.get("stake", 0))
        odds = float(r.get("odds", 0))
        home_pts = float(r.get("home_pts", 0))
        away_pts = float(r.get("away_pts", 0))

        if home_pts > away_pts:
            winner = home
        elif away_pts > home_pts:
            winner = away
        else:
            winner = "DRAW"

        if winner == "DRAW":
            profit = -stake
            result = "LOSS"
        elif bet_team == winner:
            profit = stake * (odds - 1)
            result = "WIN"
        else:
            profit = -stake
            result = "LOSS"

        rows.append({
            "date": r["date"],
            "home": home,
            "away": away,
            "bet_side": bet_team,
            "odds": round(odds, 2),
            "stake": round(stake, 2),
            "profit": round(profit, 2),
            "result": result
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_output()

    return out.sort_values(["date", "home", "away"]).reset_index(drop=True)


def main():
    out = _from_bet_history()

    if out.empty:
        out = _from_bet_log_and_results()

    if out.empty:
        out = _empty_output()
        out.to_csv(OUT_PATH, index=False)
        print("performance.csv updated (0 settled bets)")
        return

    out.to_csv(OUT_PATH, index=False)
    print(f"performance.csv updated ({len(out)} settled bets)")


if __name__ == "__main__":
    main()
