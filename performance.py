import os
import pandas as pd

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
    return str(s).strip()


def main():
    if not os.path.exists(BET_LOG_PATH) or not os.path.exists(RESULTS_CACHE_PATH):
        _empty_output().to_csv(OUT_PATH, index=False)
        print("performance.csv created (no settled bets yet)")
        return

    try:
        bets = pd.read_csv(BET_LOG_PATH)
    except Exception as e:
        print(f"[warn] Could not read {BET_LOG_PATH}: {e}")
        _empty_output().to_csv(OUT_PATH, index=False)
        print("performance.csv created (no settled bets yet)")
        return

    try:
        results = pd.read_csv(RESULTS_CACHE_PATH)
    except Exception as e:
        print(f"[warn] Could not read {RESULTS_CACHE_PATH}: {e}")
        _empty_output().to_csv(OUT_PATH, index=False)
        print("performance.csv created (no settled bets yet)")
        return

    if bets.empty or results.empty:
        _empty_output().to_csv(OUT_PATH, index=False)
        print("performance.csv created (no settled bets yet)")
        return

    need_bets = {"date", "home", "away"}
    need_results = {"date", "home", "away", "home_pts", "away_pts"}

    if not need_bets.issubset(bets.columns):
        print(f"[warn] {BET_LOG_PATH} missing required columns.")
        _empty_output().to_csv(OUT_PATH, index=False)
        print("performance.csv created (no settled bets yet)")
        return

    if not need_results.issubset(results.columns):
        print(f"[warn] {RESULTS_CACHE_PATH} missing required columns.")
        _empty_output().to_csv(OUT_PATH, index=False)
        print("performance.csv created (no settled bets yet)")
        return

    bets = bets.copy()
    results = results.copy()

    bets["date"] = pd.to_datetime(bets["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    results["date"] = pd.to_datetime(results["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    bets["home"] = bets["home"].map(_norm)
    bets["away"] = bets["away"].map(_norm)
    results["home"] = results["home"].map(_norm)
    results["away"] = results["away"].map(_norm)

    # Flexible handling of likely bet_log column names
    if "bet_side" not in bets.columns:
        if "pick" in bets.columns:
            bets["bet_side"] = bets["pick"]
        elif "selection" in bets.columns:
            bets["bet_side"] = bets["selection"]
        else:
            bets["bet_side"] = ""

    if "odds" not in bets.columns:
        if "home_odds" in bets.columns and "bet_side" in bets.columns:
            bets["odds"] = bets.apply(
                lambda r: r["home_odds"] if str(r["bet_side"]).strip().upper() in {"HOME", str(r["home"]).strip()} else r.get("away_odds", pd.NA),
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

        # Convert bet side into actual team name when needed
        if bet_side_upper == "HOME":
            bet_team = home
        elif bet_side_upper == "AWAY":
            bet_team = away
        else:
            bet_team = bet_side_raw

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
        out = _empty_output()

    out.to_csv(OUT_PATH, index=False)
    print(f"performance.csv updated ({len(out)} settled bets)")


if __name__ == "__main__":
    main()
