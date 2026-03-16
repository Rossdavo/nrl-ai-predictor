import os
import pandas as pd

BET_HISTORY_PATH = "bet_history.csv"
BET_LOG_PATH = "bet_log.csv"
CLOSING_ODDS_PATH = "closing_odds.csv"

CLV_HISTORY_OUT = "clv_history.csv"
CLV_SUMMARY_OUT = "clv_summary.csv"


def _load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Could not read {path}: {e}")
        return pd.DataFrame()


def _norm(s: str) -> str:
    return " ".join(str(s).strip().upper().split())


def _empty_outputs(msg: str) -> None:
    pd.DataFrame(columns=[
        "date", "home", "away", "pick",
        "bet_odds", "closing_odds",
        "bet_implied_prob", "closing_implied_prob",
        "clv_diff", "clv_edge", "positive_clv"
    ]).to_csv(CLV_HISTORY_OUT, index=False)

    pd.DataFrame([{
        "bets_with_closing_odds": 0,
        "avg_clv_diff": 0.0,
        "avg_clv_edge": 0.0,
        "positive_clv_rate": 0.0
    }]).to_csv(CLV_SUMMARY_OUT, index=False)

    print(msg)


def _load_bets() -> pd.DataFrame:
    """
    Prefer new pipeline: bet_history.csv
    Fall back to old pipeline: bet_log.csv
    """
    bets = _load_csv_safe(BET_HISTORY_PATH)
    if not bets.empty:
        required = {"date", "home", "away", "pick", "bet_odds"}
        if required.issubset(set(bets.columns)):
            bets = bets.copy()
            bets["date"] = pd.to_datetime(bets["date"], errors="coerce").dt.normalize()
            bets["home"] = bets["home"].map(_norm)
            bets["away"] = bets["away"].map(_norm)
            bets["pick"] = bets["pick"].astype(str).str.strip().str.upper()
            bets["bet_odds"] = pd.to_numeric(bets["bet_odds"], errors="coerce")
            return bets.dropna(subset=["date", "home", "away", "pick", "bet_odds"]).copy()

    bets = _load_csv_safe(BET_LOG_PATH)
    if bets.empty:
        return pd.DataFrame()

    required = {"date", "home", "away", "side", "odds_taken"}
    if not required.issubset(set(bets.columns)):
        print("[warn] bet_log.csv missing required columns for CLV fallback.")
        return pd.DataFrame()

    bets = bets.copy()
    bets["date"] = pd.to_datetime(bets["date"], errors="coerce").dt.normalize()
    bets["home"] = bets["home"].map(_norm)
    bets["away"] = bets["away"].map(_norm)
    bets["pick"] = bets["side"].astype(str).str.strip().str.upper()
    bets["bet_odds"] = pd.to_numeric(bets["odds_taken"], errors="coerce")

    return bets.dropna(subset=["date", "home", "away", "pick", "bet_odds"]).copy()


def _load_closing() -> pd.DataFrame:
    closing = _load_csv_safe(CLOSING_ODDS_PATH)
    if closing.empty:
        return pd.DataFrame()

    closing = closing.copy()
    closing["date"] = pd.to_datetime(closing["date"], errors="coerce").dt.normalize()
    closing["home"] = closing["home"].map(_norm)
    closing["away"] = closing["away"].map(_norm)

    # Support multiple naming styles
    if "closing_home_odds" not in closing.columns and "home_odds_close" in closing.columns:
        closing["closing_home_odds"] = closing["home_odds_close"]
    if "closing_away_odds" not in closing.columns and "away_odds_close" in closing.columns:
        closing["closing_away_odds"] = closing["away_odds_close"]

    if "closing_home_odds" not in closing.columns and "home_odds" in closing.columns:
        closing["closing_home_odds"] = closing["home_odds"]
    if "closing_away_odds" not in closing.columns and "away_odds" in closing.columns:
        closing["closing_away_odds"] = closing["away_odds"]

    need = {"date", "home", "away", "closing_home_odds", "closing_away_odds"}
    if not need.issubset(set(closing.columns)):
        print("[warn] closing_odds.csv missing required columns.")
        return pd.DataFrame()

    closing["closing_home_odds"] = pd.to_numeric(closing["closing_home_odds"], errors="coerce")
    closing["closing_away_odds"] = pd.to_numeric(closing["closing_away_odds"], errors="coerce")

    return closing.dropna(subset=["date", "home", "away"]).copy()


def _pick_closing_odds(row):
    side = str(row.get("pick", "")).upper().strip()
    if side == "HOME":
        return row.get("closing_home_odds")
    if side == "AWAY":
        return row.get("closing_away_odds")
    return pd.NA


def main():
    bets = _load_bets()
    closing = _load_closing()

    if bets.empty:
        _empty_outputs("No usable bet history / bet log yet — skipping CLV.")
        return

    if closing.empty:
        _empty_outputs("No usable closing odds yet — skipping CLV.")
        return

    merged = bets.merge(closing, on=["date", "home", "away"], how="left")
    merged["closing_odds"] = merged.apply(_pick_closing_odds, axis=1)

    merged["bet_odds"] = pd.to_numeric(merged["bet_odds"], errors="coerce")
    merged["closing_odds"] = pd.to_numeric(merged["closing_odds"], errors="coerce")

    out = merged.dropna(subset=["bet_odds", "closing_odds"]).copy()

    if out.empty:
        _empty_outputs("No matched closing odds for tracked bets.")
        return

    out["bet_implied_prob"] = 1.0 / out["bet_odds"]
    out["closing_implied_prob"] = 1.0 / out["closing_odds"]

    # Positive = you beat the close
    out["clv_diff"] = out["bet_odds"] - out["closing_odds"]
    out["clv_edge"] = out["closing_implied_prob"] - out["bet_implied_prob"]
    out["positive_clv"] = (out["clv_edge"] > 0).astype(int)

    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    out_cols = [
        "date", "home", "away", "pick",
        "bet_odds", "closing_odds",
        "bet_implied_prob", "closing_implied_prob",
        "clv_diff", "clv_edge", "positive_clv"
    ]
    out[out_cols].to_csv(CLV_HISTORY_OUT, index=False)

    summary = pd.DataFrame([{
        "bets_with_closing_odds": int(len(out)),
        "avg_clv_diff": round(float(out["clv_diff"].mean()), 6),
        "avg_clv_edge": round(float(out["clv_edge"].mean()), 6),
        "positive_clv_rate": round(float(out["positive_clv"].mean()), 6)
    }])
    summary.to_csv(CLV_SUMMARY_OUT, index=False)

    print(
        f"CLV updated | "
        f"bets={len(out)} | "
        f"avg_clv_diff={out['clv_diff'].mean():.3f} | "
        f"avg_clv_edge={out['clv_edge'].mean():.4f} | "
        f"positive_clv_rate={out['positive_clv'].mean():.1%}"
    )


if __name__ == "__main__":
    main()
