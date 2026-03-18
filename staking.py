import pandas as pd

PRED_PATH = "predictions.csv"
OUT_PATH = "bet_log.csv"

BANKROLL = 200.0
UNIT_PCT = 0.05
UNIT_SIZE = round(BANKROLL * UNIT_PCT, 2)


def safe_numeric(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def main():
    df = pd.read_csv(PRED_PATH)

    # Prefer direct dollar stakes from predict.py
    if "stake_dollars" in df.columns:
        df["stake_dollars"] = safe_numeric(df["stake_dollars"])
    else:
        if "stake" not in df.columns:
            df["stake"] = 0.0
        df["stake"] = safe_numeric(df["stake"])
        df["stake_dollars"] = df["stake"] * UNIT_SIZE

    if "stake_units" not in df.columns:
        if "stake" in df.columns:
            df["stake_units"] = safe_numeric(df["stake"])
        else:
            df["stake_units"] = (df["stake_dollars"] / UNIT_SIZE).round(2) if UNIT_SIZE > 0 else 0.0
    else:
        df["stake_units"] = safe_numeric(df["stake_units"])

    # Ensure expected columns exist
    for col, default in [
        ("pick", ""),
        ("home", ""),
        ("away", ""),
        ("date", ""),
        ("home_odds", 0.0),
        ("away_odds", 0.0),
        ("edge", 0.0),
        ("confidence", 0.0),
        ("model_home_win_prob", pd.NA),
        ("market_home_win_prob", pd.NA),
        ("final_home_win_prob", pd.NA),
        ("favourite_team", ""),
        ("underdog_team", ""),
        ("upset_flag", 0),
        ("final_upset_score", 0.0),
        ("fragile_favourite", 0),
        ("recommended_bet", "No Bet"),
    ]:
        if col not in df.columns:
            df[col] = default

    df["home_odds"] = safe_numeric(df["home_odds"])
    df["away_odds"] = safe_numeric(df["away_odds"])
    df["edge"] = safe_numeric(df["edge"])
    df["confidence"] = safe_numeric(df["confidence"])
    df["final_upset_score"] = safe_numeric(df["final_upset_score"])
    df["upset_flag"] = safe_numeric(df["upset_flag"]).astype(int)
    df["fragile_favourite"] = safe_numeric(df["fragile_favourite"]).astype(int)

    # Only real bets
    bets = df[
        (df["stake_dollars"] > 0) &
        (df["pick"].astype(str).str.strip().isin(["HOME", "AWAY"]))
    ].copy()

    if bets.empty:
        print("0 bets generated")
        empty_cols = [
            "date",
            "home",
            "away",
            "pick",
            "selection",
            "odds",
            "stake_units",
            "stake_dollars",
            "edge",
            "confidence",
            "favourite_team",
            "underdog_team",
            "upset_flag",
            "final_upset_score",
            "fragile_favourite",
            "recommended_bet",
        ]
        pd.DataFrame(columns=empty_cols).to_csv(OUT_PATH, index=False)
        return

    bets["selection"] = bets.apply(
        lambda r: r["home"] if r["pick"] == "HOME" else r["away"],
        axis=1,
    )

    bets["odds"] = bets.apply(
        lambda r: r["home_odds"] if r["pick"] == "HOME" else r["away_odds"],
        axis=1,
    )
    bets["odds"] = safe_numeric(bets["odds"])

    out_cols = [
        "date",
        "home",
        "away",
        "pick",
        "selection",
        "odds",
        "stake_units",
        "stake_dollars",
        "edge",
        "confidence",
        "favourite_team",
        "underdog_team",
        "upset_flag",
        "final_upset_score",
        "fragile_favourite",
        "recommended_bet",
    ]

    out = bets[out_cols].copy()
    out = out.sort_values(["date", "selection"]).reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"{len(out)} bets generated")
    print(f"Bankroll=${BANKROLL:.2f} | Unit=${UNIT_SIZE:.2f}")
    print(f"Total round exposure=${out['stake_dollars'].sum():.2f}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
