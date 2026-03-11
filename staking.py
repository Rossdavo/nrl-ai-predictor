import pandas as pd

PRED_PATH = "predictions.csv"
OUT_PATH = "bet_log.csv"

BANKROLL = 200.0
UNIT_PCT = 0.05
UNIT_SIZE = round(BANKROLL * UNIT_PCT, 2)


def main():
    df = pd.read_csv(PRED_PATH)

    # Prefer dollar stake if already produced by predict.py
    if "stake_dollars" in df.columns:
        df["stake_dollars"] = pd.to_numeric(df["stake_dollars"], errors="coerce").fillna(0.0)
    else:
        # Fall back to converting stake units into dollars
        if "stake" not in df.columns:
            df["stake"] = 0.0
        else:
            df["stake"] = pd.to_numeric(df["stake"], errors="coerce").fillna(0.0)

        df["stake_dollars"] = df["stake"] * UNIT_SIZE

    # Ensure pick column exists
    if "pick" not in df.columns:
        df["pick"] = ""

    # Only bets where stake > 0 and pick is set
    bets = df[(df["stake_dollars"] > 0) & (df["pick"].astype(str).str.len() > 0)].copy()

    if bets.empty:
        print("0 bets generated")
        empty_cols = [
            "date", "home", "away", "pick", "selection",
            "odds", "stake_dollars", "edge", "confidence"
        ]
        pd.DataFrame(columns=empty_cols).to_csv(OUT_PATH, index=False)
        return

    # Determine selection team + odds
    bets["selection"] = bets.apply(
        lambda r: r["home"] if r["pick"] == "HOME" else r["away"],
        axis=1
    )

    bets["odds"] = bets.apply(
        lambda r: r["home_odds"] if r["pick"] == "HOME" else r["away_odds"],
        axis=1
    )

    # Clean numeric columns
    bets["odds"] = pd.to_numeric(bets["odds"], errors="coerce")
    bets["edge"] = pd.to_numeric(bets.get("edge", 0), errors="coerce").fillna(0.0)
    bets["confidence"] = pd.to_numeric(bets.get("confidence", 0), errors="coerce").fillna(0.0)
    bets["stake_dollars"] = pd.to_numeric(bets["stake_dollars"], errors="coerce").fillna(0.0)

    out = bets[
        ["date", "home", "away", "pick", "selection", "odds", "stake_dollars", "edge", "confidence"]
    ].copy()

    out.to_csv(OUT_PATH, index=False)

    print(f"{len(out)} bets generated")
    print(f"Bankroll=${BANKROLL:.2f} | Unit=${UNIT_SIZE:.2f}")
    print(f"Total round exposure=${out['stake_dollars'].sum():.2f}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
