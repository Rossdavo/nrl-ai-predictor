import pandas as pd

PRED_PATH = "predictions.csv"
OUT_PATH = "bet_log.csv"

def main():
    df = pd.read_csv(PRED_PATH)

    # Make sure stake is numeric
    df["stake"] = pd.to_numeric(df.get("stake", 0), errors="coerce").fillna(0.0)

    # Only bets where stake > 0 and pick is set
    bets = df[(df["stake"] > 0) & (df.get("pick", "").astype(str).str.len() > 0)].copy()

    if bets.empty:
        print("0 bets generated")
        # Create an empty bet_log.csv with headers (helps downstream scripts)
        empty_cols = ["date", "home", "away", "pick", "selection", "odds", "stake", "edge", "confidence"]
        pd.DataFrame(columns=empty_cols).to_csv(OUT_PATH, index=False)
        return

    # Work out which team is the selection and which odds apply
    def selection_team(row):
        return row["home"] if row["pick"] == "HOME" else row["away"]

    def selection_odds(row):
        return row["home_odds"] if row["pick"] == "HOME" else row["away_odds"]

    bets["selection"] = bets.apply(selection_team, axis=1)
    bets["odds"] = bets.apply(selection_odds, axis=1)

    # Clean up types
    bets["odds"] = pd.to_numeric(bets["odds"], errors="coerce")
    bets["edge"] = pd.to_numeric(bets.get("edge", 0), errors="coerce").fillna(0.0)
    bets["confidence"] = pd.to_numeric(bets.get("confidence", 0), errors="coerce").fillna(0.0)

    out = bets[["date", "home", "away", "pick", "selection", "odds", "stake", "edge", "confidence"]].copy()
    out.to_csv(OUT_PATH, index=False)

    print(f"{len(out)} bets generated")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
