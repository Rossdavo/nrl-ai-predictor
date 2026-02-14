import os
import pandas as pd
from datetime import datetime, timezone

PRED = "predictions.csv"
ODDS = "odds.csv"

PRED_HIST = "predictions_history.csv"
ODDS_HIST = "odds_history.csv"

def append_csv(src: str, dst: str, run_id: str):
    if not os.path.exists(src):
        return
    df = pd.read_csv(src)
    df.insert(0, "run_id", run_id)
    df.insert(1, "run_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    if os.path.exists(dst):
        df.to_csv(dst, mode="a", header=False, index=False)
    else:
        df.to_csv(dst, index=False)

def main():
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    append_csv(PRED, PRED_HIST, run_id)
    append_csv(ODDS, ODDS_HIST, run_id)
    print("Archived predictions + odds.")

if __name__ == "__main__":
    main()
