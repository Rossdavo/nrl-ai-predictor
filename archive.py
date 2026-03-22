import os
from datetime import datetime, timezone

import pandas as pd

PRED_PATH = "predictions.csv"
ODDS_PATH = "odds.csv"

PRED_HIST = "predictions_history.csv"
ODDS_HIST = "odds_history.csv"


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def ensure_run_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "run_id" not in out.columns:
        out["run_id"] = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    out["run_id"] = out["run_id"].astype(str).replace("", datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))

    if "run_utc" not in out.columns:
        out["run_utc"] = utc_now_str()
    out["run_utc"] = out["run_utc"].astype(str).replace("", utc_now_str())
    return out


def append_deduped(history_path: str, new_df: pd.DataFrame, subset: list[str]) -> None:
    hist = load_csv_safe(history_path)

    if hist.empty:
        combined = new_df.copy()
    else:
        combined = pd.concat([hist, new_df], ignore_index=True, sort=False)

    for col in subset:
        if col not in combined.columns:
            combined[col] = ""

    combined = combined.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
    combined.to_csv(history_path, index=False)


def main():
    pred = load_csv_safe(PRED_PATH)
    odds = load_csv_safe(ODDS_PATH)

    if not pred.empty:
        pred = ensure_run_cols(pred)
        append_deduped(
            PRED_HIST,
            pred,
            subset=["run_id", "date", "home", "away"],
        )

    if not odds.empty:
        odds = odds.copy()
        if "captured_at_utc" not in odds.columns:
            odds["captured_at_utc"] = utc_now_str()
        append_deduped(
            ODDS_HIST,
            odds,
            subset=["date", "home", "away", "captured_at_utc"],
        )

    print("Archived predictions + odds.")


if __name__ == "__main__":
    main()
