import os
from datetime import datetime, timezone

import pandas as pd

PRED_PATH = "predictions.csv"
ODDS_PATH = "odds.csv"

PRED_HIST = "predictions_history.csv"
ODDS_HIST = "odds_history.csv"


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def make_round_key(df: pd.DataFrame) -> str:
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])

    if work.empty:
        return make_run_id()

    start = work["date"].min().strftime("%Y-%m-%d")
    end = work["date"].max().strftime("%Y-%m-%d")
    return f"{start}_to_{end}"


def ensure_run_cols(df: pd.DataFrame, round_key: str) -> pd.DataFrame:
    out = df.copy()

    run_id = make_run_id()

    out["run_id"] = run_id
    out["run_utc"] = utc_now_str()
    out["round_key"] = round_key

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


def round_already_archived(history_path: str, round_key: str) -> bool:
    hist = load_csv_safe(history_path)

    if hist.empty:
        return False

    if "round_key" not in hist.columns:
        return False

    existing = set(hist["round_key"].astype(str).str.strip())
    return str(round_key).strip() in existing


def main():
    pred = load_csv_safe(PRED_PATH)
    odds = load_csv_safe(ODDS_PATH)

    if pred.empty:
        print("[archive] No predictions.csv found to archive.")
    else:
        round_key = make_round_key(pred)

        if round_already_archived(PRED_HIST, round_key):
            print(f"[archive] Round already archived ({round_key}) — skipping duplicate $200 allocation.")
        else:
            pred = ensure_run_cols(pred, round_key)
            append_deduped(
                PRED_HIST,
                pred,
                subset=["round_key", "date", "home", "away"],
            )
            print(f"[archive] Archived official betting round: {round_key}")

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
