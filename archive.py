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
    except Exception as e:
        print(f"[archive] Warning: could not read {path}: {e}")
        return pd.DataFrame()


def make_round_key(df: pd.DataFrame) -> str:
    if df.empty or "date" not in df.columns:
        return make_run_id()

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

    if "run_id" not in out.columns or out["run_id"].astype(str).str.strip().eq("").all():
        out["run_id"] = make_run_id()
    else:
        out["run_id"] = out["run_id"].astype(str).str.strip()

    if "run_utc" not in out.columns or out["run_utc"].astype(str).str.strip().eq("").all():
        out["run_utc"] = utc_now_str()

    out["round_key"] = round_key
    return out


def append_deduped(history_path: str, new_df: pd.DataFrame, subset: list[str]) -> None:
    if new_df.empty:
        return

    hist = load_csv_safe(history_path)
    combined = new_df.copy() if hist.empty else pd.concat([hist, new_df], ignore_index=True, sort=False)

    for col in subset:
        if col not in combined.columns:
            combined[col] = ""

    combined = combined.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
    combined.to_csv(history_path, index=False)


def archive_predictions(pred: pd.DataFrame) -> None:
    if pred.empty:
        print("[archive] No predictions.csv found to archive.")
        return

    required = {"date", "home", "away"}
    missing = required - set(pred.columns)
    if missing:
        print(f"[archive] predictions.csv missing required columns: {sorted(missing)}")
        return

    round_key = make_round_key(pred)
    pred = ensure_run_cols(pred, round_key)

    append_deduped(
        PRED_HIST,
        pred,
        subset=["run_id", "date", "home", "away"],
    )

    run_ids = pred["run_id"].astype(str).dropna().unique().tolist()
    run_id = run_ids[0] if run_ids else "unknown"

    print(
        f"[archive] Archived prediction snapshot: "
        f"round={round_key} run_id={run_id} rows={len(pred)}"
    )


def archive_odds(odds: pd.DataFrame) -> None:
    if odds.empty:
        print("[archive] No odds.csv found to archive.")
        return

    required = {"date", "home", "away"}
    missing = required - set(odds.columns)
    if missing:
        print(f"[archive] odds.csv missing required columns: {sorted(missing)}")
        return

    odds = odds.copy()

    if "captured_at_utc" not in odds.columns:
        odds["captured_at_utc"] = utc_now_str()
    else:
        odds["captured_at_utc"] = odds["captured_at_utc"].fillna("").astype(str)
        blank_mask = odds["captured_at_utc"].str.strip().eq("")
        if blank_mask.any():
            odds.loc[blank_mask, "captured_at_utc"] = utc_now_str()

    append_deduped(
        ODDS_HIST,
        odds,
        subset=["date", "home", "away", "captured_at_utc"],
    )

    print(f"[archive] Archived odds snapshot: rows={len(odds)}")


def main():
    pred = load_csv_safe(PRED_PATH)
    odds = load_csv_safe(ODDS_PATH)

    archive_predictions(pred)
    archive_odds(odds)

    print("[archive] Prediction + odds archive complete.")


if __name__ == "__main__":
    main()
