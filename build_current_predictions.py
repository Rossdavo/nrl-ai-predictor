import os
import csv
import pandas as pd

PRED_HISTORY_PATH = "predictions_history.csv"
CURRENT_OUT = "predictions.csv"

EXPECTED_COLS = [
    "run_id",
    "run_utc",
    "mode",
    "rating_mode",
    "date",
    "kickoff_local",
    "venue",
    "home",
    "away",
    "model_home_win_prob",
    "market_home_win_prob",
    "final_home_win_prob",
    "exp_margin_home",
    "exp_total",
    "confidence",
    "home_odds",
    "away_odds",
    "favourite_team",
    "underdog_team",
    "auto_upset_score",
    "auto_upset_reasons",
    "manual_upset_team",
    "manual_upset_score",
    "manual_upset_notes",
    "upset_team",
    "final_upset_score",
    "upset_flag",
    "fragile_favourite",
    "required_edge",
    "value_flag",
    "pick",
    "edge",
    "stake",
    "stake_units",
    "stake_dollars",
    "recommended_bet",
    "generated_at",
]

LEGACY_ALIASES = {
    "home_win_prob": "model_home_win_prob",
}

NUMERIC_COLS = [
    "model_home_win_prob",
    "market_home_win_prob",
    "final_home_win_prob",
    "exp_margin_home",
    "exp_total",
    "confidence",
    "home_odds",
    "away_odds",
    "auto_upset_score",
    "manual_upset_score",
    "final_upset_score",
    "required_edge",
    "edge",
    "stake",
    "stake_units",
    "stake_dollars",
    "upset_flag",
    "fragile_favourite",
]

TEXT_COLS = [
    "run_id",
    "run_utc",
    "mode",
    "rating_mode",
    "date",
    "kickoff_local",
    "venue",
    "home",
    "away",
    "favourite_team",
    "underdog_team",
    "auto_upset_reasons",
    "manual_upset_team",
    "manual_upset_notes",
    "upset_team",
    "value_flag",
    "pick",
    "recommended_bet",
    "generated_at",
]


def _norm_text(s: str) -> str:
    return " ".join(str(s).strip().split())


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _coerce_text(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).map(_norm_text)
        df[col] = df[col].replace({"nan": "", "NaT": ""})
    return df


def _load_history_flex(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=EXPECTED_COLS)

    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"[warn] Could not read {path}: {e}")
        return pd.DataFrame(columns=EXPECTED_COLS)

    if df.empty:
        return pd.DataFrame(columns=EXPECTED_COLS)

    # rename old columns into new schema where possible
    for old_name, new_name in LEGACY_ALIASES.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    # ensure all expected columns exist
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = ""

    # discard extra columns not needed for current output shape
    df = df[EXPECTED_COLS].copy()

    # normalize types
    df = _coerce_text(df, TEXT_COLS)
    df = _coerce_numeric(df, NUMERIC_COLS)

    # regenerate stake_units if missing/zero but dollars exist
    if "stake_units" in df.columns and "stake_dollars" in df.columns:
        missing_units = df["stake_units"].isna() | (df["stake_units"] == 0)
        df.loc[missing_units, "stake_units"] = (df.loc[missing_units, "stake_dollars"] / 10.0).round(2)

    # normalize date-like fields
    if "generated_at" in df.columns:
        df["generated_at"] = pd.to_datetime(df["generated_at"], errors="coerce", utc=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if "kickoff_local" in df.columns:
        df["kickoff_local"] = df["kickoff_local"].astype(str).replace("nan", "").str.strip()

    if "upset_flag" in df.columns:
        df["upset_flag"] = pd.to_numeric(df["upset_flag"], errors="coerce").fillna(0).astype(int)

    if "fragile_favourite" in df.columns:
        df["fragile_favourite"] = pd.to_numeric(df["fragile_favourite"], errors="coerce").fillna(0).astype(int)

    return df


def _rewrite_clean_history(df: pd.DataFrame, path: str) -> None:
    out = df.copy()

    # write generated_at back as string
    if "generated_at" in out.columns:
        out["generated_at"] = out["generated_at"].dt.strftime("%Y-%m-%d %H:%M:%S%z").fillna("")

    out.to_csv(path, index=False, quoting=csv.QUOTE_ALL)


def main():
    if not os.path.exists(PRED_HISTORY_PATH):
        print(f"[warn] Missing {PRED_HISTORY_PATH}")
        pd.DataFrame(columns=EXPECTED_COLS).to_csv(CURRENT_OUT, index=False)
        return

    df = _load_history_flex(PRED_HISTORY_PATH)

    if df.empty:
        print("[warn] No valid history rows found")
        pd.DataFrame(columns=EXPECTED_COLS).to_csv(CURRENT_OUT, index=False)
        return

    required = {"run_id", "date", "home", "away", "generated_at"}
    missing = required - set(df.columns)
    if missing:
        print(f"[warn] Missing required columns: {sorted(missing)}")
        pd.DataFrame(columns=EXPECTED_COLS).to_csv(CURRENT_OUT, index=False)
        return

    # remove rows that have no valid generated_at
    df = df[~df["generated_at"].isna()].copy()
    if df.empty:
        print("[warn] No rows with valid generated_at")
        pd.DataFrame(columns=EXPECTED_COLS).to_csv(CURRENT_OUT, index=False)
        return

    # identify latest run by generated_at max inside each run_id
    run_meta = (
        df.groupby("run_id", dropna=False)["generated_at"]
        .max()
        .reset_index()
        .sort_values("generated_at")
    )

    if run_meta.empty:
        print("[warn] No run metadata found")
        pd.DataFrame(columns=EXPECTED_COLS).to_csv(CURRENT_OUT, index=False)
        return

    latest_run_id = run_meta.iloc[-1]["run_id"]
    out = df[df["run_id"] == latest_run_id].copy()

    # sort current round rows
    sort_cols = [c for c in ["date", "kickoff_local", "home"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)

    out = out[EXPECTED_COLS].copy()

    # rewrite repaired full history so future runs stay aligned
    _rewrite_clean_history(df, PRED_HISTORY_PATH)

    # write latest run snapshot for website / downstream files
    write_out = out.copy()
    if "generated_at" in write_out.columns:
        write_out["generated_at"] = write_out["generated_at"].dt.strftime("%Y-%m-%d %H:%M:%S%z").fillna("")

    write_out.to_csv(CURRENT_OUT, index=False)

    print(f"[info] Repaired history and wrote {len(out)} rows for latest run_id={latest_run_id}")


if __name__ == "__main__":
    main()
