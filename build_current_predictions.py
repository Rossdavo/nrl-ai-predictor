import os
import csv
import pandas as pd

PRED_HISTORY_PATH = "predictions_history.csv"
CURRENT_OUT = "predictions.csv"

EXPECTED_COLS = [
    "run_id",
    "run_utc",
    "date",
    "kickoff_local",
    "home",
    "away",
    "home_win_prob",
    "exp_margin_home",
    "home_odds",
    "away_odds",
    "pick",
    "edge",
    "stake",
    "stake_dollars",
    "recommended_bet",
    "home_top_try",
    "away_top_try",
    "generated_at",
]


def _norm(s: str) -> str:
    return " ".join(str(s).strip().split())


def _load_and_repair_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=EXPECTED_COLS)

    repaired_rows = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return pd.DataFrame(columns=EXPECTED_COLS)

    # skip existing header, whatever shape it is
    data_rows = rows[1:]

    for i, row in enumerate(data_rows, start=2):
        if not row:
            continue

        # old format: 16 fields
        if len(row) == 16:
            row = row[:15] + ["", ""] + [row[15]]

        # new format: 18 fields
        elif len(row) == 18:
            pass

        # too short: pad
        elif len(row) < 18:
            row = row + [""] * (18 - len(row))

        # too long: trim
        elif len(row) > 18:
            row = row[:18]

        repaired_rows.append(row)

    df = pd.DataFrame(repaired_rows, columns=EXPECTED_COLS)
    return df


def _rewrite_clean_history(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)


def main():
    if not os.path.exists(PRED_HISTORY_PATH):
        print(f"[warn] Missing {PRED_HISTORY_PATH}")
        pd.DataFrame(columns=EXPECTED_COLS).to_csv(CURRENT_OUT, index=False)
        return

    df = _load_and_repair_history(PRED_HISTORY_PATH)

    if df.empty:
        print("[warn] No valid history rows found")
        pd.DataFrame(columns=EXPECTED_COLS).to_csv(CURRENT_OUT, index=False)
        return

    # rewrite repaired history so future runs are clean
    _rewrite_clean_history(df, PRED_HISTORY_PATH)

    required = {"run_id", "date", "home", "away", "generated_at"}
    missing = required - set(df.columns)

    if missing:
        print(f"[warn] Missing required columns: {sorted(missing)}")
        pd.DataFrame(columns=EXPECTED_COLS).to_csv(CURRENT_OUT, index=False)
        return

    df["generated_at"] = pd.to_datetime(df["generated_at"], errors="coerce", utc=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["home"] = df["home"].map(_norm)
    df["away"] = df["away"].map(_norm)
    df["kickoff_local"] = df["kickoff_local"].astype(str).replace("nan", "").str.strip()
    df["home_top_try"] = df["home_top_try"].astype(str).replace("nan", "").str.strip()
    df["away_top_try"] = df["away_top_try"].astype(str).replace("nan", "").str.strip()

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

    sort_cols = [c for c in ["date", "kickoff_local", "home"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)

    out = out[EXPECTED_COLS].copy()
    out.to_csv(CURRENT_OUT, index=False)

    print(f"[info] Repaired history and wrote {len(out)} rows for latest run_id={latest_run_id}")


if __name__ == "__main__":
    main()
