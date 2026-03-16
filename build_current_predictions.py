import os
import pandas as pd

PRED_HISTORY_PATH = "predictions_history.csv"
CURRENT_OUT = "predictions.csv"


def _norm(s: str) -> str:
    return " ".join(str(s).strip().split())


def main():
    if not os.path.exists(PRED_HISTORY_PATH):
        print(f"[warn] Missing {PRED_HISTORY_PATH}")
        pd.DataFrame().to_csv(CURRENT_OUT, index=False)
        return

    df = pd.read_csv(PRED_HISTORY_PATH)

    required = {"run_id", "date", "home", "away", "generated_at"}
    missing = required - set(df.columns)

    if missing:
        print(f"[warn] Missing required columns: {sorted(missing)}")
        pd.DataFrame().to_csv(CURRENT_OUT, index=False)
        return

    df["generated_at"] = pd.to_datetime(df["generated_at"], errors="coerce", utc=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["home"] = df["home"].map(_norm)
    df["away"] = df["away"].map(_norm)

    run_meta = (
        df.groupby("run_id", dropna=False)["generated_at"]
        .max()
        .reset_index()
        .sort_values("generated_at")
    )

    if run_meta.empty:
        print("[warn] No run metadata found")
        pd.DataFrame().to_csv(CURRENT_OUT, index=False)
        return

    latest_run_id = run_meta.iloc[-1]["run_id"]

    out = df[df["run_id"] == latest_run_id].copy()

    sort_cols = [c for c in ["date", "kickoff_local", "home"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)

    out.to_csv(CURRENT_OUT, index=False)

    print(f"[info] Wrote {len(out)} rows for latest run_id={latest_run_id}")


if __name__ == "__main__":
    main()
