import os
import pandas as pd

def main():
    src = "results_2025.csv"  # <-- ROOT file (matches your repo)
    if not os.path.exists(src) or os.path.getsize(src) == 0:
        print(f"[error] {src} not found or empty in repo root.")
        return

    df = pd.read_csv(src)
    print(f"[info] Read {len(df)} rows from {src}")
    print(f"[info] Columns: {list(df.columns)}")

    # Your file ALREADY has these columns:
    # date, home, away, home_pts, away_pts
    needed = ["date", "home", "away", "home_pts", "away_pts"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print("[error] Missing required columns:", missing)
        return

    df = df[needed].copy()

    # normalise date
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date","home","away","home_pts","away_pts"])

    # append to results_cache.csv (create if not exists)
    if os.path.exists("results_cache.csv") and os.path.getsize("results_cache.csv") > 0:
        existing = pd.read_csv("results_cache.csv")
        df = pd.concat([existing, df], ignore_index=True)

    df = df.drop_duplicates(subset=["date","home","away"])
    df.to_csv("results_cache.csv", index=False)

    print(f"[ok] Loaded {len(df)} total results into results_cache.csv")

if __name__ == "__main__":
    main()
