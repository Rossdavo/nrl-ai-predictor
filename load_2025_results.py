import os
import pandas as pd

def main():
    # your repo has results_2025.csv at ROOT (not /data), so try both safely
    candidates = ["results_2025.csv", "data/results_2025.csv"]
    src = None
    for c in candidates:
        if os.path.exists(c):
            src = c
            break

    if src is None:
        print("Could not find results_2025.csv (tried root + data/)")
        # write an empty cache so downstream never crashes
        pd.DataFrame(columns=["date","home","away","home_pts","away_pts"]).to_csv("results_cache.csv", index=False)
        return

    print(f"Reading: {src}")

    # IMPORTANT: your pasted data is TAB separated
    # and has a header line like: date,home,away,home_pts,away_pts
    # followed by tabbed rows.
    df = pd.read_csv(src, sep=None, engine="python")  # auto-detect delimiter

    # Normalize column names if needed
    df.columns = [c.strip().lower() for c in df.columns]

    # Expected columns
    needed = ["date", "home", "away", "home_pts", "away_pts"]
    if not set(needed).issubset(df.columns):
        print("File columns found:", df.columns.tolist())
        print("Expected:", needed)
        pd.DataFrame(columns=needed).to_csv("results_cache.csv", index=False)
        return

    df = df[needed].copy()

    # Convert date to ISO YYYY-MM-DD (Australian format in your file is D/M/YYYY)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")

    # Ensure scores are numeric
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")

    df = df.dropna(subset=["date", "home", "away", "home_pts", "away_pts"])

    # Append to results_cache.csv if exists
    if os.path.exists("results_cache.csv"):
        existing = pd.read_csv("results_cache.csv")
        df = pd.concat([existing, df], ignore_index=True)

    df = df.drop_duplicates(subset=["date", "home", "away"]).sort_values(["date","home"])
    df.to_csv("results_cache.csv", index=False)

    print(f"Loaded {len(df)} total results into results_cache.csv")

if __name__ == "__main__":
    main()
