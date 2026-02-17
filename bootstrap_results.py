import os
import pandas as pd

HIST_PATH = "data/results_2025.csv"
CACHE_PATH = "results_cache.csv"

COLS = ["date", "home", "away", "home_pts", "away_pts"]

def main():
    if not os.path.exists(HIST_PATH):
        print(f"No {HIST_PATH} found — skipping bootstrap.")
        return

    # header=None means “no header row in file”
    hist = pd.read_csv(HIST_PATH, header=None, names=COLS)

    # Basic cleanup
    hist["date"] = hist["date"].astype(str).str.strip()
    hist["home"] = hist["home"].astype(str).str.strip()
    hist["away"] = hist["away"].astype(str).str.strip()
    hist["home_pts"] = pd.to_numeric(hist["home_pts"], errors="coerce")
    hist["away_pts"] = pd.to_numeric(hist["away_pts"], errors="coerce")

    hist = hist.dropna(subset=["date", "home", "away", "home_pts", "away_pts"]).copy()
    hist["home_pts"] = hist["home_pts"].astype(int)
    hist["away_pts"] = hist["away_pts"].astype(int)

    if os.path.exists(CACHE_PATH):
        cache = pd.read_csv(CACHE_PATH)
        for c in COLS:
            if c not in cache.columns:
                cache[c] = ""
        cache = cache[COLS].copy()
        cache["date"] = cache["date"].astype(str)

        merged = pd.concat([cache, hist], ignore_index=True)
    else:
        merged = hist

    merged = merged.drop_duplicates(subset=["date", "home", "away"], keep="last")
    merged = merged.sort_values(["date", "home", "away"])

    merged.to_csv(CACHE_PATH, index=False)
    print(f"Bootstrapped results_cache.csv ({len(merged)} rows)")

if __name__ == "__main__":
    main()
