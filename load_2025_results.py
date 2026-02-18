import os
import pandas as pd
from pandas.errors import EmptyDataError

def main():
    src = "data/results_2025.csv"  # FORCE data folder

    print("PWD:", os.getcwd())
    print("Reading:", src)

    if not os.path.exists(src):
        print(f"ERROR: {src} not found")
        pd.DataFrame(columns=["date","home","away","home_pts","away_pts"]).to_csv("results_cache.csv", index=False)
        return

    size = os.path.getsize(src)
    print("File size:", size, "bytes")
    if size == 0:
        print("ERROR: results_2025.csv is EMPTY (0 bytes)")
        pd.DataFrame(columns=["date","home","away","home_pts","away_pts"]).to_csv("results_cache.csv", index=False)
        return

    try:
        df = pd.read_csv(src)
    except EmptyDataError:
        print("ERROR: CSV has no columns (empty or broken file)")
        pd.DataFrame(columns=["date","home","away","home_pts","away_pts"]).to_csv("results_cache.csv", index=False)
        return

    # If your file already has correct headers, keep it simple:
    expected = {"date","home","away","home_pts","away_pts"}
    if expected.issubset(set(df.columns)):
        df = df[["date","home","away","home_pts","away_pts"]]
    else:
        # fallback rename mapping (only used if your headers differ)
        df = df.rename(columns={
            "Date": "date",
            "Home": "home",
            "Away": "away",
            "Homescore": "home_pts",
            "Awayscore": "away_pts",
        })
        df = df[["date","home","away","home_pts","away_pts"]]

    # Append to results_cache.csv
    if os.path.exists("results_cache.csv") and os.path.getsize("results_cache.csv") > 0:
        try:
            existing = pd.read_csv("results_cache.csv")
            df = pd.concat([existing, df], ignore_index=True)
        except Exception:
            pass

    df = df.drop_duplicates(subset=["date","home","away"])
    df.to_csv("results_cache.csv", index=False)

    print(f"Loaded {len(df)} total results into results_cache.csv")

if __name__ == "__main__":
    main()
