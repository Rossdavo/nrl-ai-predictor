import pandas as pd
import os

def main():
    src = "data/results_2025.csv"   # FIXED PATH

    if not os.path.exists(src):
        print(f"File not found: {src}")
        return

    df = pd.read_csv(src)

    # Standardise column names if needed
    df = df.rename(columns={
        "Date": "date",
        "Home": "home",
        "Away": "away",
        "Homescore": "home_pts",
        "Awayscore": "away_pts"
    })

    # Keep only required columns
    df = df[["date", "home", "away", "home_pts", "away_pts"]]

    # Append to results_cache.csv (create if not exists)
    if os.path.exists("results_cache.csv"):
        existing = pd.read_csv("results_cache.csv")
        df = pd.concat([existing, df], ignore_index=True)

    df = df.drop_duplicates(subset=["date", "home", "away"])
    df.to_csv("results_cache.csv", index=False)

    print(f"Loaded {len(df)} total results into results_cache.csv")

if __name__ == "__main__":
    main()
