import pandas as pd
import os

def main():
    src = "data/results_2025.csv"   # file in data folder

    if not os.path.exists(src):
        print("results_2025.csv not found in data/")
        return

    df = pd.read_csv(src)

    # If file is already in correct format do nothing
    expected_cols = {"date", "home", "away", "home_pts", "away_pts"}

    if not expected_cols.issubset(df.columns):
        # Attempt rename from alternate column naming
        df = df.rename(columns={
            "Date": "date",
            "Home": "home",
            "Away": "away",
            "Homescore": "home_pts",
            "Awayscore": "away_pts"
        })

    df = df[["date", "home", "away", "home_pts", "away_pts"]]

    # append to results_cache.csv (create if not exists)
    if os.path.exists("results_cache.csv"):
        existing = pd.read_csv("results_cache.csv")
        df = pd.concat([existing, df], ignore_index=True)

    df = df.drop_duplicates(subset=["date", "home", "away"])
    df.to_csv("results_cache.csv", index=False)

    print(f"Loaded {len(df)} total results into results_cache.csv")


if __name__ == "__main__":
    main()
