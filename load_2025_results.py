import pandas as pd

def main():
    src = "results_2025.csv"   # your uploaded file
    df = pd.read_csv(src)

    df = df.rename(columns={
        "Date": "date",
        "Home": "home",
        "Away": "away",
        "Homescore": "home_pts",
        "Awayscore": "away_pts"
    })

    df = df[["date","home","away","home_pts","away_pts"]]

    # append to results_cache.csv (create if not exists)
    try:
        existing = pd.read_csv("results_cache.csv")
        df = pd.concat([existing, df], ignore_index=True)
    except:
        pass

    df = df.drop_duplicates(subset=["date","home","away"])
    df.to_csv("results_cache.csv", index=False)

    print(f"Loaded {len(df)} total results into results_cache.csv")

if __name__ == "__main__":
    main()
