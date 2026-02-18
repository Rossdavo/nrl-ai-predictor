import os
import pandas as pd
from pandas.errors import EmptyDataError

CANDIDATE_PATHS = [
    "data/results_2025.csv",   # <-- most likely (you said it's in /data)
    "results_2025.csv",        # fallback if you move it to repo root
]

def read_first_nonempty_csv(paths):
    last_err = None
    for p in paths:
        if not os.path.exists(p):
            continue
        # skip truly empty files
        if os.path.getsize(p) == 0:
            last_err = f"{p} exists but is 0 bytes"
            continue
        try:
            # sep=None lets pandas auto-detect commas/tabs; utf-8-sig handles BOM
            return p, pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")
        except EmptyDataError:
            last_err = f"{p} parsed as empty (no columns)"
            continue
        except Exception as e:
            last_err = f"{p} read failed: {e}"
            continue
    raise RuntimeError(f"Could not load results_2025 CSV. Last error: {last_err}")

def main():
    src_path, df = read_first_nonempty_csv(CANDIDATE_PATHS)

    # If your file already has correct columns, keep them
    expected = {"date", "home", "away", "home_pts", "away_pts"}
    if not expected.issubset(set(df.columns)):
        # Try common alternate headings
        df = df.rename(columns={
            "Date": "date",
            "Home": "home",
            "Away": "away",
            "Homescore": "home_pts",
            "Awayscore": "away_pts",
            "HomeScore": "home_pts",
            "AwayScore": "away_pts",
        })

    # Keep only the required columns if they exist now
    missing = [c for c in ["date", "home", "away", "home_pts", "away_pts"] if c not in df.columns]
    if missing:
        raise RuntimeError(f"{src_path} is missing columns: {missing}. Found: {list(df.columns)}")

    df = df[["date", "home", "away", "home_pts", "away_pts"]].copy()

    # Clean types
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["home"] = df["home"].astype(str).str.strip()
    df["away"] = df["away"].astype(str).str.strip()
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")

    df = df.dropna(subset=["date", "home", "away", "home_pts", "away_pts"])

    # Append into results_cache.csv (create if missing)
    if os.path.exists("results_cache.csv") and os.path.getsize("results_cache.csv") > 0:
        try:
            existing = pd.read_csv("results_cache.csv")
            df = pd.concat([existing, df], ignore_index=True)
        except Exception:
            pass

    df = df.drop_duplicates(subset=["date", "home", "away"]).sort_values(["date", "home"])
    df.to_csv("results_cache.csv", index=False)

    print(f"Loaded {len(df)} total results into results_cache.csv (source: {src_path})")

if __name__ == "__main__":
    main()
