import os
import pandas as pd

EXPECTED = ["date", "home", "away", "home_pts", "away_pts"]

def read_results_file(path: str) -> pd.DataFrame:
    """
    Robust reader for mixed-delimiter files.
    Tries:
      1) normal CSV (comma)
      2) tab-separated
      3) manual parse lines into 5 columns
    """
    # 1) comma CSV
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if set(EXPECTED).issubset(df.columns):
            return df[EXPECTED].copy()
    except Exception:
        pass

    # 2) tab-separated
    try:
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.strip().lower() for c in df.columns]
        if set(EXPECTED).issubset(df.columns):
            return df[EXPECTED].copy()
    except Exception:
        pass

    # 3) manual parse (handles "header comma, rows tab" or weird spacing)
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Skip header if it contains these column names
            lower = line.lower()
            if "date" in lower and "home" in lower and "away" in lower and "home_pts" in lower:
                continue

            # Try split by tab first
            parts = line.split("\t")
            if len(parts) == 1:
                # Try comma
                parts = line.split(",")
            if len(parts) == 1:
                # Try multiple spaces as last resort
                parts = [p for p in line.split(" ") if p]

            # We only accept lines that can form 5 fields
            if len(parts) < 5:
                continue

            # Take first 5 columns (some lines might have extra)
            d, home, away, hp, ap = parts[0], parts[1], parts[2], parts[3], parts[4]
            rows.append([d.strip(), home.strip(), away.strip(), hp.strip(), ap.strip()])

    df = pd.DataFrame(rows, columns=EXPECTED)
    return df


def main():
    # Try root first, then data folder
    candidates = ["results_2025.csv", "data/results_2025.csv"]
    src = next((c for c in candidates if os.path.exists(c)), None)

    if not src:
        print("Could not find results_2025.csv (tried root + data/)")
        pd.DataFrame(columns=EXPECTED).to_csv("results_cache.csv", index=False)
        return

    print(f"Reading: {src}")
    df = read_results_file(src)

    if df.empty:
        print("Parsed 0 rows from results_2025.csv (file format still not readable).")
        pd.DataFrame(columns=EXPECTED).to_csv("results_cache.csv", index=False)
        return

    # Date normalization (your file is day-first)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")

    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")

    df = df.dropna(subset=["date", "home", "away", "home_pts", "away_pts"])

    # Append to existing cache if it exists
    if os.path.exists("results_cache.csv"):
        try:
            existing = pd.read_csv("results_cache.csv")
            if set(EXPECTED).issubset(existing.columns):
                df = pd.concat([existing[EXPECTED], df], ignore_index=True)
        except Exception:
            pass

    df = df.drop_duplicates(subset=["date", "home", "away"]).sort_values(["date", "home"])
    df.to_csv("results_cache.csv", index=False)

    print(f"Loaded {len(df)} total results into results_cache.csv")


if __name__ == "__main__":
    main()
