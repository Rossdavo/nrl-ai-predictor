import os
import pandas as pd

EXPECTED = ["date", "home", "away", "home_pts", "away_pts"]

SRC = "data/results_2025.csv"   # <-- FORCE data folder source


def read_results_file(path: str) -> pd.DataFrame:
    """
    Robust reader for mixed-delimiter files.
    Tries:
      1) comma CSV
      2) tab-separated
      3) manual parse into 5 columns
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
        for line in f:
            line = line.strip()
            if not line:
                continue

            lower = line.lower()
            # skip header rows
            if "date" in lower and "home" in lower and "away" in lower and "home_pts" in lower:
                continue

            parts = line.split("\t")
            if len(parts) == 1:
                parts = line.split(",")
            if len(parts) == 1:
                parts = [p for p in line.split(" ") if p]

            if len(parts) < 5:
                continue

            d, home, away, hp, ap = parts[0], parts[1], parts[2], parts[3], parts[4]
            rows.append([d.strip(), home.strip(), away.strip(), hp.strip(), ap.strip()])

    return pd.DataFrame(rows, columns=EXPECTED)


def main():
    if not os.path.exists(SRC):
        print(f"Missing {SRC}")
        # keep downstream safe
        pd.DataFrame(columns=EXPECTED).to_csv("results_cache.csv", index=False)
        return

    size = os.path.getsize(SRC)
    print(f"Reading: {SRC}")
    print(f"File size: {size} bytes")

    df = read_results_file(SRC)

    if df.empty:
        print("Parsed 0 rows from data/results_2025.csv (format not readable).")
        pd.DataFrame(columns=EXPECTED).to_csv("results_cache.csv", index=False)
        return

    # Normalize types
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")
    df = df.dropna(subset=["date", "home", "away", "home_pts", "away_pts"])

    # Append to existing cache (if present)
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
