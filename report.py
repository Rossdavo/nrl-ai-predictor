import os
import pandas as pd

EXPECTED = ["date", "home", "away", "home_pts", "away_pts"]
CACHE_PATH = "results_cache.csv"
BOOTSTRAP_CANDIDATES = ["results_2025.csv", "data/results_2025.csv"]


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPECTED)


def _normalise_results_df(df: pd.DataFrame, dayfirst: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_results()

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if not set(EXPECTED).issubset(df.columns):
        return _empty_results()

    df = df[EXPECTED].copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=dayfirst, errors="coerce").dt.strftime("%Y-%m-%d")
    df["home"] = df["home"].astype(str).str.strip()
    df["away"] = df["away"].astype(str).str.strip()
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")

    df = df.dropna(subset=["date", "home", "away", "home_pts", "away_pts"]).copy()
    return df


def read_results_file(path: str, dayfirst: bool = False) -> pd.DataFrame:
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
        df = _normalise_results_df(df, dayfirst=dayfirst)
        if not df.empty:
            return df
    except Exception:
        pass

    # 2) tab-separated
    try:
        df = pd.read_csv(path, sep="\t")
        df = _normalise_results_df(df, dayfirst=dayfirst)
        if not df.empty:
            return df
    except Exception:
        pass

    # 3) manual parse fallback
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                lower = line.lower()
                if "date" in lower and "home" in lower and "away" in lower and "home_pts" in lower:
                    continue

                parts = line.split("\t")
                if len(parts) == 1:
                    parts = line.split(",")
                if len(parts) == 1:
                    parts = [p for p in line.split(" ") if p]

                if len(parts) < 5:
                    continue

                d, home, away, hp, ap = parts[:5]
                rows.append([d.strip(), home.strip(), away.strip(), hp.strip(), ap.strip()])
    except Exception:
        return _empty_results()

    df = pd.DataFrame(rows, columns=EXPECTED)
    return _normalise_results_df(df, dayfirst=dayfirst)


def main():
    # 1) Load existing cache first — this is the main source of truth
    if os.path.exists(CACHE_PATH):
        existing = read_results_file(CACHE_PATH, dayfirst=False)
        print(f"Reading cache: {CACHE_PATH}")
    else:
        existing = _empty_results()
        print("No existing results_cache.csv found. Starting fresh.")

    # 2) Optionally backfill from 2025 file if present
    bootstrap_src = next((c for c in BOOTSTRAP_CANDIDATES if os.path.exists(c)), None)

    if bootstrap_src:
        print(f"Backfilling from: {bootstrap_src}")
        df_2025 = read_results_file(bootstrap_src, dayfirst=True)
    else:
        df_2025 = _empty_results()
        print("No results_2025.csv found for backfill.")

    # 3) Merge safely and never wipe current cache
    combined = pd.concat([existing, df_2025], ignore_index=True)

    if combined.empty:
        _empty_results().to_csv(CACHE_PATH, index=False)
        print("No results available. Wrote empty results_cache.csv")
        return

    combined = (
        combined.drop_duplicates(subset=["date", "home", "away"], keep="last")
        .sort_values(["date", "home", "away"])
        .reset_index(drop=True)
    )

    combined.to_csv(CACHE_PATH, index=False)
    print(f"Loaded {len(combined)} total results into {CACHE_PATH}")


if __name__ == "__main__":
    main()
