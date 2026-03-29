import os
import re
import time
from io import StringIO
from datetime import datetime, timezone

import pandas as pd
import requests


RESULTS_CACHE_PATH = "results_cache.csv"
MANUAL_RESULTS_2026_PATH = "results_2026.csv"

RESULTS_SOURCES = {
    2025: "https://fixturedownload.com/results/nrl-2025",
    2026: "https://fixturedownload.com/results/nrl-2026",
}

EXPECTED_COLS = ["date", "home", "away", "home_pts", "away_pts", "season_year"]

TEAM_NAME_NORMALISE = {
    "Canterbury Bulldogs": "Bulldogs",
    "Canterbury-Bankstown Bulldogs": "Bulldogs",
    "St George Illawarra Dragons": "Dragons",
    "St Geo Illa": "Dragons",
    "Newcastle Knights": "Knights",
    "North Queensland Cowboys": "Cowboys",
    "North Qld": "Cowboys",
    "Melbourne Storm": "Storm",
    "Parramatta Eels": "Eels",
    "New Zealand Warriors": "Warriors",
    "Sydney Roosters": "Roosters",
    "Brisbane Broncos": "Broncos",
    "Penrith Panthers": "Panthers",
    "Cronulla Sutherland Sharks": "Sharks",
    "Cronulla-Sutherland Sharks": "Sharks",
    "Gold Coast Titans": "Titans",
    "Manly Warringah Sea Eagles": "Sea Eagles",
    "Manly-Warringah Sea Eagles": "Sea Eagles",
    "Canberra Raiders": "Raiders",
    "South Sydney Rabbitohs": "Rabbitohs",
    "Wests Tigers": "Wests Tigers",
    "The Dolphins": "Dolphins",
    "Dolphins": "Dolphins",
    "Bulldogs": "Bulldogs",
    "Dragons": "Dragons",
    "Knights": "Knights",
    "Cowboys": "Cowboys",
    "Storm": "Storm",
    "Eels": "Eels",
    "Warriors": "Warriors",
    "Roosters": "Roosters",
    "Broncos": "Broncos",
    "Panthers": "Panthers",
    "Sharks": "Sharks",
    "Titans": "Titans",
    "Sea Eagles": "Sea Eagles",
    "Raiders": "Raiders",
    "Rabbitohs": "Rabbitohs",
}


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def norm_team(name: str) -> str:
    name = str(name or "").strip()
    return TEAM_NAME_NORMALISE.get(name, name)


def empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPECTED_COLS)


def normalise_results_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_results()

    work = df.copy()
    for col in ["date", "home", "away"]:
        if col not in work.columns:
            work[col] = ""

    for col in ["home_pts", "away_pts"]:
        if col not in work.columns:
            work[col] = pd.NA

    work["date"] = pd.to_datetime(work["date"], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d")
    work["home"] = work["home"].astype(str).map(norm_team)
    work["away"] = work["away"].astype(str).map(norm_team)
    work["home_pts"] = pd.to_numeric(work["home_pts"], errors="coerce")
    work["away_pts"] = pd.to_numeric(work["away_pts"], errors="coerce")

    work = work.dropna(subset=["date", "home", "away", "home_pts", "away_pts"]).copy()
    work["season_year"] = pd.to_datetime(work["date"], errors="coerce").dt.year
    work = work.dropna(subset=["season_year"]).copy()
    work["season_year"] = work["season_year"].astype(int)

    work = work[EXPECTED_COLS].copy()
    work = work.drop_duplicates(subset=["date", "home", "away"], keep="last").reset_index(drop=True)
    work = work.sort_values(["date", "home", "away"]).reset_index(drop=True)
    return work


def load_existing_cache(path: str = RESULTS_CACHE_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return empty_results()

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Could not read {path}: {e}")
        return empty_results()

    return normalise_results_df(df)


def load_manual_results(path: str = MANUAL_RESULTS_2026_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return empty_results()

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Could not read {path}: {e}")
        return empty_results()

    return normalise_results_df(df)


def _extract_scores(result_text: str):
    s = str(result_text or "").strip()
    if not s or s == "-":
        return (pd.NA, pd.NA)

    m = re.search(r"(\d+)\s*[-–]\s*(\d+)", s)
    if not m:
        return (pd.NA, pd.NA)

    return (int(m.group(1)), int(m.group(2)))


def _extract_results_from_table(table: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in table.columns}

    required = {"date", "home team", "away team", "result"}
    if not required.issubset(cols.keys()):
        return empty_results()

    scores = table[cols["result"]].apply(_extract_scores)

    out = pd.DataFrame({
        "date": table[cols["date"]],
        "home": table[cols["home team"]],
        "away": table[cols["away team"]],
        "home_pts": scores.apply(lambda x: x[0]),
        "away_pts": scores.apply(lambda x: x[1]),
    })

    return normalise_results_df(out)


def fetch_results_page(year: int, url: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    html = None
    last_err = None

    for attempt in range(3):
        try:
            r = requests.get(url, timeout=45, headers=headers)
            r.raise_for_status()
            html = r.text
            break
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))

    if html is None:
        print(f"[warn] Failed to fetch {year} results page: {last_err}")
        return empty_results()

    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        print(f"[warn] pd.read_html failed for {year}: {e}")
        return empty_results()

    if not tables:
        print(f"[warn] No tables found for {year}")
        return empty_results()

    best = empty_results()
    for t in tables:
        extracted = _extract_results_from_table(t)
        if len(extracted) > len(best):
            best = extracted.copy()

    if best.empty:
        print(f"[warn] Could not extract results table for {year}")
        return empty_results()

    best["season_year"] = year
    best = best[EXPECTED_COLS].copy()
    best = best.drop_duplicates(subset=["date", "home", "away"], keep="last").reset_index(drop=True)
    print(f"[info] fetched {len(best)} completed results for {year}")
    return best


def fetch_all_results() -> pd.DataFrame:
    frames = []
    for year, url in RESULTS_SOURCES.items():
        df = fetch_results_page(year, url)
        if not df.empty:
            frames.append(df)

    if not frames:
        return empty_results()

    out = pd.concat(frames, ignore_index=True)
    out = normalise_results_df(out)
    return out


def build_merged_results() -> pd.DataFrame:
    cache_df = load_existing_cache()
    manual_df = load_manual_results()
    fetched_df = fetch_all_results()

    merged = pd.concat([cache_df, manual_df, fetched_df], ignore_index=True)
    merged = normalise_results_df(merged)
    merged = merged[merged["season_year"] >= 2025].copy()
    return merged


def write_manual_2026_from_merged(merged: pd.DataFrame, path: str = MANUAL_RESULTS_2026_PATH) -> None:
    out = merged[merged["season_year"] == 2026][["date", "home", "away", "home_pts", "away_pts"]].copy()
    out = out.sort_values(["date", "home", "away"]).reset_index(drop=True)
    out.to_csv(path, index=False)
    print(f"[info] wrote {len(out)} rows -> {path}")


def main():
    merged = build_merged_results()
    if merged.empty:
        print("[warn] No results available to write")
        empty_results().to_csv(RESULTS_CACHE_PATH, index=False)
        return

    merged.to_csv(RESULTS_CACHE_PATH, index=False)
    print(f"[info] wrote {len(merged)} rows -> {RESULTS_CACHE_PATH}")

    write_manual_2026_from_merged(merged)

    latest_date = merged["date"].max()
    print(f"[info] latest completed result date: {latest_date}")
    print(f"[info] sync completed at {utc_now_str()}")


if __name__ == "__main__":
    main()
