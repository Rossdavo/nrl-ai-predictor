#!/usr/bin/env python3
"""
build_selected17_strength.py

Experimental 2027 feature builder.

Purpose
-------
Build selected-17 squad-strength features from:
  - team_lists_history.csv
  - player_ratings.csv (preferred) or player_ratings_template.csv (fallback)

This script DOES NOT alter predictions. It writes auditable feature files only.

Outputs
-------
  selected17_strength.csv
  selected17_strength_latest.csv

Design rules
------------
- Exact 17-player team lists only.
- No invented rating for an unrated player.
- Coverage is reported explicitly.
- Team strength is only considered usable when rating coverage reaches
  MIN_RATING_COVERAGE.
- Key-position strength is reported separately for jerseys 1, 6, 7 and 9.
- No arbitrary injury penalty is applied here. The purpose is to create
  clean inputs for later backtesting.
"""

from __future__ import annotations

from pathlib import Path
import math
import re
import sys
import pandas as pd

TEAM_LISTS_FILE = Path("team_lists_history.csv")
RATINGS_FILES = [Path("player_ratings.csv"), Path("player_ratings_template.csv")]

OUTPUT_HISTORY = Path("selected17_strength.csv")
OUTPUT_LATEST = Path("selected17_strength_latest.csv")

MIN_RATING_COVERAGE = 0.80
KEY_JERSEYS = {1, 6, 7, 9}

TEAM_ALIASES = {
    "brisbane broncos": "Broncos",
    "broncos": "Broncos",
    "canberra raiders": "Raiders",
    "raiders": "Raiders",
    "canterbury bankstown bulldogs": "Bulldogs",
    "canterbury-bankstown bulldogs": "Bulldogs",
    "bulldogs": "Bulldogs",
    "cronulla sharks": "Sharks",
    "cronulla-sutherland sharks": "Sharks",
    "sharks": "Sharks",
    "dolphins": "Dolphins",
    "the dolphins": "Dolphins",
    "gold coast titans": "Titans",
    "titans": "Titans",
    "manly sea eagles": "Sea Eagles",
    "manly warringah sea eagles": "Sea Eagles",
    "sea eagles": "Sea Eagles",
    "melbourne storm": "Storm",
    "storm": "Storm",
    "newcastle knights": "Knights",
    "knights": "Knights",
    "new zealand warriors": "Warriors",
    "nz warriors": "Warriors",
    "warriors": "Warriors",
    "north queensland cowboys": "Cowboys",
    "cowboys": "Cowboys",
    "parramatta eels": "Eels",
    "eels": "Eels",
    "penrith panthers": "Panthers",
    "panthers": "Panthers",
    "south sydney rabbitohs": "Rabbitohs",
    "rabbitohs": "Rabbitohs",
    "st george illawarra dragons": "Dragons",
    "dragons": "Dragons",
    "sydney roosters": "Roosters",
    "roosters": "Roosters",
    "wests tigers": "Wests Tigers",
    "tigers": "Wests Tigers",
    "wests tigers": "Wests Tigers",
    "perth bears": "Perth Bears",
    "bears": "Perth Bears",
}


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalise_name(value) -> str:
    text = clean_text(value).lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9' -]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_team(value) -> str:
    raw = clean_text(value)
    key = normalise_name(raw)
    return TEAM_ALIASES.get(key, raw)


def choose_ratings_file() -> Path:
    for path in RATINGS_FILES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No player ratings file found. Expected player_ratings.csv "
        "or player_ratings_template.csv."
    )


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def load_team_lists() -> pd.DataFrame:
    if not TEAM_LISTS_FILE.exists():
        raise FileNotFoundError(f"Missing {TEAM_LISTS_FILE}")

    df = pd.read_csv(TEAM_LISTS_FILE)
    require_columns(
        df,
        {"round_id", "round_start", "round_end", "captured_at",
         "team", "jersey", "player", "position"},
        str(TEAM_LISTS_FILE),
    )

    df["team"] = df["team"].map(normalise_team)
    df["player"] = df["player"].map(clean_text)
    df["player_key"] = df["player"].map(normalise_name)
    df["position"] = df["position"].map(clean_text)
    df["jersey"] = pd.to_numeric(df["jersey"], errors="coerce")
    df["round_start"] = pd.to_datetime(df["round_start"], errors="coerce")
    df["round_end"] = pd.to_datetime(df["round_end"], errors="coerce")
    df["captured_at"] = pd.to_datetime(df["captured_at"], errors="coerce", utc=True)

    df = df[
        df["jersey"].between(1, 17, inclusive="both")
        & df["player_key"].ne("")
        & df["team"].ne("")
    ].copy()

    df["jersey"] = df["jersey"].astype(int)
    return df


def load_ratings() -> tuple[pd.DataFrame, Path]:
    path = choose_ratings_file()
    df = pd.read_csv(path)

    require_columns(df, {"team", "player", "impact_points"}, str(path))

    if "position" not in df.columns:
        df["position"] = ""

    df["team"] = df["team"].map(normalise_team)
    df["player"] = df["player"].map(clean_text)
    df["player_key"] = df["player"].map(normalise_name)
    df["impact_points"] = pd.to_numeric(df["impact_points"], errors="coerce")

    df = df[
        df["team"].ne("")
        & df["player_key"].ne("")
        & df["impact_points"].notna()
    ].copy()

    # Keep one rating per team/player. If duplicates exist, latest row in file wins.
    df = df.drop_duplicates(["team", "player_key"], keep="last")
    return df, path


def complete_snapshots(team_lists: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only snapshots containing exactly one player in every jersey 1-17.
    If the same round/team was captured more than once, use the latest capture.
    """
    valid_groups = []

    keys = ["round_id", "team"]
    for (round_id, team), grp in team_lists.groupby(keys, sort=False):
        # If multiple captures exist inside the same round/team, select latest
        # capture first, then validate the resulting 17.
        captures = grp["captured_at"].dropna()
        if not captures.empty:
            latest_capture = captures.max()
            candidate = grp[grp["captured_at"] == latest_capture].copy()
        else:
            candidate = grp.copy()

        candidate = candidate.sort_values(["jersey", "player_key"])
        candidate = candidate.drop_duplicates("jersey", keep="last")

        jerseys = set(candidate["jersey"].tolist())
        if len(candidate) == 17 and jerseys == set(range(1, 18)):
            valid_groups.append(candidate)

    if not valid_groups:
        return team_lists.iloc[0:0].copy()

    return pd.concat(valid_groups, ignore_index=True)


def build_snapshot_features(
    snapshots: pd.DataFrame,
    ratings: pd.DataFrame,
    ratings_file: Path,
) -> pd.DataFrame:
    rating_lookup = ratings[
        ["team", "player_key", "impact_points"]
    ].rename(columns={"impact_points": "player_impact_points"})

    merged = snapshots.merge(
        rating_lookup,
        on=["team", "player_key"],
        how="left",
        validate="many_to_one",
    )

    rows = []

    for (round_id, team), grp in merged.groupby(["round_id", "team"], sort=False):
        grp = grp.sort_values("jersey").copy()

        rated = grp["player_impact_points"].notna()
        key_mask = grp["jersey"].isin(KEY_JERSEYS)
        key_rated = key_mask & rated

        rated_count = int(rated.sum())
        coverage = rated_count / 17.0

        key_count = int(key_mask.sum())       # normally 4
        key_rated_count = int(key_rated.sum())
        key_coverage = key_rated_count / key_count if key_count else math.nan

        total_strength = (
            float(grp.loc[rated, "player_impact_points"].sum())
            if rated_count else math.nan
        )
        key_strength = (
            float(grp.loc[key_rated, "player_impact_points"].sum())
            if key_rated_count else math.nan
        )

        starters = grp[grp["jersey"].between(1, 13)]
        starter_rated = starters["player_impact_points"].notna()
        starter_count = int(starter_rated.sum())
        starter_coverage = starter_count / 13.0
        starter_strength = (
            float(starters.loc[starter_rated, "player_impact_points"].sum())
            if starter_count else math.nan
        )

        bench = grp[grp["jersey"].between(14, 17)]
        bench_rated = bench["player_impact_points"].notna()
        bench_count = int(bench_rated.sum())
        bench_coverage = bench_count / 4.0
        bench_strength = (
            float(bench.loc[bench_rated, "player_impact_points"].sum())
            if bench_count else math.nan
        )

        unrated_players = "; ".join(
            f"{int(r.jersey)} {r.player}"
            for r in grp.loc[~rated, ["jersey", "player"]].itertuples(index=False)
        )

        selected_players = "; ".join(
            f"{int(r.jersey)} {r.player}"
            for r in grp[["jersey", "player"]].itertuples(index=False)
        )

        usable = coverage >= MIN_RATING_COVERAGE

        rows.append({
            "round_id": round_id,
            "round_start": grp["round_start"].iloc[0].date().isoformat()
                if pd.notna(grp["round_start"].iloc[0]) else "",
            "round_end": grp["round_end"].iloc[0].date().isoformat()
                if pd.notna(grp["round_end"].iloc[0]) else "",
            "captured_at": grp["captured_at"].max().isoformat()
                if grp["captured_at"].notna().any() else "",
            "team": team,
            "selected17_players": selected_players,
            "rated_players": rated_count,
            "unrated_players_count": 17 - rated_count,
            "rating_coverage": round(coverage, 4),
            "strength_usable": int(usable),
            "selected17_strength": round(total_strength, 4)
                if usable and pd.notna(total_strength) else math.nan,
            "starter_rated_players": starter_count,
            "starter_rating_coverage": round(starter_coverage, 4),
            "starter_strength": round(starter_strength, 4)
                if starter_coverage >= MIN_RATING_COVERAGE and pd.notna(starter_strength)
                else math.nan,
            "bench_rated_players": bench_count,
            "bench_rating_coverage": round(bench_coverage, 4),
            "bench_strength": round(bench_strength, 4)
                if bench_coverage >= MIN_RATING_COVERAGE and pd.notna(bench_strength)
                else math.nan,
            "key_players_rated": key_rated_count,
            "key_rating_coverage": round(key_coverage, 4),
            "key_position_strength": round(key_strength, 4)
                if key_coverage == 1.0 and pd.notna(key_strength) else math.nan,
            "unrated_players": unrated_players,
            "ratings_source": ratings_file.name,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["round_start_sort"] = pd.to_datetime(out["round_start"], errors="coerce")
    out = out.sort_values(
        ["round_start_sort", "round_id", "team"]
    ).drop(columns=["round_start_sort"]).reset_index(drop=True)

    return out


def build_latest(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history.copy()

    temp = history.copy()
    temp["_round_start"] = pd.to_datetime(temp["round_start"], errors="coerce")
    temp["_captured_at"] = pd.to_datetime(
        temp["captured_at"], errors="coerce", utc=True
    )

    temp = temp.sort_values(
        ["team", "_round_start", "_captured_at"],
        na_position="first",
    )

    latest = temp.groupby("team", as_index=False).tail(1)
    latest = latest.drop(columns=["_round_start", "_captured_at"])
    return latest.sort_values("team").reset_index(drop=True)


def print_summary(history: pd.DataFrame, ratings_file: Path) -> None:
    print("\nSELECTED-17 STRENGTH BUILD")
    print("=" * 60)
    print(f"Ratings source: {ratings_file}")
    print(f"Minimum usable rating coverage: {MIN_RATING_COVERAGE:.0%}")

    if history.empty:
        print("No complete 17-player snapshots found.")
        return

    print(f"Complete team snapshots: {len(history)}")
    print(f"Teams represented: {history['team'].nunique()}")
    print(
        "Usable strength snapshots: "
        f"{int(history['strength_usable'].sum())}/{len(history)}"
    )
    print(
        "Average player-rating coverage: "
        f"{history['rating_coverage'].mean():.1%}"
    )

    incomplete = history[history["strength_usable"] == 0]
    if not incomplete.empty:
        print("\nSnapshots withheld because rating coverage is too low:")
        for row in incomplete.itertuples(index=False):
            print(
                f"  {row.round_id} | {row.team}: "
                f"{row.rated_players}/17 rated"
            )

    print(f"\nSaved: {OUTPUT_HISTORY}")
    print(f"Saved: {OUTPUT_LATEST}")
    print("\nNOTE: No prediction adjustment has been applied.")


def main() -> int:
    try:
        team_lists = load_team_lists()
        ratings, ratings_file = load_ratings()
        snapshots = complete_snapshots(team_lists)

        history = build_snapshot_features(snapshots, ratings, ratings_file)
        latest = build_latest(history)

        history.to_csv(OUTPUT_HISTORY, index=False)
        latest.to_csv(OUTPUT_LATEST, index=False)

        print_summary(history, ratings_file)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
