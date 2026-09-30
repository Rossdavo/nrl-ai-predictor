#!/usr/bin/env python3
"""
build_player_ratings_v2.py

Experimental all-club player rating framework for the NRL predictor.

This script deliberately does NOT use fox_player_stats.csv.

Purpose
-------
Create a transparent player-ratings file from the canonical player master and
any verified rating/experience information already available.

The important design decision is that UNKNOWN IS NOT AVERAGE.
A player without enough evidence remains unrated. This prevents incomplete
player data from silently changing match predictions.

Inputs
------
player_master.csv                         required
player_experience.csv                     optional
player_ratings.csv                        optional legacy/verified ratings
player_rating_overrides.csv               optional explicit overrides

Outputs
-------
player_ratings_v2.csv
player_ratings_coverage.csv

Rating sources, in priority order
---------------------------------
1. Explicit override in player_rating_overrides.csv
2. Existing verified impact_points from player_ratings.csv
3. No rating (blank)

Experience is carried alongside the rating for future backtesting, but is NOT
automatically converted into impact points in this version.

This file does NOT alter predict.py.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import pandas as pd

MASTER = Path("player_master.csv")
EXPERIENCE = Path("player_experience.csv")
LEGACY_RATINGS = Path("player_ratings.csv")
OVERRIDES = Path("player_rating_overrides.csv")

OUTPUT = Path("player_ratings_v2.csv")
COVERAGE_OUTPUT = Path("player_ratings_coverage.csv")

RATING_COLUMNS = [
    "season",
    "team",
    "player",
    "player_key",
    "primary_position",
    "spine_player",
    "roster_status",
    "impact_points",
    "rating_status",
    "rating_source",
    "nrl_games",
    "nrl_finals_games",
    "previous_grand_finals",
    "grand_final_wins",
    "state_of_origin_games",
    "test_international_games",
    "super_league_games",
    "other_first_class_games",
    "season_games",
    "experience_data_status",
    "source_team_lists",
    "source_roster_seed",
    "last_seen",
    "notes",
]

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
    "perth bears": "Perth Bears",
    "bears": "Perth Bears",
}

EXPERIENCE_FIELDS = [
    "nrl_games",
    "nrl_finals_games",
    "previous_grand_finals",
    "grand_final_wins",
    "state_of_origin_games",
    "test_international_games",
    "super_league_games",
    "other_first_class_games",
    "season_games",
]


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def name_key(value) -> str:
    text = clean(value).lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' -]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_team(value) -> str:
    raw = clean(value)
    return TEAM_ALIASES.get(name_key(raw), raw)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def numeric(value):
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return val


def prep_master(df: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "team", "player"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{MASTER} missing required columns: {', '.join(sorted(missing))}"
        )

    out = df.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out = out[out["season"].notna()].copy()
    out["season"] = out["season"].astype(int)
    out["team"] = out["team"].map(normalise_team)
    out["player"] = out["player"].map(clean)

    if "player_key" not in out.columns:
        out["player_key"] = out["player"].map(name_key)
    else:
        out["player_key"] = out["player_key"].map(clean)
        blank = out["player_key"] == ""
        out.loc[blank, "player_key"] = out.loc[blank, "player"].map(name_key)

    for col, default in [
        ("primary_position", ""),
        ("spine_player", 0),
        ("roster_status", ""),
        ("source_team_lists", 0),
        ("source_roster_seed", 0),
        ("last_seen", ""),
        ("notes", ""),
    ]:
        if col not in out.columns:
            out[col] = default

    out = out[(out["team"] != "") & (out["player_key"] != "")].copy()
    out = out.drop_duplicates(["season", "team", "player_key"], keep="last")
    return out


def prep_experience(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    team_col = first_col(df, ["team", "club"])
    player_col = first_col(df, ["player", "player_name", "name"])
    season_col = first_col(df, ["season", "year"])

    if not team_col or not player_col:
        return {}

    records = {}
    for _, row in df.iterrows():
        team = normalise_team(row.get(team_col, ""))
        player = clean(row.get(player_col, ""))
        if not team or not player:
            continue

        try:
            season = int(float(row.get(season_col, 2027))) if season_col else 2027
        except (TypeError, ValueError):
            season = 2027

        rec = {}
        for field in EXPERIENCE_FIELDS:
            if field in df.columns:
                rec[field] = numeric(row.get(field))
            else:
                rec[field] = pd.NA

        status_col = first_col(
            df, ["experience_data_status", "data_status", "verification_status"]
        )
        rec["experience_data_status"] = (
            clean(row.get(status_col, "")) if status_col else ""
        )

        records[(season, team, name_key(player))] = rec

    return records


def prep_rating_lookup(df: pd.DataFrame, source_label: str) -> dict:
    if df.empty:
        return {}

    team_col = first_col(df, ["team", "club"])
    player_col = first_col(df, ["player", "player_name", "name"])
    season_col = first_col(df, ["season", "year"])
    rating_col = first_col(
        df, ["impact_points", "rating", "player_rating", "impact_rating"]
    )

    if not team_col or not player_col or not rating_col:
        return {}

    lookup = {}
    for _, row in df.iterrows():
        team = normalise_team(row.get(team_col, ""))
        player = clean(row.get(player_col, ""))
        rating = numeric(row.get(rating_col))

        if not team or not player or pd.isna(rating):
            continue

        try:
            season = int(float(row.get(season_col, 2027))) if season_col else 2027
        except (TypeError, ValueError):
            season = 2027

        lookup[(season, team, name_key(player))] = {
            "impact_points": float(rating),
            "rating_source": source_label,
        }

    return lookup


def fallback_lookup(
    lookup: dict,
    season: int,
    team: str,
    player_key_value: str,
):
    """
    Exact season/team/player first.
    If not found, allow same team/player from another season and take the
    most recent available season. This preserves verified ratings through
    offseason roster construction without inventing a value.
    """
    exact = lookup.get((season, team, player_key_value))
    if exact:
        return exact

    candidates = [
        (s, value)
        for (s, t, p), value in lookup.items()
        if t == team and p == player_key_value and s <= season
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def experience_lookup(
    lookup: dict,
    season: int,
    team: str,
    player_key_value: str,
):
    exact = lookup.get((season, team, player_key_value))
    if exact:
        return exact

    candidates = [
        (s, value)
        for (s, t, p), value in lookup.items()
        if t == team and p == player_key_value and s <= season
    ]
    if not candidates:
        return {}

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def build(
    master: pd.DataFrame,
    experience: dict,
    legacy: dict,
    overrides: dict,
) -> pd.DataFrame:
    rows = []

    for _, p in master.iterrows():
        season = int(p["season"])
        team = p["team"]
        pkey = p["player_key"]

        # Explicit override has highest priority.
        rating_rec = fallback_lookup(overrides, season, team, pkey)
        if rating_rec:
            rating = rating_rec["impact_points"]
            rating_source = rating_rec["rating_source"]
            rating_status = "rated"
        else:
            rating_rec = fallback_lookup(legacy, season, team, pkey)
            if rating_rec:
                rating = rating_rec["impact_points"]
                rating_source = rating_rec["rating_source"]
                rating_status = "rated"
            else:
                rating = pd.NA
                rating_source = ""
                rating_status = "unrated"

        exp = experience_lookup(experience, season, team, pkey)

        row = {
            "season": season,
            "team": team,
            "player": p["player"],
            "player_key": pkey,
            "primary_position": clean(p.get("primary_position", "")),
            "spine_player": int(pd.to_numeric(
                pd.Series([p.get("spine_player", 0)]),
                errors="coerce"
            ).fillna(0).iloc[0]),
            "roster_status": clean(p.get("roster_status", "")),
            "impact_points": rating,
            "rating_status": rating_status,
            "rating_source": rating_source,
            "experience_data_status": clean(
                exp.get("experience_data_status", "")
            ),
            "source_team_lists": int(pd.to_numeric(
                pd.Series([p.get("source_team_lists", 0)]),
                errors="coerce"
            ).fillna(0).iloc[0]),
            "source_roster_seed": int(pd.to_numeric(
                pd.Series([p.get("source_roster_seed", 0)]),
                errors="coerce"
            ).fillna(0).iloc[0]),
            "last_seen": clean(p.get("last_seen", "")),
            "notes": clean(p.get("notes", "")),
        }

        for field in EXPERIENCE_FIELDS:
            row[field] = exp.get(field, pd.NA)

        rows.append(row)

    out = pd.DataFrame(rows)
    for col in RATING_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    return out[RATING_COLUMNS].sort_values(
        ["season", "team", "player_key"]
    ).reset_index(drop=True)


def coverage_table(ratings: pd.DataFrame) -> pd.DataFrame:
    if ratings.empty:
        return pd.DataFrame(columns=[
            "season", "team", "players", "rated_players",
            "rating_coverage", "experience_players",
            "experience_coverage"
        ])

    work = ratings.copy()
    work["_rated"] = work["impact_points"].notna().astype(int)

    exp_fields = [
        "nrl_games",
        "nrl_finals_games",
        "state_of_origin_games",
        "test_international_games",
    ]
    available_exp = [c for c in exp_fields if c in work.columns]
    if available_exp:
        work["_experience"] = work[available_exp].notna().any(axis=1).astype(int)
    else:
        work["_experience"] = 0

    rows = []
    for (season, team), grp in work.groupby(["season", "team"]):
        players = len(grp)
        rated = int(grp["_rated"].sum())
        experienced = int(grp["_experience"].sum())

        rows.append({
            "season": int(season),
            "team": team,
            "players": players,
            "rated_players": rated,
            "rating_coverage": round(rated / players, 4) if players else 0,
            "experience_players": experienced,
            "experience_coverage": round(experienced / players, 4)
                if players else 0,
        })

    return pd.DataFrame(rows).sort_values(["season", "team"]).reset_index(drop=True)


def report(ratings: pd.DataFrame, coverage: pd.DataFrame) -> None:
    print("\nPLAYER RATINGS V2")
    print("=" * 64)
    print("Fox Sports stats used: NO")
    print("Automatic experience -> impact weighting: NO")
    print("Unknown players treated as average: NO")
    print(f"Players: {len(ratings)}")

    rated = int(ratings["impact_points"].notna().sum()) if not ratings.empty else 0
    print(f"Currently rated: {rated}/{len(ratings)}")

    if not coverage.empty:
        latest = int(coverage["season"].max())
        print(f"\nCoverage for {latest}:")
        current = coverage[coverage["season"] == latest]
        for r in current.itertuples(index=False):
            print(
                f"  {r.team}: {r.rated_players}/{r.players} rated "
                f"({r.rating_coverage:.0%})"
            )

    print(f"\nSaved: {OUTPUT}")
    print(f"Saved: {COVERAGE_OUTPUT}")
    print("\nNOTE: This is infrastructure, not a live model adjustment.")


def main() -> int:
    try:
        if not MASTER.exists():
            raise FileNotFoundError(
                f"Missing {MASTER}. Run build_player_master.py first."
            )

        master = prep_master(read_csv(MASTER))
        experience = prep_experience(read_csv(EXPERIENCE))
        legacy = prep_rating_lookup(
            read_csv(LEGACY_RATINGS), "existing_verified_rating"
        )
        overrides = prep_rating_lookup(
            read_csv(OVERRIDES), "explicit_override"
        )

        ratings = build(master, experience, legacy, overrides)
        coverage = coverage_table(ratings)

        ratings.to_csv(OUTPUT, index=False)
        coverage.to_csv(COVERAGE_OUTPUT, index=False)

        report(ratings, coverage)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
