#!/usr/bin/env python3
"""
build_player_master.py

Experimental player-system foundation for the NRL predictor.

Purpose
-------
Create and maintain one canonical player master for all clubs without using
the separate Fox Sports player-stat experiment.

Inputs (when available)
-----------------------
1. team_lists_history.csv
   - Primary automatic source of players actually selected in NRL team lists.
2. nrl_roster_seed_2026_2027.csv
   - Optional seed for preseason/off-season rosters, including Perth Bears.
3. player_experience.csv
   - Optional enrichment only. Does not create ratings.

Output
------
player_master.csv

Important
---------
- This file does NOT assign player impact ratings.
- It does NOT change predict.py.
- It does NOT use fox_player_stats.csv.
- New players appearing in archived team lists are added automatically.
- Existing manual/verified information is preserved where possible.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import pandas as pd

TEAM_LISTS = Path("team_lists_history.csv")
ROSTER_SEED = Path("nrl_roster_seed_2026_2027.csv")
EXPERIENCE = Path("player_experience.csv")
EXISTING_MASTER = Path("player_master.csv")
OUTPUT = Path("player_master.csv")

CURRENT_SEASON = 2027

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

EXPECTED_2027_TEAMS = [
    "Broncos", "Raiders", "Bulldogs", "Sharks", "Dolphins", "Titans",
    "Sea Eagles", "Storm", "Knights", "Warriors", "Cowboys", "Eels",
    "Panthers", "Rabbitohs", "Dragons", "Roosters", "Wests Tigers",
    "Perth Bears",
]

MASTER_COLUMNS = [
    "season",
    "team",
    "player",
    "player_key",
    "primary_position",
    "last_selected_position",
    "last_selected_jersey",
    "first_seen",
    "last_seen",
    "times_selected",
    "spine_player",
    "roster_status",
    "source_team_lists",
    "source_roster_seed",
    "source_experience",
    "data_status",
    "notes",
]


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def key(value) -> str:
    text = clean(value).lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' -]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_team(value) -> str:
    raw = clean(value)
    return TEAM_ALIASES.get(key(raw), raw)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def first_existing_column(df: pd.DataFrame, names: list[str]) -> str | None:
    lowered = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def infer_season(round_start) -> int:
    dt = pd.to_datetime(round_start, errors="coerce")
    if pd.isna(dt):
        return CURRENT_SEASON
    return int(dt.year)


def build_from_team_lists(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    needed = {"team", "player"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"{TEAM_LISTS} must contain at least: team, player"
        )

    work = df.copy()
    work["team"] = work["team"].map(norm_team)
    work["player"] = work["player"].map(clean)
    work["player_key"] = work["player"].map(key)

    if "round_start" in work.columns:
        work["season"] = work["round_start"].map(infer_season)
        work["_date"] = pd.to_datetime(work["round_start"], errors="coerce")
    else:
        work["season"] = CURRENT_SEASON
        work["_date"] = pd.NaT

    if "position" not in work.columns:
        work["position"] = ""
    if "jersey" not in work.columns:
        work["jersey"] = pd.NA

    work["position"] = work["position"].map(clean)
    work["jersey"] = pd.to_numeric(work["jersey"], errors="coerce")
    work = work[(work["team"] != "") & (work["player_key"] != "")].copy()

    rows = []
    for (season, team, pkey), grp in work.groupby(
        ["season", "team", "player_key"], sort=False
    ):
        grp = grp.sort_values("_date", na_position="first")
        latest = grp.iloc[-1]

        positions = grp.loc[grp["position"] != "", "position"]
        primary_position = (
            positions.mode().iloc[0] if not positions.empty else ""
        )

        last_jersey = latest["jersey"]
        if pd.isna(last_jersey):
            last_jersey_out = ""
            spine = 0
        else:
            last_jersey_out = int(last_jersey)
            spine = int(int(last_jersey) in {1, 6, 7, 9})

        valid_dates = grp["_date"].dropna()
        first_seen = (
            valid_dates.min().date().isoformat() if not valid_dates.empty else ""
        )
        last_seen = (
            valid_dates.max().date().isoformat() if not valid_dates.empty else ""
        )

        rows.append({
            "season": int(season),
            "team": team,
            "player": latest["player"],
            "player_key": pkey,
            "primary_position": primary_position,
            "last_selected_position": latest["position"],
            "last_selected_jersey": last_jersey_out,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "times_selected": int(len(grp)),
            "spine_player": spine,
            "roster_status": "selected",
            "source_team_lists": 1,
            "source_roster_seed": 0,
            "source_experience": 0,
            "data_status": "observed",
            "notes": "",
        })

    return pd.DataFrame(rows, columns=MASTER_COLUMNS)


def build_from_seed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    team_col = first_existing_column(df, ["team", "club"])
    player_col = first_existing_column(df, ["player", "player_name", "name"])
    season_col = first_existing_column(df, ["season", "year"])
    pos_col = first_existing_column(
        df, ["primary_position", "position", "pos"]
    )
    status_col = first_existing_column(
        df, ["roster_status", "status"]
    )

    if not team_col or not player_col:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    rows = []
    for _, r in df.iterrows():
        team = norm_team(r.get(team_col, ""))
        player = clean(r.get(player_col, ""))
        if not team or not player:
            continue

        season_raw = r.get(season_col, CURRENT_SEASON) if season_col else CURRENT_SEASON
        try:
            season = int(float(season_raw))
        except (TypeError, ValueError):
            season = CURRENT_SEASON

        position = clean(r.get(pos_col, "")) if pos_col else ""
        status = clean(r.get(status_col, "")) if status_col else ""
        if not status:
            status = "roster"

        rows.append({
            "season": season,
            "team": team,
            "player": player,
            "player_key": key(player),
            "primary_position": position,
            "last_selected_position": "",
            "last_selected_jersey": "",
            "first_seen": "",
            "last_seen": "",
            "times_selected": 0,
            "spine_player": 0,
            "roster_status": status,
            "source_team_lists": 0,
            "source_roster_seed": 1,
            "source_experience": 0,
            "data_status": "seeded",
            "notes": "",
        })

    return pd.DataFrame(rows, columns=MASTER_COLUMNS)


def experience_keys(df: pd.DataFrame) -> set[tuple[int, str, str]]:
    if df.empty:
        return set()

    team_col = first_existing_column(df, ["team", "club"])
    player_col = first_existing_column(df, ["player", "player_name", "name"])
    season_col = first_existing_column(df, ["season", "year"])

    if not team_col or not player_col:
        return set()

    found = set()
    for _, r in df.iterrows():
        team = norm_team(r.get(team_col, ""))
        player = clean(r.get(player_col, ""))
        if not team or not player:
            continue
        try:
            season = int(float(r.get(season_col, CURRENT_SEASON))) if season_col else CURRENT_SEASON
        except (TypeError, ValueError):
            season = CURRENT_SEASON
        found.add((season, team, key(player)))
    return found


def combine_sources(
    observed: pd.DataFrame,
    seeded: pd.DataFrame,
    existing: pd.DataFrame,
    exp_keys: set[tuple[int, str, str]],
) -> pd.DataFrame:
    records: dict[tuple[int, str, str], dict] = {}

    # Existing first, so verified/manual notes survive rebuilds.
    if not existing.empty:
        for _, r in existing.iterrows():
            try:
                season = int(float(r.get("season", CURRENT_SEASON)))
            except (TypeError, ValueError):
                season = CURRENT_SEASON
            team = norm_team(r.get("team", ""))
            player = clean(r.get("player", ""))
            pkey = clean(r.get("player_key", "")) or key(player)
            if not team or not pkey:
                continue
            rec = {c: r.get(c, "") for c in MASTER_COLUMNS}
            rec["season"] = season
            rec["team"] = team
            rec["player"] = player
            rec["player_key"] = pkey
            records[(season, team, pkey)] = rec

    # Seed provides preseason roster population.
    for _, r in seeded.iterrows():
        k = (int(r["season"]), r["team"], r["player_key"])
        if k not in records:
            records[k] = r.to_dict()
        else:
            rec = records[k]
            rec["source_roster_seed"] = 1
            if not clean(rec.get("primary_position", "")):
                rec["primary_position"] = r["primary_position"]
            if not clean(rec.get("roster_status", "")):
                rec["roster_status"] = r["roster_status"]

    # Actual team-list observations take precedence for dynamic fields.
    for _, r in observed.iterrows():
        k = (int(r["season"]), r["team"], r["player_key"])
        if k not in records:
            records[k] = r.to_dict()
        else:
            rec = records[k]
            rec["player"] = r["player"]
            rec["source_team_lists"] = 1
            rec["last_selected_position"] = r["last_selected_position"]
            rec["last_selected_jersey"] = r["last_selected_jersey"]
            rec["first_seen"] = r["first_seen"] or rec.get("first_seen", "")
            rec["last_seen"] = r["last_seen"]
            rec["times_selected"] = r["times_selected"]
            rec["spine_player"] = r["spine_player"]
            rec["roster_status"] = "selected"
            rec["data_status"] = "observed"
            if not clean(rec.get("primary_position", "")):
                rec["primary_position"] = r["primary_position"]

    for k, rec in records.items():
        if k in exp_keys:
            rec["source_experience"] = 1

        # Normalise integer-ish fields.
        for col in ["source_team_lists", "source_roster_seed",
                    "source_experience", "spine_player", "times_selected"]:
            try:
                rec[col] = int(float(rec.get(col, 0) or 0))
            except (TypeError, ValueError):
                rec[col] = 0

        if not clean(rec.get("data_status", "")):
            rec["data_status"] = (
                "observed" if rec["source_team_lists"] else "seeded"
            )

    out = pd.DataFrame(list(records.values()))
    for c in MASTER_COLUMNS:
        if c not in out.columns:
            out[c] = ""

    out = out[MASTER_COLUMNS].copy()
    out = out.sort_values(
        ["season", "team", "player_key"]
    ).reset_index(drop=True)
    return out


def print_report(master: pd.DataFrame) -> None:
    print("\nPLAYER MASTER BUILD")
    print("=" * 64)
    print("Fox Sports player stats used: NO")
    print(f"Rows: {len(master)}")

    if master.empty:
        print("No players found.")
        return

    latest_season = int(pd.to_numeric(master["season"], errors="coerce").max())
    latest = master[pd.to_numeric(master["season"], errors="coerce") == latest_season]

    print(f"Latest season represented: {latest_season}")
    print(f"Clubs represented: {latest['team'].nunique()}")

    print("\nPlayers by club:")
    counts = latest.groupby("team")["player_key"].nunique().sort_index()
    for team, count in counts.items():
        print(f"  {team}: {count}")

    if latest_season >= 2027:
        missing = [t for t in EXPECTED_2027_TEAMS if t not in set(latest["team"])]
        if missing:
            print("\n2027 clubs not yet populated:")
            for team in missing:
                print(f"  {team}")
        else:
            print("\nAll 18 expected 2027 clubs are represented.")

    observed = int((latest["source_team_lists"] == 1).sum())
    seeded = int((latest["source_roster_seed"] == 1).sum())
    print(f"\nPlayers observed in team lists: {observed}")
    print(f"Players supplied by roster seed: {seeded}")
    print(f"Saved: {OUTPUT}")
    print("\nNOTE: This creates player identity/roster infrastructure only.")
    print("      It does not create ratings or alter predictions.")


def main() -> int:
    try:
        team_lists = read_csv_safe(TEAM_LISTS)
        seed = read_csv_safe(ROSTER_SEED)
        experience = read_csv_safe(EXPERIENCE)
        existing = read_csv_safe(EXISTING_MASTER)

        observed = build_from_team_lists(team_lists)
        seeded = build_from_seed(seed)
        exp_keys = experience_keys(experience)

        master = combine_sources(observed, seeded, existing, exp_keys)
        master.to_csv(OUTPUT, index=False)

        print_report(master)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
