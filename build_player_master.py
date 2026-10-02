#!/usr/bin/env python3

"""
build_player_master.py

League-wide player master builder for the NRL AI Predictor.

Population sources:
1. rlp_season_players.csv
   Primary 2026 NRL population.

2. nrl_roster_seed_2026_2027.csv
   Future/offseason roster seed including Perth Bears.

3. team_lists_history.csv
   Adds jersey, position and selection-history information where available.

4. Existing player_master.csv
   Preserves useful information already collected.

5. player_experience.csv
   Optional enrichment flag only.

This script does NOT:
- create player ratings
- use Fox Sports data
- alter predict.py
- alter live predictions
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


RLP_POPULATION = Path(
    "rlp_season_players.csv"
)

ROSTER_SEED = Path(
    "nrl_roster_seed_2026_2027.csv"
)

TEAM_LISTS = Path(
    "team_lists_history.csv"
)

EXISTING_MASTER = Path(
    "player_master.csv"
)

EXPERIENCE = Path(
    "player_experience.csv"
)

OUTPUT = Path(
    "player_master.csv"
)


EXPECTED_2026_TEAMS = {
    "Broncos",
    "Raiders",
    "Bulldogs",
    "Sharks",
    "Dolphins",
    "Titans",
    "Sea Eagles",
    "Storm",
    "Knights",
    "Cowboys",
    "Warriors",
    "Eels",
    "Panthers",
    "Rabbitohs",
    "Dragons",
    "Roosters",
    "Wests Tigers",
}


EXPECTED_2027_TEAMS = (
    EXPECTED_2026_TEAMS
    | {"Perth Bears"}
)


TEAM_ALIASES = {

    "brisbane broncos":
        "Broncos",

    "broncos":
        "Broncos",

    "canberra raiders":
        "Raiders",

    "raiders":
        "Raiders",

    "canterbury bankstown bulldogs":
        "Bulldogs",

    "canterbury-bankstown bulldogs":
        "Bulldogs",

    "bulldogs":
        "Bulldogs",

    "cronulla sharks":
        "Sharks",

    "cronulla-sutherland sharks":
        "Sharks",

    "sharks":
        "Sharks",

    "dolphins":
        "Dolphins",

    "the dolphins":
        "Dolphins",

    "gold coast titans":
        "Titans",

    "titans":
        "Titans",

    "manly sea eagles":
        "Sea Eagles",

    "manly warringah sea eagles":
        "Sea Eagles",

    "sea eagles":
        "Sea Eagles",

    "melbourne storm":
        "Storm",

    "storm":
        "Storm",

    "newcastle knights":
        "Knights",

    "knights":
        "Knights",

    "north queensland cowboys":
        "Cowboys",

    "cowboys":
        "Cowboys",

    "new zealand warriors":
        "Warriors",

    "nz warriors":
        "Warriors",

    "warriors":
        "Warriors",

    "parramatta eels":
        "Eels",

    "eels":
        "Eels",

    "penrith panthers":
        "Panthers",

    "panthers":
        "Panthers",

    "south sydney rabbitohs":
        "Rabbitohs",

    "rabbitohs":
        "Rabbitohs",

    "st george illawarra dragons":
        "Dragons",

    "dragons":
        "Dragons",

    "sydney roosters":
        "Roosters",

    "roosters":
        "Roosters",

    "wests tigers":
        "Wests Tigers",

    "tigers":
        "Wests Tigers",

    "perth bears":
        "Perth Bears",
}


OUTPUT_COLUMNS = [

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

    "source_rlp_season",
    "source_team_lists",
    "source_roster_seed",
    "source_experience",

    "data_status",

    "notes",
]


def clean(value):

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except (
        TypeError,
        ValueError,
    ):

        pass

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def ascii_text(value):

    text = unicodedata.normalize(
        "NFKD",
        clean(value),
    )

    return "".join(
        character
        for character in text
        if not unicodedata.combining(
            character
        )
    )


def norm_key(value):

    text = (
        ascii_text(value)
        .lower()
        .replace(
            "’",
            "'",
        )
    )

    text = re.sub(
        r"[^a-z0-9' -]",
        "",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def make_player_key(name):

    return norm_key(
        name
    )


def normalise_team(team):

    raw = clean(
        team
    )

    return TEAM_ALIASES.get(
        norm_key(raw),
        raw,
    )


def as_int(
    value,
    default=0,
):

    try:

        if pd.isna(value):
            return default

        return int(
            float(value)
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


def read_optional(path):

    if not path.exists():

        return pd.DataFrame()

    try:

        return pd.read_csv(
            path
        )

    except pd.errors.EmptyDataError:

        return pd.DataFrame()


def safe_date(value):

    value = clean(
        value
    )

    if not value:
        return ""

    parsed = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(parsed):

        return value

    return parsed.strftime(
        "%Y-%m-%d"
    )


def blank_record(
    season,
    team,
    player,
):

    return {

        "season":
            int(season),

        "team":
            normalise_team(team),

        "player":
            clean(player),

        "player_key":
            make_player_key(
                player
            ),

        "primary_position":
            "",

        "last_selected_position":
            "",

        "last_selected_jersey":
            "",

        "first_seen":
            "",

        "last_seen":
            "",

        "times_selected":
            0,

        "spine_player":
            False,

        "roster_status":
            "",

        "source_rlp_season":
            False,

        "source_team_lists":
            False,

        "source_roster_seed":
            False,

        "source_experience":
            False,

        "data_status":
            "",

        "notes":
            "",
    }


def record_id(
    season,
    team,
    player_key,
):

    return (
        int(season),
        normalise_team(team),
        clean(player_key),
    )


def load_existing_master():

    records = {}

    old = read_optional(
        EXISTING_MASTER
    )

    if old.empty:

        return records


    for _, row in old.iterrows():

        season = as_int(
            row.get(
                "season"
            ),
            2026,
        )

        team = normalise_team(
            row.get(
                "team",
                "",
            )
        )

        player = clean(
            row.get(
                "player",
                "",
            )
        )

        key = (
            clean(
                row.get(
                    "player_key",
                    "",
                )
            )
            or make_player_key(
                player
            )
        )


        if (
            not team
            or not player
            or not key
        ):

            continue


        rec = blank_record(
            season,
            team,
            player,
        )

        rec["player_key"] = key


        for col in OUTPUT_COLUMNS:

            if col not in old.columns:
                continue

            value = row.get(
                col
            )


            if col in {

                "source_rlp_season",
                "source_team_lists",
                "source_roster_seed",
                "source_experience",
                "spine_player",

            }:

                if isinstance(
                    value,
                    str,
                ):

                    rec[col] = (
                        value
                        .strip()
                        .lower()
                        in {
                            "1",
                            "true",
                            "yes",
                            "y",
                        }
                    )

                else:

                    rec[col] = (
                        bool(value)
                        if pd.notna(value)
                        else False
                    )


            elif col == "times_selected":

                rec[col] = as_int(
                    value,
                    0,
                )


            elif col not in {

                "season",
                "team",
                "player",
                "player_key",

            }:

                rec[col] = clean(
                    value
                )


        records[
            record_id(
                season,
                team,
                key,
            )
        ] = rec


    return records


def apply_rlp_population(
    records,
):

    if not RLP_POPULATION.exists():

        raise FileNotFoundError(

            "Required population file "
            f"not found: "
            f"{RLP_POPULATION}. "
            "Run "
            "collect_rlp_season_players.py "
            "first."
        )


    df = pd.read_csv(
        RLP_POPULATION
    )


    required = {
        "season",
        "team",
        "player",
    }


    missing = (
        required
        - set(df.columns)
    )


    if missing:

        raise RuntimeError(

            f"{RLP_POPULATION} "
            "missing required "
            f"columns: "
            f"{sorted(missing)}"
        )


    for _, row in df.iterrows():

        season = as_int(
            row.get(
                "season"
            ),
            2026,
        )


        team = normalise_team(
            row.get(
                "team",
                "",
            )
        )


        player = clean(
            row.get(
                "player",
                "",
            )
        )


        key = (
            clean(
                row.get(
                    "player_key",
                    "",
                )
            )
            or make_player_key(
                player
            )
        )


        if (
            not team
            or not player
            or not key
        ):

            continue


        rid = record_id(
            season,
            team,
            key,
        )


        rec = records.get(

            rid,

            blank_record(
                season,
                team,
                player,
            ),
        )


        rec["player"] = (
            player
        )


        rec["player_key"] = (
            key
        )


        source = clean(
            row.get(
                "source",
                "",
            )
        )


        if (
            source
            == "rlp_season_players"
        ):

            rec[
                "source_rlp_season"
            ] = True


        position = clean(
            row.get(
                "primary_position",
                "",
            )
        )


        if position:

            rec[
                "primary_position"
            ] = position


        status = clean(
            row.get(
                "roster_status",
                "",
            )
        )


        if status:

            rec[
                "roster_status"
            ] = status


        row_status = clean(
            row.get(
                "data_status",
                "",
            )
        )


        if season == 2026:

            rec[
                "data_status"
            ] = (
                "rlp_2026_population"
            )


        elif (
            row_status
            == "seed_only"
        ):

            rec[
                "source_roster_seed"
            ] = True

            rec[
                "data_status"
            ] = "future_seed"


        records[rid] = rec


def apply_roster_seed(
    records,
):

    seed = read_optional(
        ROSTER_SEED
    )


    if seed.empty:

        return


    if not {
        "team",
        "player",
    }.issubset(
        seed.columns
    ):

        return


    for _, row in seed.iterrows():

        season = as_int(
            row.get(
                "season"
            ),
            2027,
        )


        team = normalise_team(
            row.get(
                "team",
                "",
            )
        )


        player = clean(
            row.get(
                "player",
                "",
            )
        )


        key = make_player_key(
            player
        )


        if (
            not team
            or not player
            or not key
        ):

            continue


        rid = record_id(
            season,
            team,
            key,
        )


        rec = records.get(

            rid,

            blank_record(
                season,
                team,
                player,
            ),
        )


        rec[
            "source_roster_seed"
        ] = True


        position = clean(
            row.get(
                "primary_position",
                "",
            )
        )


        if (
            position
            and not clean(
                rec.get(
                    "primary_position"
                )
            )
        ):

            rec[
                "primary_position"
            ] = position


        status = clean(
            row.get(
                "roster_status",
                "",
            )
        )


        if status:

            rec[
                "roster_status"
            ] = status


        if season > 2026:

            rec[
                "data_status"
            ] = "future_seed"


        records[rid] = rec


def infer_snapshot_date(
    row,
):

    for col in (
        "captured_at",
        "round_end",
        "round_start",
    ):

        value = safe_date(
            row.get(
                col,
                "",
            )
        )

        if value:

            return value


    return ""


def team_list_enrichment():

    df = read_optional(
        TEAM_LISTS
    )


    if (
        df.empty
        or not {
            "team",
            "player",
        }.issubset(
            df.columns
        )
    ):

        return {}


    work = df.copy()


    work[
        "team_norm"
    ] = work[
        "team"
    ].map(
        normalise_team
    )


    work[
        "player_clean"
    ] = work[
        "player"
    ].map(
        clean
    )


    work[
        "player_key_calc"
    ] = work[
        "player_clean"
    ].map(
        make_player_key
    )


    work[
        "seen_date"
    ] = work.apply(
        infer_snapshot_date,
        axis=1,
    )


    if (
        "jersey"
        not in work.columns
    ):

        work[
            "jersey"
        ] = pd.NA


    if (
        "position"
        not in work.columns
    ):

        work[
            "position"
        ] = ""


    enrich = {}


    grouped = work.groupby(

        [
            "team_norm",
            "player_key_calc",
        ],

        dropna=False,
    )


    for (
        team,
        key,
    ), group in grouped:

        if (
            not team
            or not key
        ):

            continue


        g = group.copy()


        g[
            "_seen_sort"
        ] = pd.to_datetime(

            g[
                "seen_date"
            ],

            errors="coerce",
        )


        g = g.sort_values(

            [
                "_seen_sort"
            ],

            na_position="first",
        )


        latest = g.iloc[-1]


        dates = [

            d

            for d in g[
                "seen_date"
            ].tolist()

            if clean(d)
        ]


        jersey = as_int(
            latest.get(
                "jersey"
            ),
            0,
        )


        last_position = clean(
            latest.get(
                "position",
                "",
            )
        )


        enrich[
            (
                team,
                key,
            )
        ] = {

            "last_selected_position":
                last_position,

            "last_selected_jersey":
                (
                    jersey
                    if jersey
                    else ""
                ),

            "first_seen":
                (
                    min(dates)
                    if dates
                    else ""
                ),

            "last_seen":
                (
                    max(dates)
                    if dates
                    else ""
                ),

            "times_selected":
                len(g),

            "spine_player":
                jersey
                in {
                    1,
                    6,
                    7,
                    9,
                },
        }


    return enrich


def apply_team_lists(
    records,
):

    enrich = (
        team_list_enrichment()
    )


    for rec in records.values():

        key = (
            rec["team"],
            rec["player_key"],
        )


        info = enrich.get(
            key
        )


        if not info:

            continue


        rec[
            "source_team_lists"
        ] = True


        rec[
            "last_selected_position"
        ] = info[
            "last_selected_position"
        ]


        rec[
            "last_selected_jersey"
        ] = info[
            "last_selected_jersey"
        ]


        rec[
            "first_seen"
        ] = info[
            "first_seen"
        ]


        rec[
            "last_seen"
        ] = info[
            "last_seen"
        ]


        rec[
            "times_selected"
        ] = info[
            "times_selected"
        ]


        rec[
            "spine_player"
        ] = bool(
            info[
                "spine_player"
            ]
        )


        if (
            not clean(
                rec.get(
                    "primary_position"
                )
            )
            and info[
                "last_selected_position"
            ]
        ):

            rec[
                "primary_position"
            ] = info[
                "last_selected_position"
            ]


def apply_experience_flag(
    records,
):

    exp = read_optional(
        EXPERIENCE
    )


    if (
        exp.empty
        or "player"
        not in exp.columns
    ):

        return


    generic_keys = set()

    exact_keys = set()


    for _, row in exp.iterrows():

        player = clean(
            row.get(
                "player",
                "",
            )
        )


        if not player:

            continue


        key = make_player_key(
            player
        )


        generic_keys.add(
            key
        )


        if (
            "team"
            in exp.columns
        ):

            team = normalise_team(
                row.get(
                    "team",
                    "",
                )
            )

        else:

            team = ""


        if (
            "season"
            in exp.columns
        ):

            season = as_int(
                row.get(
                    "season"
                ),
                0,
            )

        else:

            season = 0


        if (
            season
            and team
        ):

            exact_keys.add(
                (
                    season,
                    team,
                    key,
                )
            )


    for rec in records.values():

        exact = (
            rec["season"],
            rec["team"],
            rec["player_key"],
        )


        if (
            exact
            in exact_keys
            or rec[
                "player_key"
            ]
            in generic_keys
        ):

            rec[
                "source_experience"
            ] = True


def remove_stale_population(
    records,
):

    cleaned = {}


    for rid, rec in records.items():

        season = rec[
            "season"
        ]


        if (
            season == 2026
            and not rec[
                "source_rlp_season"
            ]
        ):

            continue


        if (
            season > 2026
            and not rec[
                "source_roster_seed"
            ]
        ):

            continue


        cleaned[rid] = rec


    return cleaned


def finalise(
    records,
):

    records = (
        remove_stale_population(
            records
        )
    )


    rows = []


    for rec in records.values():

        if not rec[
            "roster_status"
        ]:

            if (
                rec["season"]
                == 2026
            ):

                rec[
                    "roster_status"
                ] = (
                    "appeared_in_nrl_season"
                )

            else:

                rec[
                    "roster_status"
                ] = "roster_seed"


        if not rec[
            "data_status"
        ]:

            if (
                rec["season"]
                == 2026
            ):

                rec[
                    "data_status"
                ] = (
                    "rlp_2026_population"
                )

            else:

                rec[
                    "data_status"
                ] = "future_seed"


        rows.append(
            {
                col:
                    rec.get(
                        col,
                        "",
                    )

                for col
                in OUTPUT_COLUMNS
            }
        )


    if not rows:

        raise RuntimeError(
            "Player master "
            "would be empty"
        )


    out = pd.DataFrame(

        rows,

        columns=OUTPUT_COLUMNS,
    )


    out[
        "season"
    ] = pd.to_numeric(

        out[
            "season"
        ],

        errors="coerce",

    ).astype(
        "Int64"
    )


    out[
        "times_selected"
    ] = pd.to_numeric(

        out[
            "times_selected"
        ],

        errors="coerce",

    ).fillna(
        0
    ).astype(
        int
    )


    out = (

        out

        .drop_duplicates(

            [
                "season",
                "team",
                "player_key",
            ],

            keep="last",
        )

        .sort_values(

            [
                "season",
                "team",
                "player_key",
            ]
        )

        .reset_index(
            drop=True
        )
    )


    return out


def print_summary(
    df,
):

    print(
        "\n=== PLAYER MASTER SUMMARY ==="
    )


    print(
        f"Rows: {len(df)}"
    )


    print(
        "Unique player identities: "
        f"{df['player_key'].nunique()}"
    )


    seasons = sorted(

        df[
            "season"
        ]

        .dropna()

        .astype(int)

        .unique()
    )


    for season in seasons:

        part = df[
            df["season"]
            == season
        ]


        teams = sorted(

            part[
                "team"
            ]

            .dropna()

            .astype(str)

            .unique()
        )


        print(
            f"\n=== {season} "
            "PLAYERS BY CLUB ==="
        )


        print(

            part

            .groupby(
                "team"
            )[
                "player_key"
            ]

            .nunique()

            .sort_index()

            .to_string()
        )


        print(
            f"\n{season} clubs: "
            f"{len(teams)}"
        )


        if season == 2026:

            missing = sorted(

                EXPECTED_2026_TEAMS
                - set(teams)
            )


            print(

                "Missing 2026 clubs: "

                + (

                    "NONE"

                    if not missing

                    else ", ".join(
                        missing
                    )
                )
            )


        if season == 2027:

            print(

                "2027 seeded clubs "
                "currently represented: "

                + (

                    ", ".join(
                        teams
                    )

                    if teams

                    else "NONE"
                )
            )


    print(
        "\n=== SOURCE COVERAGE ==="
    )


    for col in (

        "source_rlp_season",
        "source_team_lists",
        "source_roster_seed",
        "source_experience",

    ):

        count = int(

            df[
                col
            ]

            .astype(bool)

            .sum()
        )


        print(
            f"{col}: {count}"
        )


    selected = int(

        (
            df[
                "times_selected"
            ]
            > 0
        ).sum()
    )


    print(

        "Rows enriched by "
        "archived team selections: "
        f"{selected}"
    )


def main():

    print(
        "PLAYER MASTER v2 - "
        "LEAGUE-WIDE RLP POPULATION"
    )


    print(
        "Fox Sports data: NOT USED"
    )


    print(
        "Primary population: "
        f"{RLP_POPULATION}"
    )


    records = (
        load_existing_master()
    )


    print(
        "Existing master rows "
        "loaded for preservation: "
        f"{len(records)}"
    )


    apply_rlp_population(
        records
    )


    apply_roster_seed(
        records
    )


    apply_team_lists(
        records
    )


    apply_experience_flag(
        records
    )


    output = finalise(
        records
    )


    current = output[
        output[
            "season"
        ]
        == 2026
    ]


    found_2026 = set(

        current[
            "team"
        ]

        .dropna()

        .astype(str)
    )


    missing_2026 = (

        EXPECTED_2026_TEAMS
        - found_2026
    )


    if missing_2026:

        raise RuntimeError(

            "Refusing to write "
            "player_master.csv because "
            "2026 population is incomplete. "
            "Missing: "
            f"{sorted(missing_2026)}"
        )


    unique_2026 = (

        current[
            "player_key"
        ]

        .nunique()
    )


    if unique_2026 < 450:

        raise RuntimeError(

            "Refusing to write "
            "player_master.csv because "
            "fewer than 450 unique "
            "2026 players were found."
        )


    output.to_csv(
        OUTPUT,
        index=False,
    )


    print_summary(
        output
    )


    print(
        f"\nSaved: {OUTPUT}"
    )


    print(
        "COMPLETE"
    )


    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
