import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd


# ============================================================
# NRL AUTOMATIC TEAM CHANGE BUILDER
# ============================================================
#
# Compares:
#
# LAST ROUND'S FINAL 1-17
#            vs
# CURRENT ROUND'S 1-17
#
# Then creates team_changes.csv for predict.py.
#
# IMPORTANT:
# We compare against the previous ROUND, not yesterday's
# scrape. This prevents daily workflow runs from wiping out
# genuine team changes.
#
# ============================================================


CURRENT_TEAMS_FILE = "current_team_lists.csv"
PLAYER_RATINGS_FILE = "player_ratings.csv"
ODDS_FILE = "odds.csv"

HISTORY_FILE = "team_lists_history.csv"
CHANGES_FILE = "team_changes.csv"


CHANGE_COLUMNS = [
    "date",
    "team",
    "player",
    "status",
    "replacement_player",
    "position",
    "impact_points",
    "replacement_impact_points",
    "notes",
]


HISTORY_COLUMNS = [
    "round_id",
    "round_start",
    "round_end",
    "captured_at",
    "team",
    "jersey",
    "player",
    "position",
]


# ============================================================
# HELPERS
# ============================================================


def clean_text(value):

    if value is None:
        return ""

    return " ".join(
        str(value).strip().split()
    )


def team_key(value):

    return clean_text(
        value
    ).casefold()


def player_key(value):

    return clean_text(
        value
    ).casefold()


def normalise_position(value):

    p = clean_text(
        value
    ).lower()

    mapping = {

        "fullback": "fullback",

        "wing": "wing",
        "winger": "wing",

        "centre": "centre",
        "center": "centre",

        "five-eighth": "five eighth",
        "five eighth": "five eighth",
        "5/8": "five eighth",

        "half": "halfback",
        "halfback": "halfback",

        "prop": "prop",

        "hooker": "hooker",

        "second row": "second row",
        "2nd row": "second row",
        "back row": "second row",

        "lock": "lock",

        "interchange": "bench",
        "bench": "bench",

        "reserve": "reserve",
        "reserves": "reserve",
    }

    return mapping.get(
        p,
        p,
    )


# ============================================================
# LOAD CURRENT TEAM LIST
# ============================================================


def load_current_team():

    if not os.path.exists(
        CURRENT_TEAMS_FILE
    ):

        raise FileNotFoundError(
            f"{CURRENT_TEAMS_FILE} not found. "
            "Run team_lists.py first."
        )

    df = pd.read_csv(
        CURRENT_TEAMS_FILE
    )

    required = {
        "team",
        "jersey",
        "player",
        "position",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            f"{CURRENT_TEAMS_FILE} "
            f"missing columns: {sorted(missing)}"
        )

    df = df.copy()

    df["team"] = (
        df["team"]
        .map(clean_text)
    )

    df["player"] = (
        df["player"]
        .map(clean_text)
    )

    df["position"] = (
        df["position"]
        .map(normalise_position)
    )

    df["jersey"] = pd.to_numeric(
        df["jersey"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["jersey"]
    )

    df["jersey"] = (
        df["jersey"]
        .astype(int)
    )

    df = df[
        (df["team"] != "")
        &
        (df["player"] != "")
    ].copy()

    # --------------------------------------------------------
    # Only the selected match-day 1-17 affects availability.
    # Reserves 18-22 are not treated as starting changes.
    # --------------------------------------------------------

    active = df[
        (df["jersey"] >= 1)
        &
        (df["jersey"] <= 17)
    ].copy()

    teams = sorted(
        df["team"].unique()
    )

    problems = []

    for team in teams:

        count = (
            active[
                active["team"] == team
            ]["player"]
            .nunique()
        )

        if count < 17:

            problems.append(
                f"{team}: {count}"
            )

    if problems:

        raise RuntimeError(
            "Active 1-17 validation failed: "
            + ", ".join(problems)
        )

    return (
        active
        .sort_values(
            [
                "team",
                "jersey",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# PLAYER RATINGS
# ============================================================


def load_player_ratings():

    if not os.path.exists(
        PLAYER_RATINGS_FILE
    ):

        raise FileNotFoundError(
            f"{PLAYER_RATINGS_FILE} not found."
        )

    df = pd.read_csv(
        PLAYER_RATINGS_FILE
    )

    required = {
        "team",
        "player",
        "impact_points",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            f"{PLAYER_RATINGS_FILE} "
            f"missing columns: {sorted(missing)}"
        )

    ratings = {}

    for _, row in df.iterrows():

        team = team_key(
            row.get(
                "team",
                "",
            )
        )

        player = player_key(
            row.get(
                "player",
                "",
            )
        )

        impact = pd.to_numeric(
            row.get(
                "impact_points",
                None,
            ),
            errors="coerce",
        )

        if (
            not team
            or
            not player
            or
            pd.isna(impact)
        ):
            continue

        impact = max(
            0.0,
            min(
                6.5,
                float(impact),
            ),
        )

        ratings[
            (
                team,
                player,
            )
        ] = impact

    print(
        f"[changes] Loaded "
        f"{len(ratings)} player ratings"
    )

    return ratings


def get_player_rating(
    team,
    player,
    ratings,
):

    return float(
        ratings.get(
            (
                team_key(team),
                player_key(player),
            ),
            1.0,
        )
    )


# ============================================================
# CURRENT ROUND / MATCH DATES
# ============================================================


def get_round_information(
    current,
):

    sydney = ZoneInfo(
        "Australia/Sydney"
    )

    today = (
        datetime
        .now(sydney)
        .date()
    )

    dates = []

    fixture_dates = {}

    if os.path.exists(
        ODDS_FILE
    ):

        try:

            odds = pd.read_csv(
                ODDS_FILE
            )

            required = {
                "date",
                "home",
                "away",
            }

            if required.issubset(
                odds.columns
            ):

                odds = odds.copy()

                odds["parsed_date"] = (
                    pd.to_datetime(
                        odds["date"],
                        errors="coerce",
                    )
                )

                odds = odds.dropna(
                    subset=[
                        "parsed_date"
                    ]
                )

                for _, row in odds.iterrows():

                    game_date = (
                        row[
                            "parsed_date"
                        ]
                        .date()
                    )

                    date_string = (
                        game_date
                        .isoformat()
                    )

                    dates.append(
                        game_date
                    )

                    fixture_dates[
                        team_key(
                            row["home"]
                        )
                    ] = date_string

                    fixture_dates[
                        team_key(
                            row["away"]
                        )
                    ] = date_string

        except Exception as exc:

            print(
                "[changes] WARNING: "
                f"could not read "
                f"{ODDS_FILE}: {exc}"
            )

    if dates:

        round_start = min(
            dates
        )

        round_end = max(
            dates
        )

    else:

        round_start = today
        round_end = today

    round_id = (
        f"{round_start.isoformat()}"
        f"_to_"
        f"{round_end.isoformat()}"
    )

    fallback_date = (
        round_start.isoformat()
    )

    for team in (
        current["team"]
        .unique()
    ):

        fixture_dates.setdefault(
            team_key(team),
            fallback_date,
        )

    return (
        round_id,
        round_start.isoformat(),
        round_end.isoformat(),
        fixture_dates,
    )


# ============================================================
# HISTORY
# ============================================================


def load_history():

    if not os.path.exists(
        HISTORY_FILE
    ):

        return pd.DataFrame(
            columns=HISTORY_COLUMNS
        )

    try:

        history = pd.read_csv(
            HISTORY_FILE
        )

    except Exception:

        return pd.DataFrame(
            columns=HISTORY_COLUMNS
        )

    for column in HISTORY_COLUMNS:

        if column not in history.columns:

            history[column] = ""

    history = history[
        HISTORY_COLUMNS
    ].copy()

    history["team"] = (
        history["team"]
        .map(clean_text)
    )

    history["player"] = (
        history["player"]
        .map(clean_text)
    )

    history["position"] = (
        history["position"]
        .map(normalise_position)
    )

    history["jersey"] = (
        pd.to_numeric(
            history["jersey"],
            errors="coerce",
        )
    )

    return history


# ============================================================
# FIND PREVIOUS ROUND FOR TEAM
# ============================================================


def previous_round_team(
    history,
    team,
    current_round_start,
):

    if history.empty:

        return (
            pd.DataFrame(
                columns=HISTORY_COLUMNS
            ),
            None,
        )

    team_history = history[
        history["team"]
        .map(team_key)
        ==
        team_key(team)
    ].copy()

    if team_history.empty:

        return (
            pd.DataFrame(
                columns=HISTORY_COLUMNS
            ),
            None,
        )

    team_history[
        "_round_start"
    ] = pd.to_datetime(
        team_history[
            "round_start"
        ],
        errors="coerce",
    )

    current_start = (
        pd.to_datetime(
            current_round_start,
            errors="coerce",
        )
    )

    # --------------------------------------------------------
    # THIS IS IMPORTANT:
    #
    # Only use a genuinely EARLIER round.
    #
    # We deliberately ignore earlier workflow runs from the
    # current week.
    # --------------------------------------------------------

    team_history = (
        team_history[
            team_history[
                "_round_start"
            ]
            <
            current_start
        ]
        .copy()
    )

    if team_history.empty:

        return (
            pd.DataFrame(
                columns=HISTORY_COLUMNS
            ),
            None,
        )

    latest_round_start = (
        team_history[
            "_round_start"
        ]
        .max()
    )

    team_history = (
        team_history[
            team_history[
                "_round_start"
            ]
            ==
            latest_round_start
        ]
        .copy()
    )

    # In case multiple snapshots somehow exist,
    # use the most recent one.

    latest_capture = (
        team_history[
            "captured_at"
        ]
        .astype(str)
        .max()
    )

    team_history = (
        team_history[
            team_history[
                "captured_at"
            ]
            .astype(str)
            ==
            latest_capture
        ]
        .copy()
    )

    if team_history.empty:

        return (
            pd.DataFrame(
                columns=HISTORY_COLUMNS
            ),
            None,
        )

    previous_round_id = clean_text(
        team_history[
            "round_id"
        ].iloc[0]
    )

    return (
        team_history[
            HISTORY_COLUMNS
        ].copy(),
        previous_round_id,
    )


# ============================================================
# IDENTIFY PERSONNEL CHANGES
# ============================================================


def identify_changes(
    previous,
    current,
):

    previous = previous.copy()
    current = current.copy()

    previous["_player_key"] = (
        previous["player"]
        .map(player_key)
    )

    current["_player_key"] = (
        current["player"]
        .map(player_key)
    )

    previous_players = set(
        previous[
            "_player_key"
        ]
    )

    current_players = set(
        current[
            "_player_key"
        ]
    )

    unchanged = (
        previous_players
        &
        current_players
    )

    outs = previous[
        ~previous[
            "_player_key"
        ]
        .isin(unchanged)
    ].copy()

    ins = current[
        ~current[
            "_player_key"
        ]
        .isin(unchanged)
    ].copy()

    return (
        outs,
        ins,
    )


# ============================================================
# MATCH OUTGOING PLAYERS TO REPLACEMENTS
# ============================================================


def match_replacements(
    outs,
    ins,
):

    pairs = []

    used_out = set()
    used_in = set()

    # --------------------------------------------------------
    # 1. SAME JERSEY
    #
    # Best indicator of direct replacement.
    # --------------------------------------------------------

    for out_index, out_row in (
        outs.iterrows()
    ):

        candidates = ins[
            (~ins.index.isin(
                used_in
            ))
            &
            (
                ins["jersey"]
                ==
                out_row["jersey"]
            )
        ]

        if candidates.empty:
            continue

        in_index = (
            candidates
            .index[0]
        )

        pairs.append(
            (
                out_index,
                in_index,
                "same jersey",
            )
        )

        used_out.add(
            out_index
        )

        used_in.add(
            in_index
        )

    # --------------------------------------------------------
    # 2. SAME POSITION
    # --------------------------------------------------------

    for out_index, out_row in (
        outs.iterrows()
    ):

        if out_index in used_out:
            continue

        candidates = ins[
            (~ins.index.isin(
                used_in
            ))
            &
            (
                ins["position"]
                ==
                out_row["position"]
            )
        ]

        if candidates.empty:
            continue

        in_index = (
            candidates
            .index[0]
        )

        pairs.append(
            (
                out_index,
                in_index,
                "same position",
            )
        )

        used_out.add(
            out_index
        )

        used_in.add(
            in_index
        )

    # --------------------------------------------------------
    # 3. REMAINING PLAYERS
    #
    # Team reshuffles can mean jersey numbers and positions
    # both change.
    #
    # Pair remaining changes using nearest jersey number.
    # --------------------------------------------------------

    remaining_out = [
        index
        for index in outs.index
        if index not in used_out
    ]

    remaining_in = [
        index
        for index in ins.index
        if index not in used_in
    ]

    while (
        remaining_out
        and
        remaining_in
    ):

        out_index = (
            remaining_out
            .pop(0)
        )

        out_row = outs.loc[
            out_index
        ]

        in_index = min(
            remaining_in,
            key=lambda index:
                abs(
                    int(
                        ins.loc[
                            index,
                            "jersey",
                        ]
                    )
                    -
                    int(
                        out_row[
                            "jersey"
                        ]
                    )
                ),
        )

        remaining_in.remove(
            in_index
        )

        pairs.append(
            (
                out_index,
                in_index,
                "team reshuffle",
            )
        )

        used_out.add(
            out_index
        )

        used_in.add(
            in_index
        )

    return (
        pairs,
        used_out,
        used_in,
    )


# ============================================================
# BUILD MODEL CHANGE ROWS
# ============================================================


def build_team_changes(
    previous,
    current,
    team,
    match_date,
    previous_round_id,
    ratings,
):

    rows = []

    outs, ins = identify_changes(
        previous,
        current,
    )

    pairs, used_out, used_in = (
        match_replacements(
            outs,
            ins,
        )
    )

    # --------------------------------------------------------
    # MATCHED SWAPS
    # --------------------------------------------------------

    for (
        out_index,
        in_index,
        match_reason,
    ) in pairs:

        old = outs.loc[
            out_index
        ]

        new = ins.loc[
            in_index
        ]

        old_rating = (
            get_player_rating(
                team,
                old["player"],
                ratings,
            )
        )

        new_rating = (
            get_player_rating(
                team,
                new["player"],
                ratings,
            )
        )

        # ----------------------------------------------------
        # If current player is weaker:
        #
        # OLD PLAYER = OUT
        #
        # Example:
        # Cleary 6.5 -> replacement 2.2
        #
        # predictor receives -4.3
        # ----------------------------------------------------

        if (
            old_rating
            >
            new_rating + 0.05
        ):

            player = (
                old["player"]
            )

            replacement = (
                new["player"]
            )

            status = "out"

            position = (
                old["position"]
            )

            impact = old_rating

            replacement_impact = (
                new_rating
            )

        # ----------------------------------------------------
        # If current player is stronger:
        #
        # CURRENT PLAYER = IN
        #
        # Example:
        # replacement 2.2 -> Cleary 6.5
        #
        # predictor receives +4.3
        # ----------------------------------------------------

        elif (
            new_rating
            >
            old_rating + 0.05
        ):

            player = (
                new["player"]
            )

            replacement = (
                old["player"]
            )

            status = "in"

            position = (
                new["position"]
            )

            impact = new_rating

            replacement_impact = (
                old_rating
            )

        # ----------------------------------------------------
        # Effectively neutral swap.
        #
        # "change" is intentionally ignored by predict.py.
        # We still retain it for auditing.
        # ----------------------------------------------------

        else:

            player = (
                old["player"]
            )

            replacement = (
                new["player"]
            )

            status = "change"

            position = (
                old["position"]
            )

            impact = old_rating

            replacement_impact = (
                new_rating
            )

        notes = (
            f"auto vs {previous_round_id}; "
            f"{match_reason}; "
            f"{int(old['jersey'])} "
            f"{old['player']} -> "
            f"{int(new['jersey'])} "
            f"{new['player']}"
        )

        rows.append(
            {
                "date": match_date,
                "team": team,
                "player": player,
                "status": status,
                "replacement_player":
                    replacement,
                "position": position,
                "impact_points":
                    round(
                        impact,
                        2,
                    ),
                "replacement_impact_points":
                    round(
                        replacement_impact,
                        2,
                    ),
                "notes": notes,
            }
        )

    # --------------------------------------------------------
    # UNMATCHED OUTS
    #
    # Should be rare because both sides contain 17 players.
    # --------------------------------------------------------

    for out_index, old in (
        outs.iterrows()
    ):

        if out_index in used_out:
            continue

        old_rating = (
            get_player_rating(
                team,
                old["player"],
                ratings,
            )
        )

        rows.append(
            {
                "date": match_date,
                "team": team,
                "player":
                    old["player"],
                "status": "out",
                "replacement_player": "",
                "position":
                    old["position"],
                "impact_points":
                    round(
                        old_rating,
                        2,
                    ),
                "replacement_impact_points":
                    0.0,
                "notes":
                    (
                        f"auto vs "
                        f"{previous_round_id}; "
                        f"unmatched OUT "
                        f"jersey "
                        f"{int(old['jersey'])}"
                    ),
            }
        )

    # --------------------------------------------------------
    # UNMATCHED INS
    # --------------------------------------------------------

    for in_index, new in (
        ins.iterrows()
    ):

        if in_index in used_in:
            continue

        new_rating = (
            get_player_rating(
                team,
                new["player"],
                ratings,
            )
        )

        rows.append(
            {
                "date": match_date,
                "team": team,
                "player":
                    new["player"],
                "status": "in",
                "replacement_player": "",
                "position":
                    new["position"],
                "impact_points":
                    round(
                        new_rating,
                        2,
                    ),
                "replacement_impact_points":
                    0.0,
                "notes":
                    (
                        f"auto vs "
                        f"{previous_round_id}; "
                        f"unmatched IN "
                        f"jersey "
                        f"{int(new['jersey'])}"
                    ),
            }
        )

    return rows


# ============================================================
# SAVE CURRENT ROUND SNAPSHOT
# ============================================================


def save_current_snapshot(
    history,
    current,
    round_id,
    round_start,
    round_end,
):

    sydney = ZoneInfo(
        "Australia/Sydney"
    )

    captured_at = (
        datetime
        .now(sydney)
        .isoformat(
            timespec="seconds"
        )
    )

    snapshot = (
        current.copy()
    )

    snapshot[
        "round_id"
    ] = round_id

    snapshot[
        "round_start"
    ] = round_start

    snapshot[
        "round_end"
    ] = round_end

    snapshot[
        "captured_at"
    ] = captured_at

    snapshot = snapshot[
        [
            "round_id",
            "round_start",
            "round_end",
            "captured_at",
            "team",
            "jersey",
            "player",
            "position",
        ]
    ]

    if history.empty:

        combined = snapshot

    else:

        # ----------------------------------------------------
        # Replace this week's snapshot with the newest one.
        #
        # That means next round uses the FINAL version of
        # this week's selected team as its baseline.
        # ----------------------------------------------------

        history = history[
            history[
                "round_id"
            ].astype(str)
            !=
            round_id
        ].copy()

        combined = pd.concat(
            [
                history[
                    HISTORY_COLUMNS
                ],
                snapshot,
            ],
            ignore_index=True,
        )

    combined.to_csv(
        HISTORY_FILE,
        index=False,
    )

    print(
        "[changes] Saved current "
        f"round snapshot: "
        f"{len(snapshot)} players "
        f"-> {HISTORY_FILE}"
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print(
        "[changes] Building automatic "
        "NRL team changes..."
    )

    current = (
        load_current_team()
    )

    ratings = (
        load_player_ratings()
    )

    history = (
        load_history()
    )

    (
        round_id,
        round_start,
        round_end,
        fixture_dates,
    ) = get_round_information(
        current
    )

    print(
        f"[changes] Current round: "
        f"{round_id}"
    )

    all_rows = []

    baseline_team_count = 0

    teams = sorted(
        current[
            "team"
        ].unique()
    )

    for team in teams:

        current_team = current[
            current["team"]
            ==
            team
        ].copy()

        (
            previous_team,
            previous_round_id,
        ) = previous_round_team(
            history,
            team,
            round_start,
        )

        # ----------------------------------------------------
        # FIRST RUN
        #
        # There is nothing reliable to compare against yet.
        # Establish baseline instead of inventing changes.
        # ----------------------------------------------------

        if previous_team.empty:

            print(
                f"[changes] {team}: "
                "no previous-round "
                "baseline yet"
            )

            continue

        baseline_team_count += 1

        match_date = (
            fixture_dates.get(
                team_key(team),
                round_start,
            )
        )

        rows = build_team_changes(
            previous_team,
            current_team,
            team,
            match_date,
            previous_round_id,
            ratings,
        )

        all_rows.extend(
            rows
        )

        impact_rows = [
            row
            for row in rows
            if row["status"]
            in {
                "out",
                "in",
            }
            and
            abs(
                float(
                    row[
                        "impact_points"
                    ]
                )
                -
                float(
                    row[
                        "replacement_impact_points"
                    ]
                )
            )
            >
            0.05
        ]

        print(
            f"[changes] {team}: "
            f"{len(rows)} personnel "
            f"changes, "
            f"{len(impact_rows)} "
            f"impact changes "
            f"vs {previous_round_id}"
        )

        for row in impact_rows:

            difference = abs(
                float(
                    row[
                        "impact_points"
                    ]
                )
                -
                float(
                    row[
                        "replacement_impact_points"
                    ]
                )
            )

            if (
                row["status"]
                ==
                "in"
            ):
                sign = "+"
            else:
                sign = "-"

            print(
                "[changes]   "
                f"{row['status'].upper()} "
                f"{row['player']} / "
                f"{row['replacement_player']} "
                f"({sign}{difference:.1f})"
            )

    # ========================================================
    # WRITE team_changes.csv
    # ========================================================

    output = pd.DataFrame(
        all_rows,
        columns=CHANGE_COLUMNS,
    )

    output.to_csv(
        CHANGES_FILE,
        index=False,
    )

    if baseline_team_count == 0:

        print(
            "[changes] First run: "
            "no earlier round is stored."
        )

        print(
            "[changes] "
            f"{CHANGES_FILE} is "
            "intentionally empty."
        )

        print(
            "[changes] Current teams "
            "are now the baseline "
            "for the next round."
        )

    else:

        impact_count = 0

        if not output.empty:

            impact_count = int(
                output[
                    "status"
                ]
                .isin(
                    [
                        "out",
                        "in",
                    ]
                )
                .sum()
            )

        print(
            f"[changes] Wrote "
            f"{len(output)} comparison "
            f"rows "
            f"({impact_count} "
            f"model-impact rows) "
            f"-> {CHANGES_FILE}"
        )

    # ========================================================
    # STORE THIS ROUND
    # ========================================================

    save_current_snapshot(
        history,
        current,
        round_id,
        round_start,
        round_end,
    )

    print(
        "[changes] Automatic "
        "team-change build complete."
    )


if __name__ == "__main__":

    main()
