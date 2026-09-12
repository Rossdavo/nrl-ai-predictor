import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup


TEAM_LIST_URLS = [
    "https://www.nrl.com/news/2026/09/10/nrl-late-mail-finals-week-1/",
    "https://www.nrl.com/news/2026/09/08/nrl-team-lists-finals-week-1/",
]

OUT_PATH = "current_team_lists.csv"
RATINGS_PATH = "player_ratings.csv"

FINALS_TEAMS = [
    "Rabbitohs",
    "Knights",
    "Warriors",
    "Dolphins",
    "Sharks",
    "Cowboys",
    "Panthers",
    "Roosters",
]

POSITION_DEFAULTS = {
    "fullback": 4.0,
    "winger": 1.2,
    "centre": 1.6,
    "five-eighth": 4.2,
    "halfback": 5.0,
    "prop": 2.5,
    "hooker": 4.0,
    "2nd row": 2.0,
    "lock": 2.5,
    "interchange": 1.0,
    "reserve": 0.7,
}


def clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def normalise_position(position):
    p = clean_text(position).lower()

    mapping = {
        "fullback": "fullback",
        "winger": "winger",
        "centre": "centre",
        "five-eighth": "five-eighth",
        "five eighth": "five-eighth",
        "halfback": "halfback",
        "prop": "prop",
        "hooker": "hooker",
        "2nd row": "2nd row",
        "second row": "2nd row",
        "lock": "lock",
        "interchange": "interchange",
        "reserve": "reserve",
    }

    return mapping.get(p, p)


def fetch_page(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    print(f"[teams] Fetching: {url}")

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    return response.text


def extract_team_lists(html):
    soup = BeautifulSoup(html, "html.parser")

    # Flatten visible article text because the NRL page often
    # splits opposing teams across different HTML elements.
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    positions = (
        r"Fullback|Winger|Centre|Five-Eighth|Five Eighth|"
        r"Halfback|Prop|Hooker|2nd Row|Second Row|Lock|"
        r"Interchange|Reserve"
    )

    teams = (
        r"Rabbitohs|Knights|Warriors|Dolphins|"
        r"Sharks|Cowboys|Panthers|Roosters"
    )

    pattern = re.compile(
        rf"({positions})\s+for\s+({teams})\s+"
        rf"is\s+number\s+(\d+)\s+"
        rf"([A-Za-zÀ-ÖØ-öø-ÿĀ-ž'’\-. ]+?)"
        rf"(?="
        rf"\s+(?:\d+\s+)?(?:{positions})\s+for\s+(?:{teams})\s+"
        rf"|"
        rf"\s+(?:Backs|Forwards|Interchange|Reserves|"
        rf"Match Officials|Last updated:|Team News|"
        rf"Rabbitohs Ins|Rabbitohs Outs|"
        rf"Knights Ins|Knights Outs|"
        rf"Warriors Ins|Warriors Outs|"
        rf"Dolphins Ins|Dolphins Outs|"
        rf"Sharks Ins|Sharks Outs|"
        rf"Cowboys Ins|Cowboys Outs|"
        rf"Panthers Ins|Panthers Outs|"
        rf"Roosters Ins|Roosters Outs)"
        rf"|$"
        rf")",
        re.IGNORECASE,
    )

    rows = []

    team_lookup = {
        x.lower(): x
        for x in FINALS_TEAMS
    }

    for match in pattern.finditer(text):
        position = normalise_position(
            match.group(1)
        )

        team = team_lookup.get(
            clean_text(
                match.group(2)
            ).lower()
        )

        jersey = int(
            match.group(3)
        )

        player = clean_text(
            match.group(4)
        )

        # Remove an accidental trailing jersey number.
        player = re.sub(
            r"\s+\d+$",
            "",
            player,
        ).strip()

        if team not in FINALS_TEAMS:
            continue

        if not player:
            continue

        if jersey < 1 or jersey > 30:
            continue

        rows.append(
            {
                "team": team,
                "jersey": jersey,
                "player": player,
                "position": position,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.drop_duplicates(
        subset=[
            "team",
            "jersey",
            "player",
            "position",
        ],
        keep="last",
    )

    df = df.sort_values(
        [
            "team",
            "jersey",
            "player",
        ]
    ).reset_index(drop=True)

    return df


def load_previous_team_lists():
    if not os.path.exists(OUT_PATH):
        print(
            "[teams] No previous "
            f"{OUT_PATH} available."
        )
        return pd.DataFrame()

    try:
        previous = pd.read_csv(
            OUT_PATH
        )
    except Exception as exc:
        print(
            "[teams] WARNING: Could not read previous "
            f"{OUT_PATH}: {exc}"
        )
        return pd.DataFrame()

    required = {
        "team",
        "jersey",
        "player",
        "position",
    }

    if not required.issubset(
        previous.columns
    ):
        print(
            "[teams] WARNING: Previous "
            f"{OUT_PATH} does not contain "
            "the required columns."
        )
        return pd.DataFrame()

    print(
        f"[teams] Loaded previous "
        f"{OUT_PATH}: {len(previous)} players"
    )

    return previous


def get_best_team_list():
    scraped_sources = []

    for url in TEAM_LIST_URLS:
        try:
            html = fetch_page(
                url
            )

            df = extract_team_lists(
                html
            )

            complete_count = 0

            for team in FINALS_TEAMS:
                if df.empty:
                    count = 0
                else:
                    count = len(
                        df[
                            df["team"] == team
                        ]
                    )

                if count >= 17:
                    complete_count += 1

            print(
                f"[teams] Extraction result: "
                f"{len(df)} players, "
                f"{complete_count}/8 teams "
                f"with at least 17 players"
            )

            for team in FINALS_TEAMS:
                if df.empty:
                    count = 0
                else:
                    count = len(
                        df[
                            df["team"] == team
                        ]
                    )

                print(
                    f"[teams]   "
                    f"{team}: {count}"
                )

            scraped_sources.append(
                {
                    "url": url,
                    "df": df,
                }
            )

        except Exception as exc:
            print(
                f"[teams] WARNING: "
                f"{url}: {exc}"
            )

    if not scraped_sources:
        raise RuntimeError(
            "Could not fetch any NRL "
            "team-list source."
        )

    previous = load_previous_team_lists()

    final_blocks = []

    print("")
    print(
        "[teams] SELECTING BEST TEAM "
        "BLOCKS"
    )

    for team in FINALS_TEAMS:
        best_fresh_df = pd.DataFrame()
        best_fresh_count = 0
        best_fresh_url = None

        # Find the largest extraction for this
        # individual team across all current sources.
        #
        # If counts are tied, the first URL wins.
        # TEAM_LIST_URLS has Late Mail first.
        for source in scraped_sources:
            df = source["df"]

            if df.empty:
                continue

            team_df = df[
                df["team"] == team
            ].copy()

            count = len(
                team_df
            )

            if count > best_fresh_count:
                best_fresh_df = team_df
                best_fresh_count = count
                best_fresh_url = source["url"]

        # Fresh complete team takes priority.
        if best_fresh_count >= 17:
            print(
                f"[teams] {team}: "
                f"using fresh source "
                f"({best_fresh_count} players)"
            )

            print(
                f"[teams]   Source: "
                f"{best_fresh_url}"
            )

            final_blocks.append(
                best_fresh_df
            )

            continue

        # Fresh extraction is incomplete.
        print(
            f"[teams] {team}: "
            f"fresh extraction incomplete "
            f"({best_fresh_count} players)"
        )

        previous_team_df = pd.DataFrame()

        if not previous.empty:
            previous_team_df = previous[
                previous["team"] == team
            ].copy()

        previous_count = len(
            previous_team_df
        )

        # Use the previous known-good team
        # only if it has at least 17 players.
        if previous_count >= 17:
            print(
                f"[teams] {team}: "
                f"FALLBACK to previous "
                f"{OUT_PATH} "
                f"({previous_count} players)"
            )

            final_blocks.append(
                previous_team_df
            )

            continue

        # No safe source exists.
        print(
            f"[teams] {team}: "
            f"ERROR - no complete fresh "
            f"or previous team available"
        )

        # Keep whatever fresh data exists.
        # Validation below will deliberately fail,
        # which prevents overwriting the good files.
        if not best_fresh_df.empty:
            final_blocks.append(
                best_fresh_df
            )

    if not final_blocks:
        raise RuntimeError(
            "Could not build any team-list data."
        )

    final_df = pd.concat(
        final_blocks,
        ignore_index=True,
    )

    final_df = final_df.drop_duplicates(
        subset=[
            "team",
            "jersey",
            "player",
            "position",
        ],
        keep="last",
    )

    team_order = {
        team: index
        for index, team
        in enumerate(FINALS_TEAMS)
    }

    final_df["_team_order"] = (
        final_df["team"]
        .map(team_order)
        .fillna(999)
    )

    final_df = final_df.sort_values(
        [
            "_team_order",
            "jersey",
            "player",
        ]
    )

    final_df = final_df.drop(
        columns=[
            "_team_order",
        ]
    ).reset_index(drop=True)

    return final_df


def validate_team_lists(df):
    problems = []

    print("")
    print(
        "[teams] FINAL TEAM COUNTS"
    )

    for team in FINALS_TEAMS:
        count = len(
            df[
                df["team"] == team
            ]
        )

        print(
            f"[teams] "
            f"{team}: {count} players"
        )

        if count < 17:
            problems.append(
                f"{team} only has {count}"
            )

    if problems:
        print("")
        print(
            "[teams] VALIDATION FAILED"
        )

        for problem in problems:
            print(
                f"[teams] ERROR: "
                f"{problem}"
            )

        raise RuntimeError(
            "Incomplete team-list extraction. "
            "No files were updated."
        )

    print("")
    print(
        "[teams] VALIDATION PASSED: "
        "all 8 finals teams found."
    )


def update_player_ratings(team_lists):
    columns = [
        "team",
        "player",
        "position",
        "impact_points",
    ]

    if os.path.exists(
        RATINGS_PATH
    ):
        try:
            ratings = pd.read_csv(
                RATINGS_PATH
            )
        except Exception:
            ratings = pd.DataFrame(
                columns=columns
            )
    else:
        ratings = pd.DataFrame(
            columns=columns
        )

    for col in columns:
        if col not in ratings.columns:
            ratings[col] = ""

    existing = set()

    for _, row in ratings.iterrows():
        team = clean_text(
            row.get(
                "team",
                "",
            )
        ).upper()

        player = clean_text(
            row.get(
                "player",
                "",
            )
        ).upper()

        if team and player:
            existing.add(
                (
                    team,
                    player,
                )
            )

    additions = []

    for _, row in team_lists.iterrows():
        team = clean_text(
            row["team"]
        )

        player = clean_text(
            row["player"]
        )

        position = normalise_position(
            row["position"]
        )

        key = (
            team.upper(),
            player.upper(),
        )

        if key in existing:
            continue

        impact = POSITION_DEFAULTS.get(
            position,
            1.0,
        )

        additions.append(
            {
                "team": team,
                "player": player,
                "position": position,
                "impact_points": impact,
            }
        )

        existing.add(
            key
        )

    if not additions:
        print(
            "[teams] No new players "
            "needed in player_ratings.csv"
        )
        return

    additions_df = pd.DataFrame(
        additions
    )

    ratings = pd.concat(
        [
            ratings,
            additions_df,
        ],
        ignore_index=True,
        sort=False,
    )

    ratings[
        "impact_points"
    ] = pd.to_numeric(
        ratings[
            "impact_points"
        ],
        errors="coerce",
    )

    ratings = ratings.sort_values(
        [
            "team",
            "position",
            "player",
        ]
    ).reset_index(
        drop=True
    )

    ratings.to_csv(
        RATINGS_PATH,
        index=False,
    )

    print(
        f"[teams] Added "
        f"{len(additions)} new players "
        f"to {RATINGS_PATH}"
    )


def main():
    print(
        "[teams] Fetching official "
        "NRL Finals Week 1 team lists..."
    )

    team_lists = (
        get_best_team_list()
    )

    # This validation happens AFTER fresh data
    # and previous-file fallbacks have been combined.
    #
    # Files are still never overwritten unless every
    # team has at least 17 players.
    validate_team_lists(
        team_lists
    )

    team_lists.to_csv(
        OUT_PATH,
        index=False,
    )

    print(
        f"[teams] Wrote "
        f"{len(team_lists)} selections "
        f"to {OUT_PATH}"
    )

    update_player_ratings(
        team_lists
    )

    print(
        "[teams] Team-list update complete."
    )


if __name__ == "__main__":
    main()
