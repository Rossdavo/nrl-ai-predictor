import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup


# ============================================================
# NRL TEAM LIST COLLECTOR
# Finals Week 1 - 2026
# ============================================================

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

VALID_POSITIONS = {
    "fullback",
    "winger",
    "centre",
    "five-eighth",
    "halfback",
    "prop",
    "hooker",
    "2nd row",
    "lock",
    "interchange",
    "reserve",
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
    """
    Extract team selections from the official NRL article.

    Important:
    We preserve newline boundaries instead of flattening the entire
    article into one long string. The earlier version flattened the
    article and caused alternating teams to be swallowed by the regex.
    """

    soup = BeautifulSoup(html, "html.parser")

    text = soup.get_text(
        separator="\n",
        strip=True,
    )

    # Collapse repeated spaces but KEEP line breaks.
    cleaned_lines = []

    for raw_line in text.splitlines():
        line = clean_text(raw_line)

        if line:
            cleaned_lines.append(line)

    rows = []

    position_pattern = (
        r"(Fullback|Winger|Centre|Five-Eighth|Five Eighth|"
        r"Halfback|Prop|Hooker|2nd Row|Second Row|Lock|"
        r"Interchange|Reserve)"
    )

    team_pattern = (
        r"(Rabbitohs|Knights|Warriors|Dolphins|"
        r"Sharks|Cowboys|Panthers|Roosters)"
    )

    player_pattern = (
        r"([A-Za-zÀ-ÖØ-öø-ÿĀ-ž'’\-\.\s]+)"
    )

    full_pattern = re.compile(
        rf"^{position_pattern}\s+for\s+{team_pattern}"
        rf"\s+is\s+number\s+(\d+)\s+{player_pattern}$",
        re.IGNORECASE,
    )

    for line in cleaned_lines:
        match = full_pattern.match(line)

        if not match:
            continue

        position = normalise_position(match.group(1))
        team_raw = clean_text(match.group(2))
        jersey = int(match.group(3))
        player = clean_text(match.group(4))

        # Standardise team capitalisation.
        team_lookup = {
            x.lower(): x
            for x in FINALS_TEAMS
        }

        team = team_lookup.get(
            team_raw.lower(),
            team_raw,
        )

        if team not in FINALS_TEAMS:
            continue

        if position not in VALID_POSITIONS:
            continue

        # Sanity checks.
        if jersey < 1 or jersey > 30:
            continue

        if len(player) < 2:
            continue

        # Ignore obvious article words if somehow captured.
        bad_words = {
            "team",
            "lists",
            "ins",
            "outs",
            "news",
        }

        if player.lower() in bad_words:
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

    # Some NRL articles can contain the same team list twice.
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


def score_extraction(df):
    """
    Give an extraction a simple score.

    Each finals club should normally have at least 17 named players,
    usually around 21-23 including reserves.
    """

    if df.empty:
        return 0

    score = 0

    for team in FINALS_TEAMS:
        count = len(
            df[df["team"] == team]
        )

        if count >= 17:
            score += 1

    return score


def get_best_team_list():
    """
    Try Late Mail first, then the original official team-list article.

    Use whichever page produces the most complete set of teams.
    """

    best_df = pd.DataFrame()
    best_score = -1
    best_url = None

    for url in TEAM_LIST_URLS:
        try:
            html = fetch_page(url)

            df = extract_team_lists(html)

            score = score_extraction(df)

            print(
                f"[teams] Extraction result: "
                f"{len(df)} players, "
                f"{score}/8 teams with at least 17 players"
            )

            if score > best_score:
                best_df = df
                best_score = score
                best_url = url

        except Exception as exc:
            print(
                f"[teams] WARNING: Could not read {url}: {exc}"
            )

    if best_df.empty:
        raise RuntimeError(
            "Could not extract any NRL team-list data."
        )

    print(
        f"[teams] Best source: {best_url}"
    )

    return best_df


def validate_team_lists(df):
    """
    Fail safely.

    We do NOT want incomplete team lists feeding the predictor.
    """

    problems = []

    print("")
    print("[teams] Team counts:")

    for team in FINALS_TEAMS:
        count = len(
            df[df["team"] == team]
        )

        print(
            f"[teams] {team}: {count} players"
        )

        if count < 17:
            problems.append(
                f"{team} only has {count} players"
            )

    if problems:
        print("")
        print("[teams] VALIDATION FAILED")

        for problem in problems:
            print(
                f"[teams] ERROR: {problem}"
            )

        raise RuntimeError(
            "Incomplete NRL team-list extraction. "
            "current_team_lists.csv was NOT updated."
        )

    print("")
    print(
        "[teams] Validation passed: "
        "all 8 finals teams found."
    )


def update_player_ratings(team_lists):
    """
    Add newly discovered players to player_ratings.csv.

    Existing individual ratings are NEVER overwritten.
    """

    required_columns = [
        "team",
        "player",
        "position",
        "impact_points",
    ]

    if os.path.exists(RATINGS_PATH):
        try:
            ratings = pd.read_csv(RATINGS_PATH)
        except Exception:
            ratings = pd.DataFrame(
                columns=required_columns
            )
    else:
        ratings = pd.DataFrame(
            columns=required_columns
        )

    for col in required_columns:
        if col not in ratings.columns:
            ratings[col] = ""

    existing = set()

    for _, row in ratings.iterrows():
        team = clean_text(
            row.get("team", "")
        ).upper()

        player = clean_text(
            row.get("player", "")
        ).upper()

        if team and player:
            existing.add(
                (team, player)
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

        existing.add(key)

    if not additions:
        print(
            "[teams] No new players needed "
            "in player_ratings.csv"
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

    ratings["impact_points"] = pd.to_numeric(
        ratings["impact_points"],
        errors="coerce",
    )

    ratings = ratings.sort_values(
        [
            "team",
            "position",
            "player",
        ]
    ).reset_index(drop=True)

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

    team_lists = get_best_team_list()

    # IMPORTANT:
    # Validate BEFORE writing anything.
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
