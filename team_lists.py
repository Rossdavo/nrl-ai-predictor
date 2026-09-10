import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

TEAM_LIST_URL = (
    "https://www.nrl.com/news/2026/09/10/"
    "nrl-late-mail-finals-week-1/"
)

OUT_PATH = "current_team_lists.csv"
RATINGS_PATH = "player_ratings.csv"

FINALS_TEAMS = {
    "Rabbitohs",
    "Knights",
    "Sharks",
    "Cowboys",
    "Warriors",
    "Dolphins",
    "Panthers",
    "Roosters",
}

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
    return " ".join(str(value).strip().split())


def normalise_position(position):
    p = clean_text(position).lower()

    replacements = {
        "five eighth": "five-eighth",
        "five-eighth": "five-eighth",
        "second row": "2nd row",
        "2nd row": "2nd row",
        "interchange": "interchange",
        "reserve": "reserve",
    }

    return replacements.get(p, p)


def fetch_page():
    print("[teams] Fetching official NRL Finals Week 1 team lists...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120 Safari/537.36"
        )
    }

    response = requests.get(
        TEAM_LIST_URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    return response.text


def extract_team_lists(html):
    soup = BeautifulSoup(html, "html.parser")

    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    rows = []

    positions = (
        "Fullback|Winger|Centre|Five-Eighth|Halfback|"
        "Prop|Hooker|2nd Row|Lock|Interchange|Reserve"
    )

    teams_pattern = "|".join(
        sorted(
            [re.escape(team) for team in FINALS_TEAMS],
            key=len,
            reverse=True,
        )
    )

    pattern = re.compile(
        rf"({positions}) for ({teams_pattern}) "
        rf"is number (\d+) ([A-Za-zÀ-ÖØ-öø-ÿ'’\-\.\s]+?)"
        rf"(?=(?:Fullback|Winger|Centre|Five-Eighth|Halfback|"
        rf"Prop|Hooker|2nd Row|Lock|Interchange|Reserve) "
        rf"for | Match Officials| Team News| Ins | Outs |$)",
        re.IGNORECASE,
    )

    for match in pattern.finditer(text):
        position = normalise_position(match.group(1))
        team = clean_text(match.group(2))
        jersey = int(match.group(3))
        player = clean_text(match.group(4))

        # Remove stray numeric separators occasionally captured from
        # NRL's side-by-side team-list formatting.
        player = re.sub(r"\s+\d+\s*$", "", player).strip()

        if team not in FINALS_TEAMS:
            continue

        if not player:
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

    # NRL pages can repeat team-list information in more than one section.
    df = df.drop_duplicates(
        subset=["team", "jersey", "player"],
        keep="last",
    )

    df = df.sort_values(
        ["team", "jersey", "player"]
    ).reset_index(drop=True)

    return df


def update_player_ratings(team_lists):
    """
    Add newly discovered players to player_ratings.csv.

    Existing ratings are NEVER overwritten.

    New players receive conservative position-based starting ratings.
    We can then manually improve the important-player ratings.
    """

    if os.path.exists(RATINGS_PATH):
        try:
            ratings = pd.read_csv(RATINGS_PATH)
        except Exception:
            ratings = pd.DataFrame()
    else:
        ratings = pd.DataFrame()

    required_columns = [
        "team",
        "player",
        "position",
        "impact_points",
    ]

    if ratings.empty:
        ratings = pd.DataFrame(columns=required_columns)

    for col in required_columns:
        if col not in ratings.columns:
            ratings[col] = ""

    existing = set()

    for _, row in ratings.iterrows():
        key = (
            clean_text(row["team"]).upper(),
            clean_text(row["player"]).upper(),
        )
        existing.add(key)

    new_rows = []

    for _, row in team_lists.iterrows():
        team = clean_text(row["team"])
        player = clean_text(row["player"])
        position = normalise_position(row["position"])

        key = (
            team.upper(),
            player.upper(),
        )

        if key in existing:
            continue

        default_rating = POSITION_DEFAULTS.get(
            position,
            1.0,
        )

        new_rows.append(
            {
                "team": team,
                "player": player,
                "position": position,
                "impact_points": default_rating,
            }
        )

        existing.add(key)

    if new_rows:
        additions = pd.DataFrame(new_rows)

        ratings = pd.concat(
            [ratings, additions],
            ignore_index=True,
            sort=False,
        )

        ratings["impact_points"] = pd.to_numeric(
            ratings["impact_points"],
            errors="coerce",
        )

        ratings = ratings.sort_values(
            ["team", "position", "player"]
        ).reset_index(drop=True)

        ratings.to_csv(
            RATINGS_PATH,
            index=False,
        )

        print(
            f"[teams] Added {len(new_rows)} new players "
            f"to {RATINGS_PATH}"
        )

    else:
        print(
            "[teams] No new players needed in player_ratings.csv"
        )


def main():
    try:
        html = fetch_page()
    except Exception as e:
        print(f"[teams] ERROR fetching NRL team lists: {e}")
        raise

    team_lists = extract_team_lists(html)

    if team_lists.empty:
        raise RuntimeError(
            "No NRL players could be extracted from the official page."
        )

    team_lists.to_csv(
        OUT_PATH,
        index=False,
    )

    print(
        f"[teams] Wrote {len(team_lists)} selections "
        f"to {OUT_PATH}"
    )

    for team in sorted(FINALS_TEAMS):
        count = len(
            team_lists[
                team_lists["team"] == team
            ]
        )

        print(
            f"[teams] {team}: {count} players"
        )

    update_player_ratings(team_lists)

    print("[teams] Team-list update complete.")


if __name__ == "__main__":
    main()
