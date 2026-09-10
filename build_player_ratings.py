import os
import pandas as pd


# ============================================================
# NRL PLAYER IMPACT RATING BUILDER
# ============================================================
#
# impact_points =
# estimated value of the player compared with a normal
# replacement-level player, expressed in expected match
# margin points.
#
# IMPORTANT:
# These are model priors, NOT official NRL ratings.
#
# The predictor ultimately uses:
#
# missing player's rating - replacement player's rating
#
# Example:
# Nathan Cleary 6.5
# replacement half 2.2
# actual team penalty = 4.3 points
#
# ============================================================


TEAM_LIST_FILE = "current_team_lists.csv"
RATINGS_FILE = "player_ratings.csv"


# ------------------------------------------------------------
# POSITIONAL STARTING VALUES
# ------------------------------------------------------------
#
# These are deliberately conservative.
#
# Player quality overrides below are more important than
# position alone.
#

POSITION_BASE = {
    "fullback": 3.2,
    "winger": 1.2,
    "centre": 1.5,
    "five-eighth": 3.4,
    "halfback": 3.8,
    "prop": 2.0,
    "hooker": 3.0,
    "2nd row": 1.8,
    "lock": 2.2,
    "interchange": 1.0,
    "reserve": 0.6,
}


# ------------------------------------------------------------
# MANUAL PLAYER QUALITY OVERRIDES
# ------------------------------------------------------------
#
# Scale:
#
# 6.0-6.5  Transformational superstar
# 5.0-5.9  Elite / major match influence
# 4.0-4.9  Very high impact player
# 3.0-3.9  Important established starter
# 2.0-2.9  Strong first-grade player
# 1.2-1.9  Solid starter / useful player
# 0.7-1.1  Bench / replacement level
# 0.0-0.6  Fringe replacement
#
# Ratings represent VALUE ABOVE A REPLACEMENT,
# not simply how good the player is.
#

PLAYER_OVERRIDES = {

    # ========================================================
    # RABBITOHS
    # ========================================================

    ("Rabbitohs", "Matthew Dufty"): 3.0,
    ("Rabbitohs", "Alex Johnston"): 1.9,
    ("Rabbitohs", "Latrell Mitchell"): 4.7,
    ("Rabbitohs", "Jack Wighton"): 2.5,
    ("Rabbitohs", "Campbell Graham"): 2.2,
    ("Rabbitohs", "Cody Walker"): 4.4,
    ("Rabbitohs", "Jayden Sullivan"): 2.8,
    ("Rabbitohs", "Tevita Tatola"): 2.3,
    ("Rabbitohs", "Brandon Smith"): 3.4,
    ("Rabbitohs", "Keaon Koloamatangi"): 3.3,
    ("Rabbitohs", "David Fifita"): 3.2,
    ("Rabbitohs", "Tallis Duncan"): 2.3,
    ("Rabbitohs", "Cameron Murray"): 4.4,
    ("Rabbitohs", "Jye Gray"): 1.8,
    ("Rabbitohs", "Lachlan Hubner"): 1.1,
    ("Rabbitohs", "Sean Keppie"): 1.5,
    ("Rabbitohs", "Jamie Humphreys"): 1.4,
    ("Rabbitohs", "Adam Elliott"): 1.7,

    # ========================================================
    # KNIGHTS
    # ========================================================

    ("Knights", "Kalyn Ponga"): 5.5,
    ("Knights", "Dominic Young"): 2.0,
    ("Knights", "Dane Gagai"): 2.0,
    ("Knights", "Bradman Best"): 2.7,
    ("Knights", "Greg Marzhew"): 1.6,
    ("Knights", "Fletcher Sharpe"): 2.9,
    ("Knights", "Sandon Smith"): 2.6,
    ("Knights", "Tyson Frizell"): 2.5,
    ("Knights", "Phoenix Crossland"): 2.9,
    ("Knights", "Cody Hopwood"): 1.5,
    ("Knights", "Jermaine McEwen"): 1.6,
    ("Knights", "Francis Manuleleua"): 1.8,
    ("Knights", "Mat Croker"): 2.0,
    ("Knights", "Harrison Graham"): 1.1,
    ("Knights", "Lachlan Crouch"): 1.0,
    ("Knights", "Pasami Saulo"): 1.2,
    ("Knights", "Thomas Cant"): 1.0,

    # ========================================================
    # WARRIORS
    # ========================================================

    ("Warriors", "Taine Tuaupiki"): 2.8,
    ("Warriors", "Charnze Nicoll-Klokstad"): 2.4,
    ("Warriors", "Ali Leiataua"): 1.7,
    ("Warriors", "Leka Halasima"): 2.2,
    ("Warriors", "Alofiana Khan-Pereira"): 1.7,
    ("Warriors", "Chanel Harris-Tavita"): 3.0,
    ("Warriors", "Te Maire Martin"): 3.3,
    ("Warriors", "James Fisher-Harris"): 3.8,
    ("Warriors", "Wayde Egan"): 4.0,
    ("Warriors", "Mitchell Barnett"): 3.4,
    ("Warriors", "Kurt Capewell"): 2.4,
    ("Warriors", "Jacob Laban"): 1.7,
    ("Warriors", "Erin Clark"): 2.8,
    ("Warriors", "Sam Healey"): 1.0,
    ("Warriors", "Tanner Stowers-Smith"): 1.1,
    ("Warriors", "Demitric Vaimauga"): 1.1,
    ("Warriors", "Jackson Ford"): 1.8,

    # ========================================================
    # DOLPHINS
    # ========================================================

    ("Dolphins", "Hamiso Tabuai-Fidow"): 5.0,
    ("Dolphins", "Jamayne Isaako"): 2.1,
    ("Dolphins", "Jack Bostock"): 1.7,
    ("Dolphins", "Herbie Farnworth"): 3.0,
    ("Dolphins", "Selwyn Cobbo"): 2.4,
    ("Dolphins", "Kodi Nikorima"): 3.2,
    ("Dolphins", "Isaiya Katoa"): 4.6,
    ("Dolphins", "Thomas Flegler"): 3.2,
    ("Dolphins", "Jeremy Marshall-King"): 3.7,
    ("Dolphins", "Tom Gilbert"): 3.0,
    ("Dolphins", "Max Plath"): 2.8,
    ("Dolphins", "Kulikefu Finefeuiaki"): 2.2,
    ("Dolphins", "Morgan Knowles"): 2.7,
    ("Dolphins", "Kurt Donoghoe"): 1.1,
    ("Dolphins", "Ray Stone"): 1.3,
    ("Dolphins", "Francis Molo"): 1.7,
    ("Dolphins", "Connelly Lemuelu"): 1.5,
    ("Dolphins", "Felise Kaufusi"): 1.7,
    ("Dolphins", "Bradley Schneider"): 1.7,
    ("Dolphins", "Jake Averillo"): 1.3,

    # ========================================================
    # SHARKS
    # ========================================================

    ("Sharks", "William Kennedy"): 3.0,
    ("Sharks", "Sione Katoa"): 1.5,
    ("Sharks", "Riley Jones"): 1.3,
    ("Sharks", "KL Iro"): 2.0,
    ("Sharks", "Ronaldo Mulitalo"): 1.9,
    ("Sharks", "Braydon Trindall"): 4.1,
    ("Sharks", "Nicho Hynes"): 4.8,
    ("Sharks", "Addin Fonua-Blake"): 4.0,
    ("Sharks", "Blayke Brailey"): 4.0,
    ("Sharks", "Jesse Colquhoun"): 1.8,
    ("Sharks", "Briton Nikora"): 2.4,
    ("Sharks", "Teig Wilton"): 2.2,
    ("Sharks", "Cameron McInnes"): 3.3,
    ("Sharks", "Siosifa Talakai"): 1.8,
    ("Sharks", "Billy Burns"): 1.1,
    ("Sharks", "Oregon Kaufusi"): 2.1,
    ("Sharks", "Thomas Hazelton"): 2.0,
    ("Sharks", "Toby Rudolf"): 1.7,

    # ========================================================
    # COWBOYS
    # ========================================================

    ("Cowboys", "Scott Drinkwater"): 4.7,
    ("Cowboys", "Braidon Burns"): 1.2,
    ("Cowboys", "Jaxon Purdue"): 1.9,
    ("Cowboys", "Tom Chester"): 1.7,
    ("Cowboys", "Zac Laybutt"): 1.4,
    ("Cowboys", "Jake Clifford"): 3.1,
    ("Cowboys", "Tom Dearden"): 4.8,
    ("Cowboys", "Griffin Neame"): 2.1,
    ("Cowboys", "Reed Mahoney"): 3.7,
    ("Cowboys", "Jason Taumalolo"): 3.2,
    ("Cowboys", "Heilum Luki"): 2.2,
    ("Cowboys", "Jeremiah Nanai"): 2.8,
    ("Cowboys", "Reuben Cotter"): 3.8,
    ("Cowboys", "Soni Luke"): 1.2,
    ("Cowboys", "John Bateman"): 1.7,
    ("Cowboys", "Thomas Mikaele"): 1.4,
    ("Cowboys", "Coen Hess"): 1.6,

    # ========================================================
    # PANTHERS
    # ========================================================

    ("Panthers", "Dylan Edwards"): 5.0,
    ("Panthers", "Thomas Jenkins"): 1.2,
    ("Panthers", "Luke Garner"): 1.9,
    ("Panthers", "Casey McLean"): 1.9,
    ("Panthers", "Brian To'o"): 2.5,
    ("Panthers", "Jack Cole"): 2.8,
    ("Panthers", "Nathan Cleary"): 6.5,
    ("Panthers", "Moses Leota"): 3.4,
    ("Panthers", "Mitch Kenny"): 3.2,
    ("Panthers", "Lindsay Smith"): 2.5,
    ("Panthers", "Isaiah Papali'i"): 2.7,
    ("Panthers", "Liam Martin"): 3.0,
    ("Panthers", "Isaah Yeo"): 5.2,
    ("Panthers", "Freddy Lussick"): 1.1,
    ("Panthers", "Scott Sorensen"): 2.0,
    ("Panthers", "Liam Henry"): 1.7,
    ("Panthers", "Billy Phillips"): 1.0,
    ("Panthers", "Blaize Talagi"): 1.8,

    # ========================================================
    # ROOSTERS
    # ========================================================

    ("Roosters", "James Tedesco"): 5.3,
    ("Roosters", "Daniel Tupou"): 1.7,
    ("Roosters", "Billy Smith"): 1.8,
    ("Roosters", "Robert Toia"): 2.3,
    ("Roosters", "Mark Nawaqanitawase"): 2.2,
    ("Roosters", "Hugo Savala"): 2.6,
    ("Roosters", "Daly Cherry-Evans"): 5.7,
    ("Roosters", "Naufahu Whyte"): 2.4,
    ("Roosters", "Reece Robson"): 4.0,
    ("Roosters", "Lindsay Collins"): 3.2,
    ("Roosters", "Salesi Foketi"): 1.5,
    ("Roosters", "Siua Wong"): 2.3,
    ("Roosters", "Victor Radley"): 3.6,
    ("Roosters", "Connor Watson"): 2.3,
    ("Roosters", "Spencer Leniu"): 2.3,
    ("Roosters", "Nat Butcher"): 1.8,
    ("Roosters", "Angus Crichton"): 3.4,
}


# ------------------------------------------------------------
# NAME / POSITION HELPERS
# ------------------------------------------------------------

def clean_text(value):
    if value is None:
        return ""

    return " ".join(
        str(value).strip().split()
    )


def normalise_position(position):
    p = clean_text(position).lower()

    mapping = {
        "fullback": "fullback",
        "wing": "winger",
        "winger": "winger",
        "centre": "centre",
        "center": "centre",
        "five eighth": "five-eighth",
        "five-eighth": "five-eighth",
        "5/8": "five-eighth",
        "half": "halfback",
        "halfback": "halfback",
        "prop": "prop",
        "hooker": "hooker",
        "second row": "2nd row",
        "2nd row": "2nd row",
        "back row": "2nd row",
        "lock": "lock",
        "interchange": "interchange",
        "bench": "interchange",
        "reserve": "reserve",
        "reserves": "reserve",
    }

    return mapping.get(
        p,
        p,
    )


def player_key(team, player):
    return (
        clean_text(team).lower(),
        clean_text(player).lower(),
    )


OVERRIDE_LOOKUP = {
    player_key(team, player): rating
    for (team, player), rating
    in PLAYER_OVERRIDES.items()
}


# ------------------------------------------------------------
# RATING CALCULATION
# ------------------------------------------------------------

def calculate_rating(
    team,
    player,
    position,
):
    """
    Return the player's current model impact rating.

    Manual individual player rating has priority.

    If no manual player rating exists, use position prior.
    """

    key = player_key(
        team,
        player,
    )

    if key in OVERRIDE_LOOKUP:
        return float(
            OVERRIDE_LOOKUP[key]
        )

    pos = normalise_position(
        position
    )

    return float(
        POSITION_BASE.get(
            pos,
            1.0,
        )
    )


# ------------------------------------------------------------
# LOAD EXISTING DATABASE
# ------------------------------------------------------------

def load_existing_ratings():

    columns = [
        "team",
        "player",
        "position",
        "impact_points",
    ]

    if not os.path.exists(
        RATINGS_FILE
    ):
        return pd.DataFrame(
            columns=columns
        )

    try:
        df = pd.read_csv(
            RATINGS_FILE
        )
    except Exception:
        return pd.DataFrame(
            columns=columns
        )

    for col in columns:
        if col not in df.columns:
            df[col] = ""

    df = df[
        columns
    ].copy()

    return df


# ------------------------------------------------------------
# LOAD CURRENT NRL TEAM LIST
# ------------------------------------------------------------

def load_current_team_lists():

    if not os.path.exists(
        TEAM_LIST_FILE
    ):
        raise FileNotFoundError(
            f"{TEAM_LIST_FILE} not found. "
            "Run team_lists.py first."
        )

    df = pd.read_csv(
        TEAM_LIST_FILE
    )

    required = {
        "team",
        "player",
        "position",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "current_team_lists.csv "
            f"is missing columns: {missing}"
        )

    return df


# ------------------------------------------------------------
# UPDATE CURRENT PLAYERS
# ------------------------------------------------------------

def build_ratings():

    current = load_current_team_lists()

    existing = load_existing_ratings()

    records = {}

    # --------------------------------------------------------
    # First preserve historical players already in the file.
    # This is important because a player who is OUT this week
    # still needs a rating.
    # --------------------------------------------------------

    for _, row in existing.iterrows():

        team = clean_text(
            row["team"]
        )

        player = clean_text(
            row["player"]
        )

        position = normalise_position(
            row["position"]
        )

        if not team or not player:
            continue

        try:
            old_rating = float(
                row["impact_points"]
            )
        except Exception:
            old_rating = calculate_rating(
                team,
                player,
                position,
            )

        # If the player now has a manual override,
        # use the latest manual rating.
        key = player_key(
            team,
            player,
        )

        if key in OVERRIDE_LOOKUP:
            old_rating = float(
                OVERRIDE_LOOKUP[key]
            )

        records[key] = {
            "team": team,
            "player": player,
            "position": position,
            "impact_points": round(
                old_rating,
                2,
            ),
        }

    # --------------------------------------------------------
    # Now update all players in the current official team list.
    # --------------------------------------------------------

    updated = 0
    added = 0
    override_count = 0

    for _, row in current.iterrows():

        team = clean_text(
            row["team"]
        )

        player = clean_text(
            row["player"]
        )

        position = normalise_position(
            row["position"]
        )

        key = player_key(
            team,
            player,
        )

        rating = calculate_rating(
            team,
            player,
            position,
        )

        if key in OVERRIDE_LOOKUP:
            override_count += 1

        if key in records:
            updated += 1
        else:
            added += 1

        records[key] = {
            "team": team,
            "player": player,
            "position": position,
            "impact_points": round(
                rating,
                2,
            ),
        }

    output = pd.DataFrame(
        list(records.values())
    )

    output = output.drop_duplicates(
        subset=[
            "team",
            "player",
        ],
        keep="last",
    )

    output = output.sort_values(
        [
            "team",
            "impact_points",
            "player",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    output.to_csv(
        RATINGS_FILE,
        index=False,
    )

    print(
        f"[ratings] {RATINGS_FILE} updated"
    )

    print(
        f"[ratings] Total players: "
        f"{len(output)}"
    )

    print(
        f"[ratings] Current players updated: "
        f"{updated}"
    )

    print(
        f"[ratings] New players added: "
        f"{added}"
    )

    print(
        f"[ratings] Manual quality overrides used: "
        f"{override_count}"
    )

    print("")
    print(
        "[ratings] Highest current player ratings:"
    )

    current_keys = {
        player_key(
            row["team"],
            row["player"],
        )
        for _, row
        in current.iterrows()
    }

    current_rated = output[
        output.apply(
            lambda r: player_key(
                r["team"],
                r["player"],
            )
            in current_keys,
            axis=1,
        )
    ]

    top_players = (
        current_rated
        .sort_values(
            "impact_points",
            ascending=False,
        )
        .head(20)
    )

    for _, row in top_players.iterrows():

        print(
            f"[ratings] "
            f"{row['team']} | "
            f"{row['player']} | "
            f"{row['position']} | "
            f"{row['impact_points']:.1f}"
        )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    print(
        "[ratings] Building NRL "
        "player impact ratings..."
    )

    build_ratings()

    print(
        "[ratings] Player rating build complete."
    )
