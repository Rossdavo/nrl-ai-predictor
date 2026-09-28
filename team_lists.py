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
    global FINALS_TEAMS
    FINALS_TEAMS = load_current_teams_from_odds()

    print(
        "[teams] Fetching official current NRL team lists "
        f"for {len(FINALS_TEAMS)} teams..."
    )

    team_lists = (
        get_best_team_list()
    )

    # This validation happens AFTER fresh data
    # and previous-file fallbacks have been combined.
    #
    # Files are still never overwritten unless every
    # team has a complete active jersey set from 1-17.
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
