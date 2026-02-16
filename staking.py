import pandas as pd
import os
import math

BANKROLL = float(os.getenv("BANKROLL", "1000"))   # default bankroll
KELLY_FRACTION = 0.5   # half-Kelly (safer)


def kelly_fraction(prob, odds):
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return 0.0
    if odds <= 1:
        return 0.0
    b = odds - 1.0
    edge = (prob * odds) - 1.0
    k = edge / b
    return max(0.0, k)


def pick_col(df, base):
    """
    If merge created base_x/base_y, prefer _y (fresh odds file), else base, else _x.
    """
    for c in (f"{base}_y", base, f"{base}_x"):
        if c in df.columns:
            return c
    return None


def main():
    if not os.path.exists("predictions.csv"):
        print("Missing predictions.csv")
        return

    preds = pd.read_csv("predictions.csv")

    # If odds.csv exists, merge it in (fresh odds). If not, rely on predictions.csv.
    if os.path.exists("odds.csv"):
        odds = pd.read_csv("odds.csv")
        df = preds.merge(odds, on=["date", "home", "away"], how="left", suffixes=("_x", "_y"))
    else:
        df = preds.copy()

    home_odds_col = pick_col(df, "home_odds")
    away_odds_col = pick_col(df, "away_odds")

    if home_odds_col is None or away_odds_col is None:
        print("No odds columns found. Writing empty bet_log.csv.")
        pd.DataFrame(columns=[
            "date", "home", "away", "bet_side", "prob", "odds", "kelly_fraction", "stake"
        ]).to_csv("bet_log.csv", index=False)
        return

    bets = []

    for _, r in df.iterrows():
        prob_home = float(r.get("home_win_prob", 0.5))

        home_odds = r.get(home_odds_col)
        away_odds = r.get(away_odds_col)

        # Kelly on both sides (only if value)
        k_home = kelly_fraction(prob_home, home_odds)
        k_away = kelly_fraction(1.0 - prob_home, away_odds)

        # Choose the bigger Kelly edge (if either is > 0)
        if k_home <= 0 and k_away <= 0:
            continue

        if k_home >= k_away:
            side = "HOME"
            prob = prob_home
            odds_val = float(home_odds) if home_odds == home_odds else float("nan")
            k = k_home
        else:
            side = "AWAY"
            prob = 1.0 - prob_home
            odds_val = float(away_odds) if away_odds == away_odds else float("nan")
            k = k_away

        stake = BANKROLL * k * KELLY_FRACTION

        bets.append({
            "date": r.get("date"),
            "home": r.get("home"),
            "away": r.get("away"),
            "bet_side": side,
            "prob": round(prob, 4),
            "odds": odds_val,
            "kelly_fraction": round(k, 4),
            "stake": round(stake, 2),
        })

    out = pd.DataFrame(bets)

    if out.empty:
        # Always write the file so downstream steps never crash
        out = pd.DataFrame(columns=[
            "date", "home", "away", "bet_side", "prob", "odds", "kelly_fraction", "stake"
        ])

    out.to_csv("bet_log.csv", index=False)
    print(f"{len(out)} bets generated")


if __name__ == "__main__":
    main()
