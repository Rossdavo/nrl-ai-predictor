#!/usr/bin/env python3
"""Experimental NRL weather backtest focused on predicting winners.

Input: historical_weather.csv
Required:
match_date,home_team,away_team,winner,home_odds,away_odds,home_score,away_score,wet

Optional:
home_experience,away_experience
home_attack_rating,away_attack_rating
home_defence_rating,away_defence_rating

wet = 1 only for genuinely wet match conditions; otherwise 0.
This script does not modify the live predictor.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path("historical_weather.csv")
OUTPUT = Path("weather_backtest_results.csv")

def rate(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s.mean() if len(s) else np.nan

def pc(x):
    return "" if pd.isna(x) else f"{x*100:.1f}%"

def group_summary(d, name):
    return {
        "group": name,
        "games": len(d),
        "favourite_win_rate": rate(d["favourite_won"]),
        "upset_rate": rate(d["upset"]),
        "avg_winning_margin": pd.to_numeric(d["winning_margin"], errors="coerce").mean()
    }

def advantage_test(df, col, label):
    h = pd.to_numeric(df["home_" + col], errors="coerce")
    a = pd.to_numeric(df["away_" + col], errors="coerce")
    valid = h.notna() & a.notna() & (h != a)
    x = df.loc[valid].copy()
    if x.empty:
        return None
    x["adv_team"] = np.where(h.loc[valid] > a.loc[valid], x["home_team"], x["away_team"])
    x["adv_won"] = (
        x["adv_team"].astype(str).str.strip().str.lower()
        == x["winner"].astype(str).str.strip().str.lower()
    ).astype(int)
    dry, wet = x[x["wet"] == 0], x[x["wet"] == 1]
    return {
        "feature": label,
        "all_games": len(x), "all_win_rate": rate(x["adv_won"]),
        "dry_games": len(dry), "dry_win_rate": rate(dry["adv_won"]),
        "wet_games": len(wet), "wet_win_rate": rate(wet["adv_won"])
    }

def main():
    if not INPUT.exists():
        print("ERROR: historical_weather.csv not found.")
        print("Required columns:")
        print("match_date,home_team,away_team,winner,home_odds,away_odds,home_score,away_score,wet")
        sys.exit(1)

    df = pd.read_csv(INPUT)
    required = ["match_date","home_team","away_team","winner","home_odds","away_odds",
                "home_score","away_score","wet"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit("Missing required columns: " + ", ".join(missing))

    for c in ["home_odds","away_odds","home_score","away_score","wet"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["wet"].isin([0,1])].copy()
    if df.empty:
        raise SystemExit("No valid wet/dry matches found.")

    df["favourite_team"] = np.where(
        df["home_odds"] < df["away_odds"], df["home_team"],
        np.where(df["away_odds"] < df["home_odds"], df["away_team"], "")
    )
    has_fav = df["favourite_team"] != ""
    winner = df["winner"].astype(str).str.strip().str.lower()
    fav = df["favourite_team"].astype(str).str.strip().str.lower()
    df["favourite_won"] = np.where(has_fav, (winner == fav).astype(int), np.nan)
    df["upset"] = np.where(has_fav, 1 - df["favourite_won"], np.nan)
    df["winning_margin"] = (df["home_score"] - df["away_score"]).abs()

    summary = pd.DataFrame([
        group_summary(df[df["wet"] == 0], "Dry"),
        group_summary(df[df["wet"] == 1], "Wet")
    ])

    print("\n=== WEATHER BACKTEST ===")
    print(f"Total games: {len(df)}")
    print(f"Dry games: {(df['wet']==0).sum()}")
    print(f"Wet games: {(df['wet']==1).sum()}")

    show = summary.copy()
    show["favourite_win_rate"] = show["favourite_win_rate"].map(pc)
    show["upset_rate"] = show["upset_rate"].map(pc)
    show["avg_winning_margin"] = show["avg_winning_margin"].map(
        lambda x: "" if pd.isna(x) else f"{x:.2f}"
    )
    print("\n=== WINNER RESULTS ===")
    print(show.to_string(index=False))

    tests = []
    for col, label in [
        ("experience", "Higher experience"),
        ("attack_rating", "Higher attack rating"),
        ("defence_rating", "Higher defence rating")
    ]:
        if "home_" + col in df.columns and "away_" + col in df.columns:
            r = advantage_test(df, col, label)
            if r:
                tests.append(r)

    if tests:
        t = pd.DataFrame(tests)
        s = t.copy()
        for c in ["all_win_rate","dry_win_rate","wet_win_rate"]:
            s[c] = s[c].map(pc)
        print("\n=== WET-WEATHER INTERACTION TESTS ===")
        print(s.to_string(index=False))
    else:
        t = pd.DataFrame()
        print("\nNo experience/attack/defence columns supplied yet.")
        print("Basic wet vs dry winner test completed.")

    rows = []
    for _, r in summary.iterrows():
        rows.append({
            "test":"weather","group":r["group"],"games":r["games"],
            "win_rate":r["favourite_win_rate"],"upset_rate":r["upset_rate"],
            "average_margin":r["avg_winning_margin"]
        })
    if not t.empty:
        for _, r in t.iterrows():
            for g in ["dry","wet"]:
                rows.append({
                    "test":r["feature"],"group":g.title(),"games":r[g+"_games"],
                    "win_rate":r[g+"_win_rate"],"upset_rate":np.nan,"average_margin":np.nan
                })

    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT}")
    print("\nKISS: weather only earns a place in predict.py if it improves winner prediction.")

if __name__ == "__main__":
    main()
