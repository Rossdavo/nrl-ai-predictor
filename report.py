import os
import math
import pandas as pd
from pandas.errors import EmptyDataError


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>NRL AI Predictions</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; }}
    h1 {{ margin-bottom: 6px; }}
    h2 {{ margin-top: 28px; }}
    .note {{ color: #444; margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 18px; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; vertical-align: top; }}
    th {{ background: #f6f6f6; text-align: left; }}
    .small {{ font-size: 12px; color: #666; }}
    .value {{ font-weight: bold; }}
  </style>
</head>
<body>
  <h1>NRL AI Predictions</h1>
  <p class="note">Automated predictions with model probabilities, odds comparison, value detection, and staking suggestions.</p>

  {table}

  {accuracy}

  {clv_roi}

  <p class="small">Generated automatically via GitHub Actions.</p>
</body>
</html>
"""


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if df is None:
            return pd.DataFrame()
        return df
    except EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def implied_prob(decimal_odds) -> float:
    try:
        if decimal_odds is None:
            return float("nan")
        o = float(decimal_odds)
        if o <= 1.0:
            return float("nan")
        return 1.0 / o
    except Exception:
        return float("nan")


def pct(x) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
        return f"{float(x):.1%}"
    except Exception:
        return ""


def money(x) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
        return f"${float(x):.2f}"
    except Exception:
        return ""


def build_accuracy_section() -> str:
    acc = safe_read_csv("accuracy.csv")
    if acc.empty:
        return "<p class='note'><b>Results & Accuracy:</b> No completed matches scored yet.</p>"

    scored = len(acc)
    win_acc = acc["winner_correct"].mean() if "winner_correct" in acc.columns else float("nan")
    brier = acc["brier"].mean() if "brier" in acc.columns else float("nan")

    mae = float("nan")
    if "abs_margin_error" in acc.columns:
        s = pd.to_numeric(acc["abs_margin_error"], errors="coerce").dropna()
        if len(s):
            mae = float(s.mean())

    headline = f"<h2>Results &amp; Accuracy</h2><p class='note'><b>Scored games:</b> {scored}"
    if win_acc == win_acc:
        headline += f" | Winner accuracy: {win_acc:.0%}"
    if brier == brier:
        headline += f" | Brier: {brier:.3f}"
    if mae == mae:
        headline += f" | Margin MAE: {mae:.2f}"
    headline += "</p>"

    show = acc.copy()
    for c in ["date", "home", "away"]:
        if c not in show.columns:
            show[c] = ""

    show = show.sort_values(["date", "home"], ascending=[True, True]).tail(15)
    return headline + show.to_html(index=False, escape=False)


def build_clv_roi_section() -> str:
    perf = safe_read_csv("performance.csv")
    if perf.empty:
        return "<p class='note'><b>CLV & ROI:</b> Nothing to report yet.</p>"

    return "<h2>CLV &amp; ROI</h2>" + perf.to_html(index=False, escape=False)


def main():
    df = safe_read_csv("predictions.csv")
    if df.empty:
        html = HTML_TEMPLATE.format(
            table="<p class='note'>No predictions.csv found yet.</p>",
            accuracy=build_accuracy_section(),
            clv_roi=build_clv_roi_section(),
        )
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(html)
        return

    # Merge odds.csv if it exists (some runs write odds into predictions.csv already)
    odds = safe_read_csv("odds.csv")
    if not odds.empty:
        for col in ["date", "home", "away"]:
            if col not in odds.columns:
                odds[col] = ""
        df = df.merge(odds, on=["date", "home", "away"], how="left", suffixes=("", "_oddsfile"))

        # Prefer odds from odds.csv if present
        if "home_odds_oddsfile" in df.columns:
            df["home_odds"] = df["home_odds_oddsfile"].combine_first(df.get("home_odds"))
        if "away_odds_oddsfile" in df.columns:
            df["away_odds"] = df["away_odds_oddsfile"].combine_first(df.get("away_odds"))

        drop_cols = [c for c in df.columns if c.endswith("_oddsfile")]
        if drop_cols:
            df = df.drop(columns=drop_cols)

    # Merge bet_log.csv for stake suggestions (only one recommended side per match)
    bets = safe_read_csv("bet_log.csv")
    if not bets.empty:
        for col in ["date", "home", "away"]:
            if col not in bets.columns:
                bets[col] = ""
        keep = [c for c in ["date", "home", "away", "bet_side", "prob", "odds", "kelly_fraction", "stake"] if c in bets.columns]
        bets = bets[keep]
        df = df.merge(bets, on=["date", "home", "away"], how="left", suffixes=("", "_bet"))

    # Compute implied probs + edge (both sides)
    if "home_odds" in df.columns:
        df["home_implied_prob"] = df["home_odds"].apply(implied_prob)
    else:
        df["home_implied_prob"] = float("nan")

    if "away_odds" in df.columns:
        df["away_implied_prob"] = df["away_odds"].apply(implied_prob)
    else:
        df["away_implied_prob"] = float("nan")

    # Model probabilities
    if "home_win_prob" not in df.columns:
        df["home_win_prob"] = 0.5

    df["away_win_prob"] = 1.0 - pd.to_numeric(df["home_win_prob"], errors="coerce").fillna(0.5)

    df["home_edge"] = pd.to_numeric(df["home_win_prob"], errors="coerce") - pd.to_numeric(df["home_implied_prob"], errors="coerce")
    df["away_edge"] = pd.to_numeric(df["away_win_prob"], errors="coerce") - pd.to_numeric(df["away_implied_prob"], errors="coerce")

    # Pick display edge based on recommended bet_side (if any)
    def pick_edge(row):
        side = row.get("bet_side", "")
        if side == "HOME":
            return row.get("home_edge", float("nan"))
        if side == "AWAY":
            return row.get("away_edge", float("nan"))
        return float("nan")

    df["edge_pct"] = df.apply(pick_edge, axis=1)
    df["stake_display"] = df.get("stake", float("nan"))

    # Pretty formatting columns
    df["home_win_prob"] = df["home_win_prob"].apply(lambda x: round(float(x), 3) if str(x) != "nan" else x)
    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].apply(lambda x: round(float(x), 2) if str(x) != "nan" else x)

    df["edge_pct"] = df["edge_pct"].apply(lambda x: pct(x))
    df["stake_display"] = df["stake_display"].apply(lambda x: money(x))

    # Build display table columns
    cols = [
        "date",
        "kickoff_local",
        "home",
        "away",
        "home_win_prob",
        "confidence",
        "home_odds",
        "away_odds",
        "value_flag",
        "bet_side",
        "edge_pct",
        "stake_display",
        "home_top_try",
        "away_top_try",
    ]
    cols = [c for c in cols if c in df.columns]

    # Clean NaNs for nicer display
    for c in ["value_flag", "bet_side"]:
        if c in df.columns:
            df[c] = df[c].fillna("")

    # Rename headers for readability
    rename = {
        "home_win_prob": "model_home_prob",
        "bet_side": "recommended_bet",
        "edge_pct": "edge",
        "stake_display": "stake",
    }
    show = df[cols].rename(columns=rename)

    table_html = show.to_html(index=False, escape=False)

    html = HTML_TEMPLATE.format(
        table=table_html,
        accuracy=build_accuracy_section(),
        clv_roi=build_clv_roi_section(),
    )

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
