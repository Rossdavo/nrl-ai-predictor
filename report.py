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
    :root {{
      --bg: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --border: #e5e7eb;
      --header: #f9fafb;
      --pos: #ecfdf5;
      --neg: #fef2f2;
      --card: #ffffff;
      --shadow: 0 6px 20px rgba(0,0,0,.06);
    }}

    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 28px; background: var(--bg); color: var(--text); }}
    h1 {{ margin: 0 0 6px 0; font-size: 28px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    .note {{ color: var(--muted); margin-top: 6px; }}
    .small {{ font-size: 12px; color: var(--muted); margin-top: 18px; }}

    .wrap {{ max-width: 1100px; margin: 0 auto; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 16px; box-shadow: var(--shadow); }}

    .downloads ul {{ display: flex; flex-wrap: wrap; gap: 10px; padding-left: 18px; }}
    .downloads a {{ text-decoration: none; border: 1px solid var(--border); padding: 8px 10px; border-radius: 10px; color: var(--text); background: var(--header); }}
    .downloads a:hover {{ filter: brightness(0.98); }}

    table {{ border-collapse: separate; border-spacing: 0; width: 100%; margin-top: 14px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--border); padding: 10px 12px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: var(--header); text-align: left; font-weight: 650; z-index: 2; border-top: 1px solid var(--border); }}
    tr:nth-child(even) td {{ background: #fcfcfd; }}

    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; border: 1px solid var(--border); background: var(--header); }}
    .badge-home {{ background: #eef2ff; }}
    .badge-away {{ background: #fff7ed; }}

    .row-pos td {{ background: var(--pos) !important; }}
    .row-neg td {{ background: var(--neg) !important; }}
    .muted {{ color: var(--muted); }}

    @media (max-width: 900px) {{
      body {{ margin: 14px; }}
      table {{ font-size: 13px; }}
      th, td {{ padding: 8px 10px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>NRL AI Predictions</h1>
      <p class="note">Automated predictions with model probabilities, odds comparison, value detection, and staking suggestions.</p>
      <div class="downloads">
        {downloads}
      </div>
    </div>

    <div class="card" style="margin-top:16px;">
      {table}
    </div>

    <div class="card" style="margin-top:16px;">
      {accuracy}
    </div>

    <div class="card" style="margin-top:16px;">
      {clv_roi}
    </div>

    <p class="small">Generated automatically via GitHub Actions.</p>
  </div>
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
    try:
        if x is None:
            return ""
        if isinstance(x, float) and math.isnan(x):
            return ""
        return f"{float(x):.1%}"
    except Exception:
        return ""


def money(x) -> str:
    try:
        if x is None:
            return ""
        if isinstance(x, float) and math.isnan(x):
            return ""
        return f"${float(x):.2f}"
    except Exception:
        return ""


def build_downloads_section() -> str:
    links = [
        ("predictions.csv", "Predictions CSV"),
        ("odds.csv", "Odds CSV"),
        ("bet_log.csv", "Bet Log CSV"),
        ("accuracy.csv", "Accuracy CSV"),
        ("performance.csv", "Performance CSV"),
        ("closing_odds.csv", "Closing Odds CSV"),
        ("bankroll_status.csv", "Bankroll Status CSV"),
        ("predictions_history.csv", "Predictions History CSV"),
        ("odds_history.csv", "Odds History CSV"),
        ("results_cache.csv", "Results Cache CSV"),
        ("ratings.json", "Ratings JSON"),
    ]

    items = []
    for filename, label in links:
        items.append(
            f"<li><a href='{filename}' target='_blank' rel='noopener'>{label}</a></li>"
        )

    return "<h2>Downloads</h2><ul>" + "".join(items) + "</ul>"


def build_accuracy_section() -> str:
    acc = safe_read_csv("accuracy.csv")
    if acc.empty:
        return "<p class='note'><b>Results &amp; Accuracy:</b> No completed matches scored yet.</p>"

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
        return "<p class='note'><b>CLV &amp; ROI:</b> Nothing to report yet.</p>"

    return "<h2>CLV &amp; ROI</h2>" + perf.to_html(index=False, escape=False)


def main():
    df = safe_read_csv("predictions.csv")
    if df.empty:
        html = HTML_TEMPLATE.format(
            downloads=build_downloads_section(),
            table="<p class='note'>No predictions.csv found yet.</p>",
            accuracy=build_accuracy_section(),
            clv_roi=build_clv_roi_section(),
        )
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(html)
        return

    # Merge odds.csv if it exists
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

    # Merge bet_log.csv for stake suggestions
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

    if "home_win_prob" not in df.columns:
        df["home_win_prob"] = 0.5

    df["away_win_prob"] = 1.0 - pd.to_numeric(df["home_win_prob"], errors="coerce").fillna(0.5)

    df["home_edge"] = pd.to_numeric(df["home_win_prob"], errors="coerce") - pd.to_numeric(df["home_implied_prob"], errors="coerce")
    df["away_edge"] = pd.to_numeric(df["away_win_prob"], errors="coerce") - pd.to_numeric(df["away_implied_prob"], errors="coerce")

    def pick_edge(row):
        side = row.get("bet_side", "")
        if side == "HOME":
            return row.get("home_edge", float("nan"))
        if side == "AWAY":
            return row.get("away_edge", float("nan"))
        return float("nan")

    df["edge_pct"] = df.apply(pick_edge, axis=1)
    df["stake_display"] = df.get("stake", float("nan"))

    df["edge_pct"] = df["edge_pct"].apply(pct)
    df["stake_display"] = df["stake_display"].apply(money)

    # Display columns
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

    for c in ["value_flag", "bet_side"]:
        if c in df.columns:
            df[c] = df[c].fillna("")

    rename = {
        "home_win_prob": "model_home_prob",
        "bet_side": "recommended_bet",
        "edge_pct": "edge",
        "stake_display": "stake",
    }
    show = df[cols].rename(columns=rename)
    table_html = show.to_html(index=False, escape=False)

    html = HTML_TEMPLATE.format(
        downloads=build_downloads_section(),
        table=table_html,
        accuracy=build_accuracy_section(),
        clv_roi=build_clv_roi_section(),
    )

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
