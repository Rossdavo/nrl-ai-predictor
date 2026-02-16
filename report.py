import os
import pandas as pd

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>NRL AI Predictions</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; }}
    h1 {{ margin-bottom: 6px; }}
    h2 {{ margin-top: 26px; }}
    .note {{ color: #444; margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; vertical-align: top; }}
    th {{ background: #f6f6f6; text-align: left; }}
    .small {{ font-size: 12px; color: #666; }}
    a {{ color: #0b66c3; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>NRL AI Predictions</h1>
  <p class="note">Automated predictions with model probabilities, odds comparison, and value detection.</p>

  <p class="note">
    <b>Downloads:</b>
    <a href="predictions.csv">predictions.csv</a> ·
    <a href="odds.csv">odds.csv</a> ·
    <a href="bet_log.csv">bet_log.csv</a> ·
    <a href="accuracy.csv">accuracy.csv</a> ·
    <a href="performance.csv">performance.csv</a> ·
    <a href="closing_odds.csv">closing_odds.csv</a>
  </p>

  {table}

  {bets}

  {accuracy}

  {clv_roi}

  <p class="small">Generated automatically via GitHub Actions.</p>
</body>
</html>
"""

def _safe_read_csv(path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        # Empty file can throw EmptyDataError
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def build_bets_section() -> str:
    bets = _safe_read_csv("bet_log.csv")
    if bets.empty:
        return "<h2>Latest Bets</h2><p class='note'>No bets generated this run (no positive Kelly edge / no odds available).</p>"

    # Keep a sensible set of columns if present
    preferred = ["date", "home", "away", "bet_side", "prob", "odds", "kelly_fraction", "stake"]
    cols = [c for c in preferred if c in bets.columns]
    if cols:
        bets = bets[cols]

    # Sort: biggest stakes first, then date
    if "stake" in bets.columns:
        bets["stake"] = pd.to_numeric(bets["stake"], errors="coerce")
        bets = bets.sort_values(["stake"], ascending=False)
    elif "date" in bets.columns:
        bets = bets.sort_values(["date"], ascending=False)

    show = bets.head(10)
    return "<h2>Latest Bets</h2>" + show.to_html(index=False)

def build_accuracy_section() -> str:
    acc = _safe_read_csv("accuracy.csv")
    if acc.empty:
        return "<h2>Results & Accuracy</h2><p class='note'>No completed matches scored yet.</p>"

    scored = len(acc)

    win_acc = 0.0
    if "winner_correct" in acc.columns:
        win_acc = pd.to_numeric(acc["winner_correct"], errors="coerce").mean()

    brier = float("nan")
    if "brier" in acc.columns:
        brier = pd.to_numeric(acc["brier"], errors="coerce").mean()

    mae = float("nan")
    if "abs_margin_error" in acc.columns:
        mae = pd.to_numeric(acc["abs_margin_error"], errors="coerce").mean()

    headline = f"<h2>Results & Accuracy</h2><p class='note'><b>Scored games:</b> {scored} | Winner accuracy: {win_acc:.0%}"
    if brier == brier:
        headline += f" | Brier: {brier:.3f}"
    if mae == mae:
        headline += f" | Margin MAE: {mae:.2f}"
    headline += "</p>"

    # Show last 10 scored games if columns exist
    sort_cols = [c for c in ["date", "home"] if c in acc.columns]
    if sort_cols:
        show = acc.sort_values(sort_cols).tail(10)
    else:
        show = acc.tail(10)

    return headline + show.to_html(index=False)

def build_clv_roi_section() -> str:
    perf = _safe_read_csv("performance.csv")
    if perf.empty:
        return "<h2>CLV & ROI</h2><p class='note'>No performance data yet (need bets + closing odds + settled results).</p>"
    return "<h2>CLV & ROI</h2>" + perf.to_html(index=False)

def main():
    df = _safe_read_csv("predictions.csv")
    if df.empty:
        table_html = "<p class='note'>No predictions.csv found yet.</p>"
    else:
        cols = [
            "date",
            "kickoff_local",
            "home",
            "away",
            "home_win_prob",
            "exp_margin_home",
            "exp_total",
            "confidence",
            "home_odds",
            "away_odds",
            "value_flag",
            "home_top_try",
            "away_top_try",
        ]
        cols = [c for c in cols if c in df.columns]
        if cols:
            df = df[cols]

        if "value_flag" in df.columns:
            df["value_flag"] = df["value_flag"].fillna("")

        table_html = df.to_html(index=False, escape=False)

    html = HTML_TEMPLATE.format(
        table=table_html,
        bets=build_bets_section(),
        accuracy=build_accuracy_section(),
        clv_roi=build_clv_roi_section(),
    )

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()
