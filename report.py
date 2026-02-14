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
    .note {{ color: #444; margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 18px; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; vertical-align: top; }}
    th {{ background: #f6f6f6; text-align: left; }}
    .small {{ font-size: 12px; color: #666; }}
  </style>
</head>
<body>
  <h1>NRL AI Predictions</h1>
  <p class="note">Automated predictions with model probabilities, odds comparison, and value detection.</p>
  {table}
  {sections}
  <p class="small">Generated automatically via GitHub Actions.</p>
</body>
</html>
"""


def build_performance_section() -> str:
    if not os.path.exists("performance.csv"):
        return ""

    try:
        perf = pd.read_csv("performance.csv")
    except Exception:
        return ""

    if perf.empty:
        return ""

    return "<h2>CLV & ROI</h2>" + perf.to_html(index=False)


def build_accuracy_section() -> str:
    if not os.path.exists("accuracy.csv"):
        return "<p class='note'><b>Accuracy:</b> No completed matches scored yet.</p>"

    try:
        acc = pd.read_csv("accuracy.csv")
    except Exception:
        return "<p class='note'><b>Accuracy:</b> Unable to read accuracy.csv</p>"

    if acc.empty:
        return "<p class='note'><b>Accuracy:</b> No completed matches scored yet.</p>"

    scored = len(acc)
    win_acc = acc["winner_correct"].mean() if "winner_correct" in acc.columns else 0.0
    brier = acc["brier"].mean() if "brier" in acc.columns else float("nan")

    mae = float("nan")
    if "abs_margin_error" in acc.columns:
        s = pd.to_numeric(acc["abs_margin_error"], errors="coerce").dropna()
        if len(s):
            mae = s.mean()

    headline = (
        f"<h2>Results & Accuracy</h2>"
        f"<p class='note'><b>Scored games:</b> {scored} | "
        f"Winner accuracy: {win_acc:.0%} | "
        f"Brier: {brier:.3f}"
    )
    if mae == mae:
        headline += f" | Margin MAE: {mae:.2f}"
    headline += "</p>"

    # Show last 10 scored matches (if columns exist)
    sort_cols = [c for c in ["date", "home"] if c in acc.columns]
    if sort_cols:
        show = acc.sort_values(sort_cols).tail(10)
    else:
        show = acc.tail(10)

    table = show.to_html(index=False)
    return headline + table
  def build_profit_section() -> str:
    if not os.path.exists("bet_log.csv"):
        return ""

    try:
        bets = pd.read_csv("bet_log.csv")
    except Exception:
        return ""

    if bets.empty:
        return ""

    # only settled bets count for ROI
    if "settled" in bets.columns:
        settled = bets[bets["settled"].astype(str).str.upper() == "Y"].copy()
    else:
        settled = pd.DataFrame()

    total_logged = len(bets)
    total_settled = len(settled)

    profit = 0.0
    roi = float("nan")

    if total_settled > 0 and "profit_units" in settled.columns:
        settled["profit_units"] = pd.to_numeric(settled["profit_units"], errors="coerce")
        profit = float(settled["profit_units"].fillna(0).sum())
        roi = profit / float(total_settled)

    html = "<h2>Profit Tracker (1 unit stakes)</h2>"
    html += f"<p class='note'><b>Bets logged:</b> {total_logged} | <b>Bets settled:</b> {total_settled}"

    if total_settled > 0:
        html += f" | <b>Total profit:</b> {profit:.2f}u | <b>ROI per bet:</b> {roi:.2%}"
    else:
        html += " | <b>Total profit:</b> 0.00u | <b>ROI per bet:</b> N/A"

    html += "</p>"

    # show last 10 bets (settled or not)
    show = bets.tail(10)
    html += show.to_html(index=False)

    return html


def main():
    df = pd.read_csv("predictions.csv")

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
    df = df[cols]

    if "value_flag" in df.columns:
        df["value_flag"] = df["value_flag"].fillna("")

    table_html = df.to_html(index=False, escape=False)

    sections = build_profit_section() + build_performance_section() + build_accuracy_section()

    html = HTML_TEMPLATE.format(table=table_html, sections=sections)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
