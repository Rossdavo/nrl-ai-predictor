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


def build_profit_section():
    if not os.path.exists("bet_log.csv"):
        return ""

    bets = pd.read_csv("bet_log.csv")
    if bets.empty:
        return ""

    settled = bets[bets.get("settled", "") == "Y"]
    profit = settled.get("profit_units", pd.Series(dtype=float)).fillna(0).sum()
    n = len(settled)

    roi = profit / n if n else 0

    html = "<h2>Profit Tracker</h2>"
    html += f"<p class='note'><b>Bets settled:</b> {n} | <b>Total profit:</b> {profit:.2f}u | <b>ROI:</b> {roi:.2%}</p>"
    html += bets.tail(10).to_html(index=False)
    return html


def main():
    df = pd.read_csv("predictions.csv")

    cols = [
        "date","kickoff_local","home","away",
        "home_win_prob","exp_margin_home","exp_total","confidence",
        "home_odds","away_odds","value_flag",
        "home_top_try","away_top_try"
    ]

    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    if "value_flag" in df.columns:
        df["value_flag"] = df["value_flag"].fillna("")

    table_html = df.to_html(index=False, escape=False)

    sections = build_profit_section()

    html = HTML_TEMPLATE.format(table=table_html, sections=sections)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
