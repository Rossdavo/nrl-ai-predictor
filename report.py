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
  <p class="note">Automated predictions with model probabilities, odds comparison, value detection, and tracking.</p>
  {table}
  {sections}
  <p class="small">Generated automatically via GitHub Actions.</p>
</body>
</html>
"""


def build_profit_section() -> str:
    if not os.path.exists("bet_log.csv"):
        return "<h2>Profit Tracker</h2><p class='note'>No bet_log.csv yet (no value bets logged).</p>"

    bets = pd.read_csv("bet_log.csv")
    if bets.empty:
        return "<h2>Profit Tracker</h2><p class='note'>No bets logged yet.</p>"

    settled = bets[bets.get("settled", "").astype(str).str.upper() == "Y"].copy()
    n_logged = len(bets)
    n_settled = len(settled)

    profit = 0.0
    roi = ""

    if n_settled and "profit_units" in settled.columns:
        settled["profit_units"] = pd.to_numeric(settled["profit_units"], errors="coerce").fillna(0.0)
        profit = float(settled["profit_units"].sum())
        roi = f"{(profit / n_settled):.2%}"

    html = "<h2>Profit Tracker (1 unit stakes)</h2>"
    html += f"<p class='note'><b>Logged:</b> {n_logged} | <b>Settled:</b> {n_settled} | <b>Profit:</b> {profit:.2f}u"
    html += f" | <b>ROI:</b> {roi if roi else 'N/A'}</p>"
    html += bets.tail(10).to_html(index=False)
    return html


def build_clv_roi_section() -> str:
    if not os.path.exists("performance.csv"):
        return "<h2>CLV & ROI</h2><p class='note'>No performance.csv yet.</p>"

    # Handle empty file safely
    try:
        if os.path.getsize("performance.csv") == 0:
            return "<h2>CLV & ROI</h2><p class='note'>No performance data yet.</p>"
    except Exception:
        pass

    try:
        perf = pd.read_csv("performance.csv")
    except Exception:
        return "<h2>CLV & ROI</h2><p class='note'>No performance data yet.</p>"

    if perf.empty:
        return "<h2>CLV & ROI</h2><p class='note'>No performance data yet.</p>"

    return "<h2>CLV & ROI</h2>" + perf.to_html(index=False)


def build_accuracy_section() -> str:
    if not os.path.exists("accuracy.csv"):
        return "<h2>Results & Accuracy</h2><p class='note'>No completed matches scored yet.</p>"

    acc = pd.read_csv("accuracy.csv")
    if acc.empty:
        return "<h2>Results & Accuracy</h2><p class='note'>No completed matches scored yet.</p>"

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
        f"<p class='note'><b>Scored games:</b> {scored} | Winner accuracy: {win_acc:.0%} | Brier: {brier:.3f}"
    )
    if mae == mae:
        headline += f" | Margin MAE: {mae:.2f}"
    headline += "</p>"

    sort_cols = [c for c in ["date", "home"] if c in acc.columns]
    show = acc.sort_values(sort_cols).tail(10) if sort_cols else acc.tail(10)

    return headline + show.to_html(index=False)


def main():
    df = pd.read_csv("predictions.csv")

    cols = [
        "date", "kickoff_local", "home", "away",
        "home_win_prob", "exp_margin_home", "exp_total", "confidence",
        "home_odds", "away_odds", "value_flag",
        "home_top_try", "away_top_try",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    if "value_flag" in df.columns:
        df["value_flag"] = df["value_flag"].fillna("")

    table_html = df.to_html(index=False, escape=False)

    sections = (
        build_profit_section() +
        build_clv_roi_section() +
        build_accuracy_section()
    )

    html = HTML_TEMPLATE.format(table=table_html, sections=sections)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
