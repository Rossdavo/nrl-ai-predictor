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
    h2 {{ margin-top: 28px; }}
  </style>
</head>
<body>
  <h1>NRL AI Predictions</h1>
  <p class="note">Automated predictions with model probabilities, odds comparison, value detection, staking, and tracking.</p>

  {table}

  {bets}

  {clv}

  {accuracy}

  {performance}

  <p class="small">Generated automatically via GitHub Actions.</p>
</body>
</html>
"""

def _read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def build_bets_section() -> str:
    df = _read_csv_safe("bet_log.csv")
    if df.empty:
        return "<h2>Bets</h2><p class='note'>No bets generated yet (likely missing odds or no value edge).</p>"
    show = df.tail(20)
    return "<h2>Bets</h2>" + show.to_html(index=False)

def build_clv_section() -> str:
    df = _read_csv_safe("clv_results.csv")
    if df.empty:
        return "<h2>CLV</h2><p class='note'>No CLV yet (needs closing_odds.csv and bets).</p>"
    # show latest rows
    show = df.tail(20)
    return "<h2>CLV</h2>" + show.to_html(index=False)

def build_accuracy_section() -> str:
    acc = _read_csv_safe("accuracy.csv")
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

    headline = f"<h2>Results & Accuracy</h2><p class='note'><b>Scored games:</b> {scored} | Winner accuracy: {win_acc:.0%} | Brier: {brier:.3f}"
    if mae == mae:
        headline += f" | Margin MAE: {mae:.2f}"
    headline += "</p>"

    show = acc.sort_values(["date", "home"]).tail(10)
    return headline + show.to_html(index=False)

def build_performance_section() -> str:
    perf = _read_csv_safe("performance.csv")
    if perf.empty:
        return "<h2>Performance</h2><p class='note'>No performance data yet.</p>"
    show = perf.tail(20)
    return "<h2>Performance</h2>" + show.to_html(index=False)

def main():
    preds = _read_csv_safe("predictions.csv")
    if preds.empty:
        raise SystemExit("predictions.csv missing/empty")

    cols = [
        "date","kickoff_local","home","away",
        "home_win_prob","exp_margin_home","exp_total","confidence",
        "home_odds","away_odds","value_flag",
        "home_top_try","away_top_try"
    ]
    cols = [c for c in cols if c in preds.columns]
    preds = preds[cols]

    if "value_flag" in preds.columns:
        preds["value_flag"] = preds["value_flag"].fillna("")

    table_html = preds.to_html(index=False, escape=False)

    html = HTML_TEMPLATE.format(
        table=table_html,
        bets=build_bets_section(),
        clv=build_clv_section(),
        accuracy=build_accuracy_section(),
        performance=build_performance_section(),
    )

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()
