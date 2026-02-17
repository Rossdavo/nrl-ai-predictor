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
    .pill {{ display:inline-block; padding:2px 8px; border:1px solid #ddd; border-radius:999px; font-size:12px; margin-right:6px; }}
  </style>
</head>
<body>
  <h1>NRL AI Predictions</h1>
  <p class="note">Automated predictions with model probabilities, odds comparison, and value detection.</p>
  {table}
  {bankroll}
  {accuracy}
  {clv_roi}
  <p class="small">Generated automatically via GitHub Actions.</p>
</body>
</html>
"""

def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    if os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def build_bankroll_section() -> str:
    df = _safe_read_csv("bankroll_status.csv")
    if df.empty:
        return "<h2>Bankroll</h2><p class='note'>No bankroll tracking yet.</p>"
    r = df.iloc[-1].to_dict()
    return (
        "<h2>Bankroll</h2>"
        f"<p class='note'>"
        f"<span class='pill'><b>Start</b> ${r.get('start_bankroll', '')}</span>"
        f"<span class='pill'><b>Current</b> ${r.get('current_bankroll', '')}</span>"
        f"<span class='pill'><b>Peak</b> ${r.get('peak_bankroll', '')}</span>"
        f"<span class='pill'><b>Drawdown</b> {r.get('drawdown_pct', '')}%</span>"
        f"<span class='pill'><b>ROI</b> {r.get('roi_pct', '')}%</span>"
        f"<span class='pill'><b>Settled bets</b> {r.get('settled_bets', '')}</span>"
        f"</p>"
    )

def build_accuracy_section() -> str:
    acc = _safe_read_csv("accuracy.csv")
    if acc.empty:
        return "<h2>Results & Accuracy</h2><p class='note'><b>Accuracy:</b> No completed matches scored yet.</p>"

    scored = len(acc)
    win_acc = acc["winner_correct"].mean() if "winner_correct" in acc.columns else 0.0
    brier = acc["brier"].mean() if "brier" in acc.columns else float("nan")

    mae = float("nan")
    if "abs_margin_error" in acc.columns:
        s = pd.to_numeric(acc["abs_margin_error"], errors="coerce").dropna()
        if len(s):
            mae = float(s.mean())

    headline = f"<h2>Results & Accuracy</h2><p class='note'><b>Scored games:</b> {scored} | Winner accuracy: {win_acc:.0%} | Brier: {brier:.3f}"
    if mae == mae:
        headline += f" | Margin MAE: {mae:.2f}"
    headline += "</p>"

    show = acc.sort_values(["date", "home"]).tail(10)
    return headline + show.to_html(index=False)

def build_clv_roi_section() -> str:
    perf = _safe_read_csv("performance.csv")
    if perf.empty:
        return "<h2>CLV & ROI</h2><p class='note'>No settled bets yet.</p>"

    # If you later add clv columns, they will show automatically.
    show = perf.tail(10)
    return "<h2>CLV & ROI</h2>" + show.to_html(index=False)

def main():
    preds = _safe_read_csv("predictions.csv")
    if preds.empty:
        table_html = "<p class='note'>No predictions yet.</p>"
    else:
        cols = [
            "date","kickoff_local","home","away",
            "home_win_prob","exp_margin_home","exp_total","confidence",
            "home_odds","away_odds","value_flag",
            "home_top_try","away_top_try",
        ]
        cols = [c for c in cols if c in preds.columns]
        preds = preds[cols].copy()
        if "value_flag" in preds.columns:
            preds["value_flag"] = preds["value_flag"].fillna("")
        table_html = preds.to_html(index=False, escape=False)

    html = HTML_TEMPLATE.format(
        table=table_html,
        bankroll=build_bankroll_section(),
        accuracy=build_accuracy_section(),
        clv_roi=build_clv_roi_section(),
    )

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()
