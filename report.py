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
    .pill {{ display:inline-block; padding:2px 8px; border:1px solid #ddd; border-radius:999px; font-size:12px; margin-right:6px; margin-bottom:6px; }}
    .box {{ border:1px solid #ddd; padding:12px; border-radius:10px; background:#fafafa; margin-top:12px; }}
    a {{ color: #0b57d0; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>NRL AI Predictions</h1>
  <p class="note">Automated predictions with model probabilities, odds comparison, and value detection.</p>

  {downloads}
  {top_value}

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

def _file_link(path: str, label: str) -> str:
    # On GitHub Pages these files live beside index.html, so relative links work.
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return f"<span class='pill'><a href='{path}' download>{label}</a></span>"
    return ""

def build_downloads_section() -> str:
    links = []
    links.append(_file_link("predictions.csv", "Download predictions.csv"))
    links.append(_file_link("odds.csv", "Download odds.csv"))
    links.append(_file_link("bet_log.csv", "Download bet_log.csv"))
    links.append(_file_link("accuracy.csv", "Download accuracy.csv"))
    links.append(_file_link("performance.csv", "Download performance.csv"))
    links.append(_file_link("bankroll_status.csv", "Download bankroll_status.csv"))
    links.append(_file_link("closing_odds.csv", "Download closing_odds.csv"))
    links.append(_file_link("results_cache.csv", "Download results_cache.csv"))

    links = [x for x in links if x]
    if not links:
        return "<div class='box'><h2>Downloads</h2><p class='note'>No files ready to download yet.</p></div>"

    return "<div class='box'><h2>Downloads</h2>" + "".join(links) + "</div>"

def build_top_value_section() -> str:
    preds = _safe_read_csv("predictions.csv")
    if preds.empty:
        return "<div class='box'><h2>Top Value Bets</h2><p class='note'>No predictions yet.</p></div>"

    # Need odds + value_flag columns to show anything meaningful
    needed = {"date", "kickoff_local", "home", "away", "home_win_prob", "home_odds", "away_odds"}
    if not needed.issubset(set(preds.columns)):
        return "<div class='box'><h2>Top Value Bets</h2><p class='note'>Odds/value not available yet.</p></div>"

    # If your predict.py writes "value_flag", use it, otherwise we still show best edges if possible
    df = preds.copy()

    # Optional: compute simple model edge if odds exist
    # home_edge = p - 1/odds
    df["home_edge"] = pd.NA
    df["away_edge"] = pd.NA

    try:
        df["home_edge"] = df["home_win_prob"] - (1.0 / df["home_odds"])
    except Exception:
        pass
    try:
        df["away_edge"] = (1.0 - df["home_win_prob"]) - (1.0 / df["away_odds"])
    except Exception:
        pass

    # Build a “best side” suggestion per match if edges are numeric
    def pick_side(row):
        he = row.get("home_edge")
        ae = row.get("away_edge")
        try:
            he = float(he)
        except Exception:
            he = float("nan")
        try:
            ae = float(ae)
        except Exception:
            ae = float("nan")

        if he != he and ae != ae:
            return None

        if (he == he) and (ae != ae or he >= ae):
            return ("HOME", he)
        if ae == ae:
            return ("AWAY", ae)
        return None

    picks = []
    for _, r in df.iterrows():
        p = pick_side(r)
        if not p:
            continue
        side, edge = p
        # only show meaningful value
        if edge != edge or edge < 0.03:
            continue
        picks.append((side, edge, r))

    if not picks:
        # If value_flag exists, still show those rows (even if edge calc failed)
        if "value_flag" in df.columns:
            vv = df[df["value_flag"].fillna("").astype(str).str.len() > 0].copy()
            if vv.empty:
                return "<div class='box'><h2>Top Value Bets</h2><p class='note'>No value bets flagged this run.</p></div>"
            show = vv[["date","kickoff_local","home","away","home_win_prob","home_odds","away_odds","value_flag"]].head(8)
            return "<div class='box'><h2>Top Value Bets</h2>" + show.to_html(index=False, escape=False) + "</div>"

        return "<div class='box'><h2>Top Value Bets</h2><p class='note'>No value bets flagged this run.</p></div>"

    # Sort by best edge descending and show top 8
    picks.sort(key=lambda x: x[1], reverse=True)
    rows = []
    for side, edge, r in picks[:8]:
        home = r.get("home")
        away = r.get("away")
        label = home if side == "HOME" else away
        odds = r.get("home_odds") if side == "HOME" else r.get("away_odds")
        prob = r.get("home_win_prob") if side == "HOME" else (1.0 - r.get("home_win_prob", 0.5))
        rows.append({
            "date": r.get("date"),
            "kickoff_local": r.get("kickoff_local"),
            "pick": f"{side}: {label}",
            "model_prob": round(float(prob), 3) if prob == prob else prob,
            "odds": odds,
            "edge": f"+{edge:.0%}",
        })

    show = pd.DataFrame(rows)
    return "<div class='box'><h2>Top Value Bets</h2>" + show.to_html(index=False, escape=False) + "</div>"

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
        downloads=build_downloads_section(),
        top_value=build_top_value_section(),
        table=table_html,
        bankroll=build_bankroll_section(),
        accuracy=build_accuracy_section(),
        clv_roi=build_clv_roi_section(),
    )

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()
