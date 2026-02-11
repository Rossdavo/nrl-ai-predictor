import pandas as pd

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>NRL Trials – AI Predictions</title>
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
  <h1>NRL Trials – AI Predictions</h1>
  <p class="note">These are trial-game estimates (higher uncertainty due to rotations). Use as a guide for tipping/betting/performance analysis.</p>
  {table}
  <p class="small">Generated automatically via GitHub Actions.</p>
</body>
</html>
"""

def main():
    df = pd.read_csv("predictions.csv")

    # Friendly formatting
    df["home_win_prob"] = (df["home_win_prob"] * 100).round(1).astype(str) + "%"
    df.rename(columns={
        "date": "Date",
        "kickoff_local": "Kickoff",
        "venue": "Venue",
        "home": "Home",
        "away": "Away",
        "home_win_prob": "Home Win %",
        "exp_margin_home": "Exp Margin (Home)",
        "exp_total": "Exp Total",
        "confidence": "Confidence",
        "home_top_try_profiles": "Home Try Profiles (Top 3)",
        "away_top_try_profiles": "Away Try Profiles (Top 3)",
        "generated_at": "Generated",
    }, inplace=True)

    table_html = df[[
        "Date","Kickoff","Venue","Home","Away","Home Win %","Exp Margin (Home)","Exp Total","Confidence",
        "Home Try Profiles (Top 3)","Away Try Profiles (Top 3)","Generated"
    ]].to_html(index=False, escape=False)

    html = HTML_TEMPLATE.format(table=table_html)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()
