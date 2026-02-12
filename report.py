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
  <p class="small">Generated automatically via GitHub Actions.</p>
</body>
</html>
"""

def main():
    # Load predictions
    df = pd.read_csv("predictions.csv")

    # Preferred display column order
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

    # Keep only columns that actually exist (prevents crashes)
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # Clean NaN value flags
    if "value_flag" in df.columns:
        df["value_flag"] = df["value_flag"].fillna("")

    # Convert table to HTML
    table_html = df.to_html(index=False, escape=False)

    html = HTML_TEMPLATE.format(table=table_html)

    # Write index.html for GitHub Pages
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()
