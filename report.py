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
  <p class="note">Trial-game estimates (higher uncertainty due to rotations). Try scorers are based on starters (1–13) where team lists are available.</p>
  {table}
  <p class="small">Generated automatically via GitHub Actions.</p>
</body>
</html>
"""
