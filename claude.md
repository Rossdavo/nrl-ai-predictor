Project: NRL AI Predictor

Purpose:
Generate automated NRL match predictions, odds comparison,
value betting signals, and long-term accuracy tracking.

Execution pipeline:
1. odds_fetch.py
2. predict.py
3. tracker.py
4. archive.py
5. performance.py
6. report.py

Key modelling rules:
- Use attack/defence ratings when available
- Fallback probabilities = 0.50
- Value bet threshold >= 3% probability edge
- Travel adjustments applied
- Store ratings.json each run
