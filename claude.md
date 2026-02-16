# NRL AI Predictor — Project Instructions (CLAUDE.md)

You are working in the repository **nrl-ai-predictor**.

## Goal
Generate automated NRL predictions + publish to GitHub Pages daily:
- Predictions table (`predictions.csv` + `index.html`)
- Odds comparison (`odds.csv`)
- Archived histories (`predictions_history.csv`, `odds_history.csv`)
- Accuracy tracking (`accuracy.csv`) once results exist
- Performance/CLV summary (`performance.csv`) once accuracy exists
- Bet logging + settlement (`bet_log.csv`) once value bets exist

This is a novice-maintained repo. Changes must be safe, minimal, and avoid breaking GitHub Actions.

---

## Golden rules (do not break these)
1. **Never commit secrets** (API keys). Use GitHub Secrets only.
2. **Never crash if a CSV exists but is empty.**
   - Wrap `pd.read_csv()` in try/except or check `os.path.getsize()`.
3. **Indentation must be consistent** (4 spaces). Never mix tabs.
4. Any HTML template using `.format()` must escape braces:
   - Use `{{` and `}}` inside CSS.

---

## Current pipeline (must stay in this order)
GitHub Actions should run in this exact order:

1) `python odds_fetch.py`  
2) `python predict.py`  
3) `python tracker.py`  
4) `python archive.py`  
5) `python performance.py`  
6) `python report.py`

`report.py` must always succeed even if:
- `results_cache.csv` does not exist yet
- `accuracy.csv` does not exist yet
- `performance.csv` exists but is empty
- `bet_log.csv` does not exist yet

---

## Modes
`predict.py` supports:
- `MODE = "TRIALS"` (hardcoded fixtures)
- `MODE = "AUTO"` (pull upcoming fixtures automatically)

Default can remain TRIALS until Round 1 week. During Round 1 week switch to AUTO.

---

## File responsibilities (what each script does)
- `odds_fetch.py`: pulls bookmaker odds and writes `odds.csv`
- `predict.py`: generates `predictions.csv`
- `tracker.py`: fetches completed match results and updates `accuracy.csv` (and/or `results_cache.csv`)
- `archive.py`: appends `predictions.csv` and `odds.csv` into history files
- `performance.py`: produces `performance.csv` using historical predictions/odds + scored results
- `report.py`: builds `index.html` from predictions + summary sections (profit/performance/accuracy)

---

## Publishing (GitHub Pages)
Workflow must create a `site/` folder and copy:
- `index.html`
- `predictions.csv`
- optional: `odds.csv`
- optional: `accuracy.csv`
- optional: `performance.csv`
- optional: `bet_log.csv`
- optional: `predictions_history.csv`
- optional: `odds_history.csv`
- optional: `ratings.json`
- optional: `results_cache.csv`

Then upload as Pages artifact and deploy.

---

## Team name normalization (Round 1 readiness)
Team names must match across fixture feed, odds feed, and results feed.
If mismatches appear, add a `TEAM_ALIASES` mapping and normalize in:
- fixtures ingest
- results ingest
- odds ingest

Never silently drop games because of naming mismatch.

---

## Value betting logic
Value bet flagging should be conservative:
- Value threshold: probability edge >= 0.03 (3%)
- If odds missing (NaN), do not flag value
- Never crash if odds file is missing or has missing rows

---

## Output stability
- `predictions.csv` column order should stay stable for the website.
- If a new column is added, `report.py` must handle it safely (select only existing cols).
- Prefer deterministic seeds for simulation so runs are comparable.

---

## When changing anything
- Keep diffs small.
- Avoid refactors unless necessary.
- Always ensure `python predict.py` and `python report.py` can run locally and in GitHub Actions.
- If adding new files, update `requirements.txt` only if needed.

End.
