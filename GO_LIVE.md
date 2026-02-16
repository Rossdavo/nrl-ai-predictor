# NRL AI Predictor — Go-Live Checklist (Round 1 Week)

Use this checklist in the week leading into Round 1 (and again after Round 1 starts).

---

## 1) Switch to AUTO fixtures
File: `predict.py`

Change:

MODE = "TRIALS"

to:

MODE = "AUTO"

Commit.

---

## 2) Run a manual test
GitHub:
- Actions → Run NRL predictions → Run workflow

Confirm green tick.

---

## 3) Confirm the website is updating
Open your GitHub Pages site:
- Hard refresh:
  - Windows: Ctrl + F5
  - Mac: Cmd + Shift + R

You should see:
- Predictions table
- Profit tracker section (may show "no bets yet")
- CLV/ROI section (may show "no performance data yet")
- Accuracy section (may show "no scored games yet")

---

## 4) Check fixture feed sanity
In Actions log (predict.py output):
- Confirm upcoming fixtures are Round 1 games (not trials).
- Confirm dates/kickoff times look correct for Sydney time.

If fixtures are wrong:
- check `FIXTURE_FEED_URL`
- confirm `MODE = "AUTO"`

---

## 5) Check team name matching (MOST IMPORTANT)
Team names must match across:
- Fixture feed (predict.py fixtures)
- Odds feed (odds_fetch.py odds.csv)
- Results feed (tracker.py / results_cache.csv)

Symptoms of mismatch:
- odds columns show NaN for games that should have odds
- tracker says no completed results even after games finished
- accuracy/performance never populates

Fix:
- Add / extend `TEAM_ALIASES` mapping
- Normalize home/away team names at ingestion in:
  - fixtures ingest
  - odds ingest
  - results ingest

---

## 6) Confirm odds are being pulled
Actions log should show something like:
- "odds.csv updated (X rows)"

If odds missing:
- confirm ODDS_API_KEY secret exists
- confirm the sport key/market in odds_fetch.py is correct
- confirm bookmaker odds exist for those games

---

## 7) After the first completed match
After Round 1 games finish:
- run workflow manually once
Expected:
- `results_cache.csv` appears
- `accuracy.csv` starts filling
- `performance.csv` starts filling

If results don’t appear:
- check Results URL/feed
- check team name matching (Step 5)

---

## 8) Page publishing check
Workflow should copy files into `site/` and deploy Pages artifact.

Ensure `Prepare site` includes:
- index.html
- predictions.csv
- odds.csv (optional but preferred)
- accuracy.csv (optional)
- performance.csv (optional)
- bet_log.csv (optional)
- predictions_history.csv (optional)
- odds_history.csv (optional)
- ratings.json (optional)
- results_cache.csv (optional)

---

## 9) Weekly maintenance (2 minutes)
Once per week:
- run workflow manually
- scan Actions log for:
  - timeouts / feed errors
  - NaN odds where odds should exist
  - "empty data" crashes (should not happen)

---

## 10) If something breaks
Fastest triage order:
1) Check Actions log for the first red traceback line
2) Fix indentation or missing file reference
3) If feed timed out, re-run workflow
4) If mismatch issue, update TEAM_ALIASES

End.
