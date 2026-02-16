# NRL AI Predictor — Troubleshooting Guide

Use this when the workflow runs red or the website is not updating.

---

# 1. Website not updating
### Step 1
Go to:
Actions → Run NRL predictions

Click the **latest run**.

### Step 2
Check if:
- All steps are green
- Deploy to GitHub Pages step ran

### Step 3
If green but site not updating:
- Hard refresh browser
  - Windows: Ctrl + F5
  - Mac: Cmd + Shift + R

---

# 2. Workflow failed (red X)
Click the failed step.

Scroll to the **first error line** (not the last).

Common causes:

## IndentationError
Cause: spacing mistake in Python

Fix:
- open the file shown in error
- re-align indentation to match surrounding lines

---

## File not found
Example:
python: can't open file 'tracker.py'

Fix:
- confirm file exists in repo
- confirm workflow step spelling matches filename

---

## EmptyDataError
Cause:
CSV file exists but is empty

Fix:
Add safe load:

try:
    df = pd.read_csv("file.csv")
except:
    df = pd.DataFrame()

---

## Feed timeout
Cause:
external data site slow

Fix:
Re-run workflow once manually:
Actions → Run workflow

Usually resolves automatically.

---

# 3. Odds missing (NaN odds)
Check:
- ODDS_API_KEY exists in Repo → Settings → Secrets
- odds_fetch.py ran successfully in logs
- odds.csv updated message appears

If odds.csv updated but predictions show NaN:
- team names may not match
- check TEAM_ALIASES mapping

---

# 4. Accuracy not updating
Check:
- results_cache.csv exists
- tracker.py ran successfully
- team names match between fixtures and results feed

---

# 5. Profit / ROI not showing
Check:
- bet_log.csv exists
- performance.py ran
- performance.csv exists

---

# 6. Quick restart procedure (fastest fix)
1. Commit any changes
2. Actions → Run workflow manually
3. Wait until green
4. Refresh website

---

# 7. Emergency rollback (rare)
If a major code change breaks everything:
- open last working commit
- copy predict.py / report.py
- paste back into current repo
- commit
- run workflow again

---

End
