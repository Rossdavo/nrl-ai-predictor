import pandas as pd

path = "predictions_history.csv"

df = pd.read_csv(path)

print("Before fix columns:")
print(df.columns.tolist())

# Add missing try scorer columns
if "home_top_try" not in df.columns:
    df["home_top_try"] = ""

if "away_top_try" not in df.columns:
    df["away_top_try"] = ""

# Correct full column order (VERY IMPORTANT)
desired_cols = [
    "run_id",
    "run_utc",
    "date",
    "kickoff_local",
    "home",
    "away",
    "home_win_prob",
    "exp_margin_home",
    "home_odds",
    "away_odds",
    "pick",
    "edge",
    "stake",
    "stake_dollars",
    "recommended_bet",
    "home_top_try",
    "away_top_try",
    "generated_at",
]

# Add any missing columns safely
for col in desired_cols:
    if col not in df.columns:
        df[col] = ""

# Reorder columns cleanly
df = df[desired_cols]

df.to_csv(path, index=False)

print("✅ predictions_history.csv repaired successfully")
