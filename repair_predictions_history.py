import csv
import os
import pandas as pd

SRC = "predictions_history.csv"
BACKUP = "predictions_history_backup.csv"
OUT = "predictions_history.csv"

NEW_COLS = [
    "run_id",
    "run_utc",
    "mode",
    "rating_mode",
    "date",
    "kickoff_local",
    "venue",
    "home",
    "away",
    "model_home_win_prob",
    "market_home_win_prob",
    "final_home_win_prob",
    "exp_margin_home",
    "exp_total",
    "confidence",
    "home_odds",
    "away_odds",
    "favourite_team",
    "underdog_team",
    "auto_upset_score",
    "auto_upset_reasons",
    "manual_upset_team",
    "manual_upset_score",
    "manual_upset_notes",
    "upset_team",
    "final_upset_score",
    "upset_flag",
    "fragile_favourite",
    "required_edge",
    "value_flag",
    "pick",
    "edge",
    "stake",
    "stake_units",
    "stake_dollars",
    "recommended_bet",
    "generated_at",
]

OLD_COLS_16 = [
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
    "generated_at",
]

OLD_COLS_18 = [
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


def blank_row():
    return {c: "" for c in NEW_COLS}


def map_old_to_new(row_dict: dict) -> dict:
    out = blank_row()

    out["run_id"] = row_dict.get("run_id", "")
    out["run_utc"] = row_dict.get("run_utc", "")
    out["date"] = row_dict.get("date", "")
    out["kickoff_local"] = row_dict.get("kickoff_local", "")
    out["home"] = row_dict.get("home", "")
    out["away"] = row_dict.get("away", "")

    hp = row_dict.get("home_win_prob", "")
    out["model_home_win_prob"] = hp
    out["final_home_win_prob"] = hp

    out["exp_margin_home"] = row_dict.get("exp_margin_home", "")
    out["home_odds"] = row_dict.get("home_odds", "")
    out["away_odds"] = row_dict.get("away_odds", "")
    out["pick"] = row_dict.get("pick", "")
    out["edge"] = row_dict.get("edge", "")
    out["stake"] = row_dict.get("stake", "")
    out["stake_units"] = row_dict.get("stake", "")
    out["stake_dollars"] = row_dict.get("stake_dollars", "")
    out["recommended_bet"] = row_dict.get("recommended_bet", "")
    out["generated_at"] = row_dict.get("generated_at", "")

    return out


def clean_value(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s in {"nan", "NaT", "<NA>"} else s


def main():
    if not os.path.exists(SRC):
        print(f"[warn] Missing {SRC}")
        return

    if not os.path.exists(BACKUP):
        with open(SRC, "r", encoding="utf-8", newline="") as fsrc, open(BACKUP, "w", encoding="utf-8", newline="") as fdst:
            fdst.write(fsrc.read())
        print(f"[info] Backup written: {BACKUP}")

    repaired = []

    with open(SRC, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("[warn] Empty history file")
        return

    data_rows = rows[1:]  # ignore whatever header is there

    for row in data_rows:
        if not row:
            continue

        if len(row) == len(NEW_COLS):
            row_dict = dict(zip(NEW_COLS, row))
            out = {c: clean_value(row_dict.get(c, "")) for c in NEW_COLS}
            repaired.append(out)
            continue

        if len(row) == len(OLD_COLS_18):
            row_dict = dict(zip(OLD_COLS_18, row))
            repaired.append(map_old_to_new(row_dict))
            continue

        if len(row) == len(OLD_COLS_16):
            row_dict = dict(zip(OLD_COLS_16, row))
            repaired.append(map_old_to_new(row_dict))
            continue

        # fallback trim/pad
        if len(row) > len(NEW_COLS):
            row = row[:len(NEW_COLS)]
            row_dict = dict(zip(NEW_COLS, row))
            out = {c: clean_value(row_dict.get(c, "")) for c in NEW_COLS}
            repaired.append(out)
        else:
            padded = row + [""] * (len(NEW_COLS) - len(row))
            row_dict = dict(zip(NEW_COLS, padded))
            out = {c: clean_value(row_dict.get(c, "")) for c in NEW_COLS}
            repaired.append(out)

    df = pd.DataFrame(repaired, columns=NEW_COLS)

    # normalize a few numeric fields
    for col in [
        "model_home_win_prob", "market_home_win_prob", "final_home_win_prob",
        "exp_margin_home", "exp_total", "confidence",
        "home_odds", "away_odds",
        "auto_upset_score", "manual_upset_score", "final_upset_score",
        "required_edge", "edge", "stake", "stake_units", "stake_dollars",
        "upset_flag", "fragile_favourite"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["generated_at"] = df["generated_at"].fillna("").astype(str)

    df.to_csv(OUT, index=False, quoting=csv.QUOTE_ALL)
    print(f"[info] Repaired history written: {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
