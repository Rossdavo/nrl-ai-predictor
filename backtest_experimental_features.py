#!/usr/bin/env python3
"""
backtest_experimental_features.py

Evaluates experimental pre-match features against the frozen 2026 model.

Inputs:
    performance.csv
    historical_features.csv

Outputs:
    experimental_feature_backtest.csv
    experimental_feature_match_audit.csv

This script DOES NOT alter model probabilities. It asks:
- When a feature strongly favoured home/away, which side actually won?
- When that signal disagreed with the baseline tip, did it correct or damage it?
- What threshold gives useful coverage without relying on tiny samples?
"""

from __future__ import annotations
import os
import math
import pandas as pd

PERFORMANCE = "performance.csv"
FEATURES = "historical_features.csv"
OUT_SUMMARY = "experimental_feature_backtest.csv"
OUT_AUDIT = "experimental_feature_match_audit.csv"

# Thresholds are diagnostic cut-points, not model weights.
TESTS = {
    "momentum_streak_edge_home": [1, 2, 3, 4],
    "momentum_last5_win_pct_edge_home": [0.10, 0.20, 0.30, 0.40],
    "momentum_last8_win_pct_edge_home": [0.10, 0.20, 0.30, 0.40],
    "momentum_last5_margin_edge_home": [3, 6, 9, 12],
    "momentum_last8_margin_edge_home": [3, 6, 9, 12],
    "continuity_retention_edge_home": [0.05, 0.10, 0.15, 0.20],
    "experience_nrl_edge_home": [100, 250, 500, 750],
    "experience_spine_edge_home": [50, 100, 200, 300],
    "experience_finals_edge_home": [5, 10, 20, 30],
    "experience_rep_edge_home": [5, 10, 20, 30],
    "experience_sl_edge_home": [25, 50, 100, 150],
}

def norm(x):
    return " ".join(str(x).strip().upper().split())

def load():
    if not os.path.exists(PERFORMANCE):
        raise FileNotFoundError(PERFORMANCE)
    if not os.path.exists(FEATURES):
        raise FileNotFoundError(FEATURES)

    p = pd.read_csv(PERFORMANCE)
    f = pd.read_csv(FEATURES)

    for d in (p, f):
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        d["_home"] = d["home"].map(norm)
        d["_away"] = d["away"].map(norm)

    # One scored prediction per match is expected from performance.py.
    p = p.drop_duplicates(["date","_home","_away"], keep="last")
    f = f.drop_duplicates(["date","_home","_away"], keep="last")

    m = p.merge(
        f.drop(columns=["home","away"], errors="ignore"),
        on=["date","_home","_away"],
        how="left",
        suffixes=("","_feature"),
    )
    return m

def actual_side(r):
    aw = norm(r.get("actual_winner",""))
    if aw == norm(r["home"]): return "HOME"
    if aw == norm(r["away"]): return "AWAY"
    return "DRAW"

def baseline_side(r):
    pw = norm(r.get("predicted_winner",""))
    if pw == norm(r["home"]): return "HOME"
    if pw == norm(r["away"]): return "AWAY"
    return ""

def evaluate_test(df, feature, threshold):
    x = pd.to_numeric(df[feature], errors="coerce")
    eligible = df[x.abs() >= threshold].copy()
    eligible["_signal_value"] = x.loc[eligible.index]
    eligible["_signal_side"] = eligible["_signal_value"].map(lambda v: "HOME" if v > 0 else "AWAY")
    eligible["_actual_side"] = eligible.apply(actual_side, axis=1)
    eligible["_baseline_side"] = eligible.apply(baseline_side, axis=1)
    eligible = eligible[eligible["_actual_side"] != "DRAW"].copy()

    n = len(eligible)
    if n == 0:
        return None, eligible

    signal_correct = (eligible["_signal_side"] == eligible["_actual_side"])
    baseline_correct = (eligible["_baseline_side"] == eligible["_actual_side"])
    disagreements = eligible["_signal_side"] != eligible["_baseline_side"]
    dis = eligible[disagreements].copy()

    corrected = ((eligible["_signal_side"] == eligible["_actual_side"]) &
                 (eligible["_baseline_side"] != eligible["_actual_side"]))
    broken = ((eligible["_signal_side"] != eligible["_actual_side"]) &
              (eligible["_baseline_side"] == eligible["_actual_side"]))

    # Net corrected winners if the signal were blindly substituted only on
    # qualifying disagreements. This is NOT a recommended live rule.
    net_corrections = int(corrected.sum() - broken.sum())

    row = {
        "feature": feature,
        "threshold_abs": threshold,
        "eligible_matches": n,
        "coverage_pct": n / max(1, len(df)),
        "signal_accuracy": signal_correct.mean(),
        "baseline_accuracy_same_matches": baseline_correct.mean(),
        "accuracy_delta": signal_correct.mean() - baseline_correct.mean(),
        "disagreements_with_baseline": int(disagreements.sum()),
        "baseline_errors_corrected": int(corrected.sum()),
        "baseline_winners_broken": int(broken.sum()),
        "net_corrected_winners": net_corrections,
        "disagreement_signal_accuracy": (
            (dis["_signal_side"] == dis["_actual_side"]).mean() if len(dis) else math.nan
        ),
        "sample_flag": (
            "TOO_SMALL" if n < 20 else
            "CAUTION" if n < 40 else
            "USABLE"
        ),
    }

    audit = eligible[[
        "date","home","away","predicted_winner","actual_winner",
        "_signal_value","_signal_side","_baseline_side","_actual_side"
    ]].copy()
    audit["feature"] = feature
    audit["threshold_abs"] = threshold
    audit["signal_correct"] = audit["_signal_side"] == audit["_actual_side"]
    audit["baseline_correct"] = audit["_baseline_side"] == audit["_actual_side"]
    audit["corrected_baseline_error"] = audit["signal_correct"] & ~audit["baseline_correct"]
    audit["broke_baseline_winner"] = ~audit["signal_correct"] & audit["baseline_correct"]
    return row, audit

def main():
    df = load()
    if df.empty:
        print("[feature_backtest] no merged matches")
        return

    base_scored = pd.to_numeric(df.get("correct"), errors="coerce").dropna()
    base_acc = base_scored.mean() if len(base_scored) else math.nan
    print(f"[feature_backtest] merged matches: {len(df)}")
    print(f"[feature_backtest] frozen baseline accuracy: {base_acc:.1%}" if not math.isnan(base_acc) else
          "[feature_backtest] frozen baseline accuracy: n/a")

    summary=[]
    audits=[]
    for feature, thresholds in TESTS.items():
        if feature not in df.columns:
            continue
        for t in thresholds:
            row,audit=evaluate_test(df,feature,t)
            if row:
                summary.append(row)
                if not audit.empty:
                    audits.append(audit)

    s=pd.DataFrame(summary)
    if not s.empty:
        # Evidence-first ranking: useful sample first, then net corrected winners,
        # then accuracy delta. This is descriptive, not an automatic model choice.
        rank={"USABLE":2,"CAUTION":1,"TOO_SMALL":0}
        s["_sample_rank"]=s["sample_flag"].map(rank)
        s=s.sort_values(
            ["_sample_rank","net_corrected_winners","accuracy_delta","eligible_matches"],
            ascending=[False,False,False,False]
        ).drop(columns="_sample_rank")
    s.to_csv(OUT_SUMMARY,index=False)

    if audits:
        pd.concat(audits,ignore_index=True).to_csv(OUT_AUDIT,index=False)
    else:
        pd.DataFrame().to_csv(OUT_AUDIT,index=False)

    print(f"[feature_backtest] wrote {OUT_SUMMARY}: {len(s)} tests")
    print(f"[feature_backtest] wrote {OUT_AUDIT}")
    if not s.empty:
        print("\nTop diagnostic results:")
        cols=["feature","threshold_abs","eligible_matches","signal_accuracy",
              "baseline_accuracy_same_matches","accuracy_delta",
              "baseline_errors_corrected","baseline_winners_broken",
              "net_corrected_winners","sample_flag"]
        print(s[cols].head(12).to_string(index=False))

if __name__ == "__main__":
    main()
