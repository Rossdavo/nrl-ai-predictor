#!/usr/bin/env python3
"""
build_historical_features.py

Creates pre-match experimental features for every completed match.

Inputs:
    results_cache.csv
    team_lists_history.csv            optional, for continuity
    player_experience.csv             optional, for experience once populated

Output:
    historical_features.csv

Critical rule:
    A match may only use information available BEFORE that match.
    Same-day matches do not see each other's results.
"""

from __future__ import annotations
import os
import pandas as pd

RESULTS_PATH = "results_cache.csv"
TEAM_LISTS_PATH = "team_lists_history.csv"
EXPERIENCE_PATH = "player_experience.csv"
OUT_PATH = "historical_features.csv"

OUTPUT_COLUMNS = [
    "date","home","away",
    "home_games_before","away_games_before",
    "home_win_streak","away_win_streak",
    "home_last5_win_pct","away_last5_win_pct",
    "home_last8_win_pct","away_last8_win_pct",
    "home_last5_avg_margin","away_last5_avg_margin",
    "home_last8_avg_margin","away_last8_avg_margin",
    "momentum_streak_edge_home",
    "momentum_last5_win_pct_edge_home",
    "momentum_last8_win_pct_edge_home",
    "momentum_last5_margin_edge_home",
    "momentum_last8_margin_edge_home",
    "home_players_retained","away_players_retained",
    "home_lineup_changes","away_lineup_changes",
    "home_retention_pct","away_retention_pct",
    "home_spine_players_retained","away_spine_players_retained",
    "home_spine_changes","away_spine_changes",
    "home_spine_retention_pct","away_spine_retention_pct",
    "continuity_retention_edge_home",
    "continuity_spine_retention_edge_home",
    "home_nrl_games_total","away_nrl_games_total",
    "home_spine_nrl_games","away_spine_nrl_games",
    "home_finals_games_total","away_finals_games_total",
    "home_origin_games_total","away_origin_games_total",
    "home_test_games_total","away_test_games_total",
    "home_super_league_games_total","away_super_league_games_total",
    "experience_nrl_edge_home","experience_spine_edge_home",
    "experience_finals_edge_home","experience_rep_edge_home",
    "experience_sl_edge_home",
    "experience_home_known_players","experience_away_known_players",
]

def norm_team(x):
    return " ".join(str(x).strip().upper().split())

def load_results():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(RESULTS_PATH)
    df = pd.read_csv(RESULTS_PATH)
    req = {"date","home","away","home_pts","away_pts"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{RESULTS_PATH} missing columns: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_pts"] = pd.to_numeric(df["home_pts"], errors="coerce")
    df["away_pts"] = pd.to_numeric(df["away_pts"], errors="coerce")
    df = df.dropna(subset=["date","home","away","home_pts","away_pts"])
    df = df.drop_duplicates(["date","home","away"], keep="last")
    return df.sort_values(["date","home","away"]).reset_index(drop=True)

def team_history_before(results, team, before_date):
    key = norm_team(team)
    prior = results[results["date"] < before_date]
    rows = []
    for _, r in prior.iterrows():
        if norm_team(r["home"]) == key:
            pf, pa = float(r["home_pts"]), float(r["away_pts"])
        elif norm_team(r["away"]) == key:
            pf, pa = float(r["away_pts"]), float(r["home_pts"])
        else:
            continue
        rows.append({"won": pf > pa, "margin": pf-pa, "date": r["date"]})
    return rows

def streak(history):
    n=0
    for r in reversed(history):
        if r["won"]: n += 1
        else: break
    return n

def recent(history, n):
    s=history[-n:]
    if not s:
        return None, None
    return sum(r["won"] for r in s)/len(s), sum(r["margin"] for r in s)/len(s)

def load_team_lists():
    """Load archived team-list snapshots.

    Current team_changes.py history uses round_start / round_end rather than a
    generic date column.  We preserve those windows so a historical match can
    only use the team list archived for the round containing that match.
    """
    if not os.path.exists(TEAM_LISTS_PATH):
        return pd.DataFrame()
    try:
        d = pd.read_csv(TEAM_LISTS_PATH)
    except Exception:
        return pd.DataFrame()

    for c in ["round_start", "round_end", "captured_at", "date"]:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")
    return d

def _team_snapshots(team_lists, team):
    if team_lists.empty or "team" not in team_lists.columns:
        return pd.DataFrame()
    d = team_lists[team_lists["team"].map(norm_team) == norm_team(team)].copy()
    if d.empty:
        return d

    # Preferred schema produced by team_changes.py.
    if {"round_start", "round_end"}.issubset(d.columns):
        d = d.dropna(subset=["round_start", "round_end"])
        return d

    # Backward-compatible support for any older date-based history.
    if "date" in d.columns:
        d = d.dropna(subset=["date"]).copy()
        d["round_start"] = d["date"]
        d["round_end"] = d["date"]
        return d
    return pd.DataFrame()

def _snapshot_round_for_match(team_lists, team, match_date):
    """Return the archived round window that actually contains match_date.

    If overlapping windows exist, prefer the most recently starting, then the
    shortest window.  This is important around finals where archived windows
    can overlap by a boundary date.
    """
    d = _team_snapshots(team_lists, team)
    if d.empty:
        return None
    rounds = d[["round_start", "round_end"]].drop_duplicates().copy()
    rounds = rounds[(rounds["round_start"] <= match_date) & (rounds["round_end"] >= match_date)]
    if rounds.empty:
        return None
    rounds["window_days"] = (rounds["round_end"] - rounds["round_start"]).dt.days
    rounds = rounds.sort_values(["round_start", "window_days"], ascending=[False, True])
    r = rounds.iloc[0]
    return r["round_start"], r["round_end"]

def _selected_from_round(team_lists, team, round_key):
    if round_key is None:
        return []
    d = _team_snapshots(team_lists, team)
    if d.empty:
        return []
    rs, re = round_key
    d = d[(d["round_start"] == rs) & (d["round_end"] == re)].copy()

    jersey_col = next((c for c in ["jersey", "number", "position_number"] if c in d.columns), None)
    player_col = next((c for c in ["player", "player_name", "name"] if c in d.columns), None)
    if not jersey_col or not player_col:
        return []

    d[jersey_col] = pd.to_numeric(d[jersey_col], errors="coerce")
    d = d[d[jersey_col].between(1, 17)].copy()
    d = d.dropna(subset=[jersey_col, player_col])
    d = d.drop_duplicates(subset=[jersey_col], keep="last")

    # Only trust complete 1-17 snapshots. Missing history stays missing rather
    # than being silently treated as zero changes.
    if set(d[jersey_col].astype(int)) != set(range(1, 18)) or len(d) != 17:
        return []
    return [(int(r[jersey_col]), str(r[player_col]).strip()) for _, r in d.iterrows()]

def selected_17(team_lists, team, match_date):
    return _selected_from_round(
        team_lists, team, _snapshot_round_for_match(team_lists, team, match_date)
    )

def prior_selected_17(team_lists, team, match_date):
    current = _snapshot_round_for_match(team_lists, team, match_date)
    if current is None:
        return []
    d = _team_snapshots(team_lists, team)
    if d.empty:
        return []
    current_start = current[0]
    rounds = d[["round_start", "round_end"]].drop_duplicates().copy()
    rounds = rounds[rounds["round_start"] < current_start]
    if rounds.empty:
        return []
    rounds = rounds.sort_values(["round_start", "round_end"], ascending=[False, False])
    r = rounds.iloc[0]
    return _selected_from_round(team_lists, team, (r["round_start"], r["round_end"]))

def continuity(current, previous):
    cur = {p for _, p in current if p}
    prev = {p for _, p in previous if p}
    if len(cur) != 17 or len(prev) != 17:
        return None, None, None
    retained = len(cur & prev)
    changes = len(cur - prev)
    return retained, changes, retained / 17.0

def spine_continuity(current, previous):
    """Continuity of the conventional NRL spine jerseys 1, 6, 7 and 9."""
    cur = {j: p for j, p in current if j in (1, 6, 7, 9) and p}
    prev = {j: p for j, p in previous if j in (1, 6, 7, 9) and p}
    if set(cur) != {1, 6, 7, 9} or set(prev) != {1, 6, 7, 9}:
        return None, None, None
    retained = sum(cur[j] == prev[j] for j in (1, 6, 7, 9))
    changes = 4 - retained
    return retained, changes, retained / 4.0

def load_experience():
    if not os.path.exists(EXPERIENCE_PATH):
        return pd.DataFrame()
    try:
        d=pd.read_csv(EXPERIENCE_PATH)
    except Exception:
        return pd.DataFrame()
    if "player" not in d.columns:
        return pd.DataFrame()
    # One career record per player is enough for this layer.
    d=d.drop_duplicates("player",keep="last").copy()
    return d.set_index("player",drop=False)

def squad_experience(selected, exp):
    fields=[
        "nrl_games","nrl_finals_games","state_of_origin_games",
        "test_international_games","super_league_games"
    ]
    out={f:None for f in fields}
    if not selected or exp.empty:
        return out | {"spine_nrl_games":None,"known_players":0}
    totals={f:0.0 for f in fields}
    known={f:0 for f in fields}
    spine=0.0; spine_known=0; any_known=set()
    for jersey,name in selected:
        if name not in exp.index: continue
        r=exp.loc[name]
        if isinstance(r,pd.DataFrame): r=r.iloc[-1]
        for f in fields:
            if f not in r.index: continue
            v=pd.to_numeric(pd.Series([r[f]]),errors="coerce").iloc[0]
            if pd.notna(v):
                totals[f]+=float(v); known[f]+=1; any_known.add(name)
                if f=="nrl_games" and jersey in (1,6,7,9):
                    spine+=float(v); spine_known+=1
    for f in fields:
        out[f]=totals[f] if known[f] else None
    out["spine_nrl_games"]=spine if spine_known else None
    out["known_players"]=len(any_known)
    return out

def edge(a,b):
    return (a-b) if a is not None and b is not None else None

def build():
    results=load_results()
    lists=load_team_lists()
    exp=load_experience()
    rows=[]

    for _,m in results.iterrows():
        date=m["date"]; home=m["home"]; away=m["away"]
        hh=team_history_before(results,home,date)
        ah=team_history_before(results,away,date)

        h5w,h5m=recent(hh,5); a5w,a5m=recent(ah,5)
        h8w,h8m=recent(hh,8); a8w,a8m=recent(ah,8)
        hs,as_=streak(hh),streak(ah)

        hc=selected_17(lists,home,date)
        ac=selected_17(lists,away,date)
        hp=prior_selected_17(lists,home,date)
        ap=prior_selected_17(lists,away,date)
        hr,hchg,hrp=continuity(hc,hp)
        ar,achg,arp=continuity(ac,ap)
        hsr,hschg,hsrp=spine_continuity(hc,hp)
        asr,aschg,asrp=spine_continuity(ac,ap)

        he=squad_experience(hc,exp)
        ae=squad_experience(ac,exp)
        hrep=(he["state_of_origin_games"] or 0)+(he["test_international_games"] or 0) if (
            he["state_of_origin_games"] is not None or he["test_international_games"] is not None) else None
        arep=(ae["state_of_origin_games"] or 0)+(ae["test_international_games"] or 0) if (
            ae["state_of_origin_games"] is not None or ae["test_international_games"] is not None) else None

        rows.append({
            "date":date.strftime("%Y-%m-%d"),"home":home,"away":away,
            "home_games_before":len(hh),"away_games_before":len(ah),
            "home_win_streak":hs,"away_win_streak":as_,
            "home_last5_win_pct":h5w,"away_last5_win_pct":a5w,
            "home_last8_win_pct":h8w,"away_last8_win_pct":a8w,
            "home_last5_avg_margin":h5m,"away_last5_avg_margin":a5m,
            "home_last8_avg_margin":h8m,"away_last8_avg_margin":a8m,
            "momentum_streak_edge_home":edge(hs,as_),
            "momentum_last5_win_pct_edge_home":edge(h5w,a5w),
            "momentum_last8_win_pct_edge_home":edge(h8w,a8w),
            "momentum_last5_margin_edge_home":edge(h5m,a5m),
            "momentum_last8_margin_edge_home":edge(h8m,a8m),
            "home_players_retained":hr,"away_players_retained":ar,
            "home_lineup_changes":hchg,"away_lineup_changes":achg,
            "home_retention_pct":hrp,"away_retention_pct":arp,
            "home_spine_players_retained":hsr,"away_spine_players_retained":asr,
            "home_spine_changes":hschg,"away_spine_changes":aschg,
            "home_spine_retention_pct":hsrp,"away_spine_retention_pct":asrp,
            "continuity_retention_edge_home":edge(hrp,arp),
            "continuity_spine_retention_edge_home":edge(hsrp,asrp),
            "home_nrl_games_total":he["nrl_games"],"away_nrl_games_total":ae["nrl_games"],
            "home_spine_nrl_games":he["spine_nrl_games"],"away_spine_nrl_games":ae["spine_nrl_games"],
            "home_finals_games_total":he["nrl_finals_games"],"away_finals_games_total":ae["nrl_finals_games"],
            "home_origin_games_total":he["state_of_origin_games"],"away_origin_games_total":ae["state_of_origin_games"],
            "home_test_games_total":he["test_international_games"],"away_test_games_total":ae["test_international_games"],
            "home_super_league_games_total":he["super_league_games"],"away_super_league_games_total":ae["super_league_games"],
            "experience_nrl_edge_home":edge(he["nrl_games"],ae["nrl_games"]),
            "experience_spine_edge_home":edge(he["spine_nrl_games"],ae["spine_nrl_games"]),
            "experience_finals_edge_home":edge(he["nrl_finals_games"],ae["nrl_finals_games"]),
            "experience_rep_edge_home":edge(hrep,arep),
            "experience_sl_edge_home":edge(he["super_league_games"],ae["super_league_games"]),
            "experience_home_known_players":he["known_players"],
            "experience_away_known_players":ae["known_players"],
        })

    return pd.DataFrame(rows,columns=OUTPUT_COLUMNS)

if __name__=="__main__":
    out=build()
    out.to_csv(OUT_PATH,index=False)
    print(f"[historical_features] wrote {OUT_PATH}: {len(out)} completed matches")
    if not out.empty:
        print(f"[historical_features] momentum coverage: {out['home_last5_win_pct'].notna().mean():.1%}")
        print(f"[historical_features] continuity coverage: {out['home_retention_pct'].notna().mean():.1%}")
        print(f"[historical_features] experience coverage: {out['home_nrl_games_total'].notna().mean():.1%}")
