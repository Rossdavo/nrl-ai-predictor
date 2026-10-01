#!/usr/bin/env python3
"""Fast/fail-safe RLP 2026 season player population collector."""
from __future__ import annotations
import re, signal, sys, time, unicodedata
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

SEASON=2026
URL=f"https://www.rugbyleagueproject.org/seasons/nrl-{SEASON}/players.html"
OUTPUT=Path("rlp_season_players.csv")
FAILURES=Path("rlp_season_players_failures.csv")
SUMMARY=Path("rlp_season_players_summary.csv")
OPTIONAL_SEED=Path("nrl_roster_seed_2026_2027.csv")
CONNECT_TIMEOUT=5; READ_TIMEOUT=15; HARD_FETCH_SECONDS=25
USER_AGENT="Mozilla/5.0 (compatible; NRL-AI-Predictor-Research/1.0; +https://github.com/rossdavo/nrl-ai-predictor)"
TEAM_CODE_MAP={
 "BRI":"Broncos","CAN":"Raiders","CBY":"Bulldogs","CRO":"Sharks","DOL":"Dolphins",
 "GLD":"Titans","MAN":"Sea Eagles","MEL":"Storm","NEW":"Knights","NQL":"Cowboys",
 "WAR":"Warriors","PAR":"Eels","PEN":"Panthers","SOU":"Rabbitohs","SGI":"Dragons",
 "SYD":"Roosters","WST":"Wests Tigers"}
EXPECTED_2026_TEAMS=set(TEAM_CODE_MAP.values())
POSITION_MAP={"FB":"Fullback","W":"Wing","C":"Centre","FE":"Five-eighth","HB":"Halfback","FR":"Front row","HK":"Hooker","2R":"Second row","L":"Lock","B":"Bench"}
TEAM_ALIASES={
 "brisbane broncos":"Broncos","broncos":"Broncos","canberra raiders":"Raiders","raiders":"Raiders",
 "canterbury bankstown bulldogs":"Bulldogs","bulldogs":"Bulldogs","cronulla sharks":"Sharks","sharks":"Sharks",
 "dolphins":"Dolphins","gold coast titans":"Titans","titans":"Titans","manly sea eagles":"Sea Eagles",
 "sea eagles":"Sea Eagles","melbourne storm":"Storm","storm":"Storm","newcastle knights":"Knights","knights":"Knights",
 "north queensland cowboys":"Cowboys","cowboys":"Cowboys","new zealand warriors":"Warriors","warriors":"Warriors",
 "parramatta eels":"Eels","eels":"Eels","penrith panthers":"Panthers","panthers":"Panthers",
 "south sydney rabbitohs":"Rabbitohs","rabbitohs":"Rabbitohs","st george illawarra dragons":"Dragons","dragons":"Dragons",
 "sydney roosters":"Roosters","roosters":"Roosters","wests tigers":"Wests Tigers","tigers":"Wests Tigers","perth bears":"Perth Bears"}

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}",flush=True)
def clean(v): return "" if pd.isna(v) else re.sub(r"\s+"," ",str(v)).strip()
def key(v):
 t=unicodedata.normalize("NFKD",clean(v)); t="".join(c for c in t if not unicodedata.combining(c)).lower().replace("’","'")
 return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9' -]","",t)).strip()
def title_player(raw):
 raw=clean(raw)
 if "," in raw:
  surname,given=[clean(x) for x in raw.split(",",1)]; raw=f"{given} {surname}"
 return raw.title()
def to_num(v):
 v=clean(v)
 if v in {"","-","–","—"}: return pd.NA
 try:return int(v.replace(",",""))
 except:return pd.NA
def parse_team_cell(raw):
 out=[]
 for token in [clean(x) for x in clean(raw).split(",") if clean(x)]:
  m=re.fullmatch(r"([A-Za-z]{2,4})\s*-\s*(\d+)",token)
  if not m: out.append(("",None,token)); continue
  code=m.group(1).upper(); out.append((TEAM_CODE_MAP.get(code,""),int(m.group(2)),token))
 return out
def primary_position(raw):
 best=("",-1)
 for token in [clean(x) for x in clean(raw).split(",") if clean(x)]:
  m=re.fullmatch(r"([A-Za-z0-9/]+)\s*-\s*(\d+)",token)
  if m and int(m.group(2))>best[1]: best=(POSITION_MAP.get(m.group(1).upper(),m.group(1).upper()),int(m.group(2)))
 return best[0]
class HardFetchTimeout(RuntimeError): pass
def _alarm(signum,frame): raise HardFetchTimeout(f"RLP fetch exceeded hard {HARD_FETCH_SECONDS}s limit")
def fetch_html():
 log(f"Fetching ONE page: {URL}")
 old=None; alarm=hasattr(signal,"SIGALRM")
 try:
  if alarm: old=signal.signal(signal.SIGALRM,_alarm); signal.alarm(HARD_FETCH_SECONDS)
  r=requests.get(URL,headers={"User-Agent":USER_AGENT,"Accept-Language":"en-AU,en;q=0.9","Connection":"close"},timeout=(CONNECT_TIMEOUT,READ_TIMEOUT))
  r.raise_for_status(); text=r.text
 finally:
  if alarm:
   signal.alarm(0)
   if old is not None: signal.signal(signal.SIGALRM,old)
 log(f"Fetch complete: HTTP {r.status_code}, {len(text):,} characters")
 if len(text)<20000 or "All Players" not in text: raise RuntimeError("Unexpected RLP response")
 return text
def parse_rows(text):
 log("Parsing HTML directly")
 soup=BeautifulSoup(text,"html.parser"); rows=[]; failures=[]; candidates=0
 for tr in soup.find_all("tr"):
  cells=tr.find_all("td")
  if len(cells)<17: continue
  a=cells[0].find("a",href=True)
  if not a or "/players/" not in clean(a.get("href","")): continue
  vals=[clean(c.get_text(" ",strip=True)) for c in cells]; candidates+=1
  player=title_player(vals[0]); raw_team=vals[2]; raw_pos=vals[3]
  for team,apps,token in parse_team_cell(raw_team):
   if not team:
    failures.append({"season":SEASON,"player":player,"raw_team":raw_team,"reason":f"unmapped_team_token:{token}"}); continue
   rows.append({"season":SEASON,"team":team,"player":player,"player_key":key(player),"primary_position":primary_position(raw_pos),"season_team_appearances":apps,"season_starts":to_num(vals[4]),"season_interchange":to_num(vals[5]),"season_total_appearances":to_num(vals[6]),"raw_rlp_team":raw_team,"raw_rlp_position":raw_pos,"rlp_player_url":clean(a.get("href","")),"roster_status":"appeared_in_nrl_season","source":"rlp_season_players","data_status":"ok"})
 log(f"Parse complete: {candidates} candidate players, {len(rows)} player-team rows")
 if candidates<300: raise RuntimeError(f"Only {candidates} player rows detected; markup may have changed")
 out=pd.DataFrame(rows).drop_duplicates(["season","team","player_key"],keep="last").sort_values(["team","player_key"]).reset_index(drop=True)
 fail=pd.DataFrame(failures,columns=["season","player","raw_team","reason"])
 return out,fail
def append_seed(base):
 if not OPTIONAL_SEED.exists(): log("No future seed found"); return base
 seed=pd.read_csv(OPTIONAL_SEED); extras=[]
 for _,r in seed.iterrows():
  player=clean(r.get("player","")); team=TEAM_ALIASES.get(key(r.get("team","")),clean(r.get("team","")))
  season=pd.to_numeric(pd.Series([r.get("season",2027)]),errors="coerce").iloc[0]; season=int(season) if pd.notna(season) else 2027
  if not player or not team or season<=SEASON: continue
  extras.append({"season":season,"team":team,"player":player,"player_key":key(player),"primary_position":clean(r.get("primary_position","")),"season_team_appearances":pd.NA,"season_starts":pd.NA,"season_interchange":pd.NA,"season_total_appearances":pd.NA,"raw_rlp_team":"","raw_rlp_position":"","rlp_player_url":"","roster_status":clean(r.get("roster_status","")) or "roster_seed","source":"nrl_roster_seed_2026_2027","data_status":"seed_only"})
 log(f"Future seed rows appended: {len(extras)}")
 if not extras:return base
 return pd.concat([base,pd.DataFrame(extras)],ignore_index=True).drop_duplicates(["season","team","player_key"],keep="first").sort_values(["season","team","player_key"]).reset_index(drop=True)
def main():
 started=time.monotonic()
 try:
  log("RLP SEASON POPULATION v3 FAST/FAIL-SAFE"); log("One web request only; no player-page loop")
  base,fail=parse_rows(fetch_html())
  found=set(base["team"]); missing=sorted(EXPECTED_2026_TEAMS-found); unexpected=sorted(found-EXPECTED_2026_TEAMS)
  log(f"Validation: {base['player_key'].nunique()} unique players, {len(found)} clubs")
  if missing: raise RuntimeError(f"Missing expected clubs: {missing}")
  if unexpected: raise RuntimeError(f"Unexpected clubs: {unexpected}")
  allrows=append_seed(base); allrows.to_csv(OUTPUT,index=False); fail.to_csv(FAILURES,index=False)
  allrows.groupby(["season","team"]).agg(players=("player_key","nunique"),source_rows=("player","size")).reset_index().to_csv(SUMMARY,index=False)
  print("\n=== 2026 PLAYERS BY CLUB ==="); print(base.groupby("team")["player_key"].nunique().sort_index().to_string())
  future=allrows[pd.to_numeric(allrows["season"],errors="coerce")>SEASON]
  if not future.empty: print("\n=== FUTURE / SEED PLAYERS ==="); print(future.groupby(["season","team"])["player_key"].nunique().to_string())
  log(f"Review/unmapped rows: {len(fail)}"); log(f"COMPLETE in {time.monotonic()-started:.1f}s"); return 0
 except Exception as e:
  log(f"FAILED after {time.monotonic()-started:.1f}s: {type(e).__name__}: {e}"); return 1
if __name__=="__main__": raise SystemExit(main())
