#!/usr/bin/env python3
"""RLP 2026 season player population collector v4: streaming parser, fail-fast."""
from __future__ import annotations
import re, signal, time, unicodedata
from html.parser import HTMLParser
from pathlib import Path
import pandas as pd
import requests

SEASON=2026
URL=f"https://www.rugbyleagueproject.org/seasons/nrl-{SEASON}/players.html"
OUTPUT=Path("rlp_season_players.csv")
FAILURES=Path("rlp_season_players_failures.csv")
SUMMARY=Path("rlp_season_players_summary.csv")
OPTIONAL_SEED=Path("nrl_roster_seed_2026_2027.csv")
TEAM_CODE_MAP={"BRI":"Broncos","CAN":"Raiders","CBY":"Bulldogs","CRO":"Sharks","DOL":"Dolphins","GLD":"Titans","MAN":"Sea Eagles","MEL":"Storm","NEW":"Knights","NQL":"Cowboys","WAR":"Warriors","PAR":"Eels","PEN":"Panthers","SOU":"Rabbitohs","SGI":"Dragons","SYD":"Roosters","WST":"Wests Tigers"}
EXPECTED=set(TEAM_CODE_MAP.values())
POSITION_MAP={"FB":"Fullback","W":"Wing","C":"Centre","FE":"Five-eighth","HB":"Halfback","FR":"Front row","HK":"Hooker","2R":"Second row","L":"Lock","B":"Bench"}

def log(s): print(f"[{time.strftime('%H:%M:%S')}] {s}",flush=True)
def clean(v): return re.sub(r"\s+"," ",str(v or "")).strip()
def norm(v):
    s=unicodedata.normalize("NFKD",clean(v)).encode("ascii","ignore").decode().lower()
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9' -]","",s)).strip()
def pname(v):
    v=clean(v)
    if "," in v:
        a,b=[clean(x) for x in v.split(",",1)]; v=f"{b} {a}"
    return v.title()
def number(v):
    try:return int(clean(v).replace(",",""))
    except:return pd.NA
def teams(raw):
    out=[]
    for token in [clean(x) for x in clean(raw).split(",") if clean(x)]:
        m=re.fullmatch(r"([A-Za-z]{2,4})\s*-\s*(\d+)",token)
        out.append((TEAM_CODE_MAP.get(m.group(1).upper(),""),int(m.group(2)),token) if m else ("",None,token))
    return out
def primary(raw):
    best,n="",-1
    for token in clean(raw).split(","):
        m=re.fullmatch(r"\s*([A-Za-z0-9/]+)\s*-\s*(\d+)\s*",token)
        if m and int(m.group(2))>n: best,n=POSITION_MAP.get(m.group(1).upper(),m.group(1).upper()),int(m.group(2))
    return best

class Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.intr=False; self.intd=False; self.cells=[]; self.bits=[]; self.href=""; self.rows=[]
    def handle_starttag(self,tag,attrs):
        if tag=="tr": self.intr=True; self.cells=[]; self.href=""
        elif tag=="td" and self.intr: self.intd=True; self.bits=[]
        elif tag=="a" and self.intd:
            h=dict(attrs).get("href","")
            if "/players/" in h and not self.href:self.href=h
    def handle_data(self,data):
        if self.intd:self.bits.append(data)
    def handle_endtag(self,tag):
        if tag=="td" and self.intd: self.cells.append(clean(" ".join(self.bits))); self.intd=False
        elif tag=="tr" and self.intr:
            if self.href and len(self.cells)>=17:self.rows.append((self.cells[:],self.href))
            self.intr=False; self.intd=False

def timeout_handler(*_): raise TimeoutError("collector exceeded 45 second hard limit")

def main():
    start=time.monotonic()
    try:
        if hasattr(signal,"SIGALRM"): signal.signal(signal.SIGALRM,timeout_handler); signal.alarm(45)
        log("RLP SEASON POPULATION v4 STREAMING PARSER")
        log("BeautifulSoup/read_html removed; one web request only")
        r=requests.get(URL,headers={"User-Agent":"Mozilla/5.0 NRL-AI-Predictor research"},timeout=(5,15)); r.raise_for_status()
        html=r.text; log(f"Fetch complete: HTTP {r.status_code}, {len(html):,} characters")
        log("Parsing HTML with built-in HTMLParser")
        p=Parser(); p.feed(html); p.close(); log(f"Parser complete: {len(p.rows)} candidate player rows")
        if len(p.rows)<300: raise RuntimeError(f"Only {len(p.rows)} candidate rows; page structure changed")
        rows=[]; bad=[]
        for c,href in p.rows:
            name=pname(c[0]); rawteam=c[2]; rawpos=c[3]
            for team,apps,token in teams(rawteam):
                if not team: bad.append({"season":SEASON,"player":name,"raw_team":rawteam,"reason":f"unmapped_team_token:{token}"}); continue
                rows.append({"season":SEASON,"team":team,"player":name,"player_key":norm(name),"primary_position":primary(rawpos),"season_team_appearances":apps,"season_starts":number(c[4]),"season_interchange":number(c[5]),"season_total_appearances":number(c[6]),"raw_rlp_team":rawteam,"raw_rlp_position":rawpos,"rlp_player_url":href,"roster_status":"appeared_in_nrl_season","source":"rlp_season_players","data_status":"ok"})
        df=pd.DataFrame(rows).drop_duplicates(["season","team","player_key"]).sort_values(["team","player_key"])
        failures=pd.DataFrame(bad,columns=["season","player","raw_team","reason"])
        found=set(df.team); missing=sorted(EXPECTED-found); unexpected=sorted(found-EXPECTED)
        log(f"Validation: {df.player_key.nunique()} unique 2026 players, {len(found)} clubs")
        if missing: raise RuntimeError(f"Missing expected clubs: {missing}")
        if unexpected: raise RuntimeError(f"Unexpected clubs: {unexpected}")
        # Optional future seed, deliberately ignored for 2026 population.
        if OPTIONAL_SEED.exists():
            seed=pd.read_csv(OPTIONAL_SEED)
            if {"team","player"}.issubset(seed.columns):
                extras=[]
                for _,x in seed.iterrows():
                    season=pd.to_numeric(x.get("season",2027),errors="coerce"); season=int(season) if pd.notna(season) else 2027
                    if season<=SEASON or not clean(x.get("player")) or not clean(x.get("team")): continue
                    extras.append({"season":season,"team":clean(x.get("team")),"player":clean(x.get("player")),"player_key":norm(x.get("player")),"primary_position":clean(x.get("primary_position")),"season_team_appearances":pd.NA,"season_starts":pd.NA,"season_interchange":pd.NA,"season_total_appearances":pd.NA,"raw_rlp_team":"","raw_rlp_position":"","rlp_player_url":"","roster_status":"roster_seed","source":"nrl_roster_seed_2026_2027","data_status":"seed_only"})
                if extras: df=pd.concat([df,pd.DataFrame(extras)],ignore_index=True).drop_duplicates(["season","team","player_key"])
        df.to_csv(OUTPUT,index=False); failures.to_csv(FAILURES,index=False)
        df.groupby(["season","team"]).agg(players=("player_key","nunique"),source_rows=("player","size")).reset_index().to_csv(SUMMARY,index=False)
        cur=df[df.season==SEASON]
        print("\n=== 2026 PLAYERS BY CLUB ===",flush=True); print(cur.groupby("team").player_key.nunique().sort_index().to_string(),flush=True)
        log(f"Review/unmapped rows: {len(failures)}"); log(f"COMPLETE in {time.monotonic()-start:.1f}s")
        return 0
    except Exception as e:
        log(f"FAILED after {time.monotonic()-start:.1f}s: {type(e).__name__}: {e}"); return 1
    finally:
        if hasattr(signal,"SIGALRM"): signal.alarm(0)

if __name__=="__main__": raise SystemExit(main())
