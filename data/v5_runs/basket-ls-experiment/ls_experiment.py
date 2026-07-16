"""Basket LS (long/short) experiment: does letting each sleeve SHORT (trend-follow
both ways) beat the current LONG-ONLY champion basket? Backtest + FundingPips sim."""
import sys, os
ROOT="/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT,"data/v5_runs/xau-longonly-champion"))
sys.path.insert(0, os.path.join(ROOT,"data/v5_runs/challenge-lab"))
import numpy as np, pandas as pd
from xau_lab import ewmac_fc, breakout_fc, load_h4, ANN_H4, SLIP_USD
from challenge_lab import fp_sim

DRIFT={"SPX":"eq_us","NDX":"eq_us","DJI":"eq_us","DAX":"eq_eu","FTSE":"eq_eu","STOXX":"eq_eu",
       "NIKKEI":"eq_ap","ASX":"eq_ap","BTC":"crypto","ETH":"crypto","SILVER":"metal"}  # tradeable (no rates/energy/copper)
def load_d1(sym):
    df=pd.read_csv(f"{ROOT}/data/{sym}_D1_long.csv",parse_dates=["time"],index_col="time").sort_index()
    df=df[~df.index.duplicated(keep="last")]; df["spread_px"]=df["spread"].clip(lower=df["spread"].median()); return df
def norm(s): return s*(1.0/s.abs().expanding(min_periods=120).mean().shift(1))
def conc(s,p=1.5): return norm(s.clip(lower=0.0)**p)

def champ_lo(close):   # LONG-ONLY champion (current basket)
    ew=ewmac_fc(close,((16,64),(32,128),(64,256))); bk=breakout_fc(close,(10,20,40))
    return (0.5*(conc(np.maximum(ew.clip(lower=0),bk.clip(lower=0)))*0.8+0.15)+0.5*(conc(bk)*0.8+0.15)).clip(0,2)
def trend_ls(close):   # LONG/SHORT trend (mirror of champ, sign preserved)
    ew=ewmac_fc(close,((16,64),(32,128),(64,256))); bk=breakout_fc(close,(10,20,40))
    L=lambda s: (conc(np.maximum(s.clip(lower=0),0))*0.8)     # long leg strength
    S=lambda s: (conc(np.maximum((-s).clip(lower=0),0))*0.8)  # short leg strength
    long = 0.5*(conc(np.maximum(ew.clip(lower=0),bk.clip(lower=0)))) + 0.5*conc(bk)
    short= 0.5*(conc(np.maximum((-ew).clip(lower=0),(-bk).clip(lower=0)))) + 0.5*conc(-bk)
    return ((long-short)*0.8).clip(-2,2)

def net_daily(df,fc,ann=252,cost_px=None,tvol=0.10):
    close=df["close"]; ret=close.pct_change()
    vol=ret.ewm(halflife=42,min_periods=20).std()*np.sqrt(ann)
    pos=(fc*(tvol/vol)).clip(-8,8)
    band=0.1*(tvol/vol).clip(0,8); p,out,held=pos.values,np.zeros(len(pos)),0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b=band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
            if abs(p[i]-held)>b: held=p[i]-np.sign(p[i]-held)*b
        out[i]=held
    pos=pd.Series(out,index=pos.index).shift(1).fillna(0.0)
    cp = cost_px if cost_px is not None else df["spread_px"]
    net=(pos*ret-pos.diff().abs().fillna(0)*(cp/close)).fillna(0.0).resample("D").sum()
    return net[net.index.dayofweek<5]

def xau_champ(kind):   # XAU H4 champion (LO) or LS
    h4=load_h4(); D=6; ann=ANN_H4; close=h4["close"]
    ew=ewmac_fc(close,tuple((f*D,s*D) for f,s in ((16,64),(32,128),(64,256)))); bk=breakout_fc(close,[d*D for d in (10,20,40)])
    if kind=="lo":
        fc=(0.5*(conc(np.maximum(ew.clip(lower=0),bk.clip(lower=0)))*0.8+0.15)+0.5*(conc(bk)*0.8+0.15)).clip(0,2)
    else:
        long=0.5*conc(np.maximum(ew.clip(lower=0),bk.clip(lower=0)))+0.5*conc(bk)
        short=0.5*conc(np.maximum((-ew).clip(lower=0),(-bk).clip(lower=0)))+0.5*conc(-bk)
        fc=((long-short)*0.8).clip(-2,2)
    return net_daily(h4,fc,ann=ann,cost_px=(h4["spread_px"]/2+SLIP_USD))

def z(d,tv=0.10):
    sd=d.std()*np.sqrt(252); return d*(tv/sd) if sd>0 else d
def sh(d,s="2017-01-01"):
    x=d.loc[s:].dropna(); return float(x.mean()/x.std()*np.sqrt(252)) if x.std()>0 else 0.0

# build per-asset LO and LS streams
def build_streams(kind):
    st={}
    for sym in DRIFT:
        try: df=load_d1(sym)
        except FileNotFoundError: continue
        fc=champ_lo(df["close"]) if kind=="lo" else trend_ls(df["close"])
        st[sym]=net_daily(df,fc)
    st["XAUCHAMP"]=xau_champ(kind)
    return st
CLS_OF={**{k:v for k,v in DRIFT.items()},"XAUCHAMP":"xau"}
def basket(streams):
    al=pd.DataFrame(streams).loc["2016-01-01":]
    classes=sorted(set(CLS_OF[s] for s in al.columns))
    cs={}
    for c in classes:
        mem=[s for s in al.columns if CLS_OF[s]==c]
        cs[c]=z((sum(z(al[m].fillna(0.0)) for m in mem)/len(mem)).dropna())
    cl=pd.DataFrame(cs).dropna(); return z(sum(z(cl[c]) for c in cl.columns)/len(cl.columns))

print("building LO and LS streams...")
lo=build_streams("lo"); ls=build_streams("ls")
LO=basket(lo); LS=basket(ls)
BL=z((z(LO)+z(LS))/2)   # 50/50 blend of the two baskets

print(f"\n{'variant':10} {'evalSR':>7} {'2021+':>7} {'2016-20':>8} {'maxDD':>7} {'yearly (eval)'}")
for nm,b in (("LONG-ONLY",LO),("LONG/SHORT",LS),("50/50 blend",BL)):
    e=(1+b.loc['2017-01-01':]).cumprod(); dd=float((e/e.cummax()-1).min()*100)
    yr=(b.loc['2017-01-01':].groupby(b.loc['2017-01-01':].index.year).apply(lambda x:x.sum()*100))
    ys=" ".join(f"{y%100:02d}:{v:+.0f}" for y,v in yr.items())
    print(f"{nm:10} {sh(b):+7.3f} {sh(b,'2021-01-01'):+7.2f} {sh(b.loc[:'2020-12-31']):+8.2f} {dd:6.1f}%  {ys}")

print("\n=== FundingPips 2-Step Standard pass @7% vol (realistic day_safety=1.5) ===")
for nm,b in (("LONG-ONLY",LO),("LONG/SHORT",LS),("50/50 blend",BL)):
    s=fp_sim(b.values,0.7,day_safety=1.5,p1=0.08,p2=0.05,dayloss=0.05,maxloss=0.10)
    print(f"  {nm:11} pass {s['passpct']:.1f}%  failDay {s['fail_day']:.1f}%  failDD {s['fail_dd']:.1f}%  median {s['med_mo']:.1f}mo")
