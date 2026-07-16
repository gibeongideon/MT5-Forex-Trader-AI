"""Improvement directions for the LO basket (goal: raise FundingPips pass rate,
whose only weak spot is drawdown-failures). Test crash-hedge, sleeve pruning, weighting."""
import sys, os
ROOT="/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,"data/v5_runs/xau-longonly-champion")); sys.path.insert(0,os.path.join(ROOT,"data/v5_runs/challenge-lab"))
import numpy as np, pandas as pd
from xau_lab import ewmac_fc, breakout_fc, load_h4, ANN_H4, SLIP_USD
from challenge_lab import fp_sim
DRIFT={"SPX":"eq_us","NDX":"eq_us","DJI":"eq_us","DAX":"eq_eu","FTSE":"eq_eu","STOXX":"eq_eu","NIKKEI":"eq_ap","ASX":"eq_ap","BTC":"crypto","ETH":"crypto","SILVER":"metal"}
def load_d1(s):
    df=pd.read_csv(f"{ROOT}/data/{s}_D1_long.csv",parse_dates=["time"],index_col="time").sort_index(); df=df[~df.index.duplicated(keep='last')]; df["spread_px"]=df["spread"].clip(lower=df["spread"].median()); return df
def norm(s): return s*(1.0/s.abs().expanding(min_periods=120).mean().shift(1))
def conc(s,p=1.5): return norm(s.clip(lower=0.0)**p)
def champ_lo(close):
    ew=ewmac_fc(close,((16,64),(32,128),(64,256))); bk=breakout_fc(close,(10,20,40))
    return (0.5*(conc(np.maximum(ew.clip(lower=0),bk.clip(lower=0)))*0.8+0.15)+0.5*(conc(bk)*0.8+0.15)).clip(0,2)
def champ_crashhedge(close, thr=1.5, beta=0.6):   # LO + short only when downtrend is DEEP
    lo=champ_lo(close); ew=ewmac_fc(close,((16,64),(32,128),(64,256)))
    deep_short=(-ew-thr).clip(lower=0)   # activates only when ew < -thr
    return (lo - beta*conc(deep_short)*0.8).clip(-2,2)
def champ_fast(close):
    ew=ewmac_fc(close,((8,32),(16,64),(32,128))); bk=breakout_fc(close,(5,10,20))
    return (0.5*(conc(np.maximum(ew.clip(lower=0),bk.clip(lower=0)))*0.8+0.15)+0.5*(conc(bk)*0.8+0.15)).clip(0,2)
def net_daily(df,fc,ann=252,cost_px=None,tv=0.10):
    close=df["close"]; ret=close.pct_change(); vol=ret.ewm(halflife=42,min_periods=20).std()*np.sqrt(ann)
    pos=(fc*(tv/vol)).clip(-8,8); band=0.1*(tv/vol).clip(0,8); p,out,held=pos.values,np.zeros(len(pos)),0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b=band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
            if abs(p[i]-held)>b: held=p[i]-np.sign(p[i]-held)*b
        out[i]=held
    pos=pd.Series(out,index=pos.index).shift(1).fillna(0.0); cp=cost_px if cost_px is not None else df["spread_px"]
    return (pos*ret-pos.diff().abs().fillna(0)*(cp/close)).fillna(0).resample("D").sum().pipe(lambda d:d[d.index.dayofweek<5])
def xau(kind):
    h4=load_h4(); D=6; close=h4["close"]
    if kind=="fast": ew=ewmac_fc(close,tuple((f*D//2,s*D//2) for f,s in ((8,32),(16,64),(32,128)))); bk=breakout_fc(close,[d*D//2 for d in (5,10,20)])
    else: ew=ewmac_fc(close,tuple((f*D,s*D) for f,s in ((16,64),(32,128),(64,256)))); bk=breakout_fc(close,[d*D for d in (10,20,40)])
    lo=(0.5*(conc(np.maximum(ew.clip(lower=0),bk.clip(lower=0)))*0.8+0.15)+0.5*(conc(bk)*0.8+0.15)).clip(0,2)
    if kind=="crash": lo=(lo-0.6*conc((-ew-1.5).clip(lower=0))*0.8).clip(-2,2)
    return net_daily(h4,lo,ann=ANN_H4,cost_px=(h4["spread_px"]/2+SLIP_USD))
def z(d,tv=0.10): sd=d.std()*np.sqrt(252); return d*(tv/sd) if sd>0 else d
def sh(d,s="2017-01-01"): x=d.loc[s:].dropna(); return float(x.mean()/x.std()*np.sqrt(252)) if x.std()>0 else 0.0
CLS_OF={**DRIFT,"XAUCHAMP":"xau"}
def streams(fn, xkind):
    st={}
    for sym in DRIFT:
        try: df=load_d1(sym)
        except: continue
        st[sym]=net_daily(df, fn(df["close"]))
    st["XAUCHAMP"]=xau(xkind); return st
def basket(st, classes=None, weights=None):
    al=pd.DataFrame(st).loc["2016-01-01":]; cls=classes or sorted(set(CLS_OF[s] for s in al.columns))
    cs={}
    for c in cls:
        mem=[s for s in al.columns if CLS_OF[s]==c]
        if mem: cs[c]=z((sum(z(al[m].fillna(0.0)) for m in mem)/len(mem)).dropna())
    cl=pd.DataFrame(cs).dropna()
    w=weights or {c:1 for c in cl.columns}
    return z(sum(w.get(c,1)*z(cl[c]) for c in cl.columns))

base=streams(champ_lo,"lo")
variants={
 "LO baseline":       basket(base),
 "crash-hedge":       basket(streams(champ_crashhedge,"crash")),
 "faster speeds":     basket(streams(champ_fast,"fast")),
 "drop eq_eu+eq_ap":  basket(base, classes=["eq_us","crypto","xau","metal"]),
 "sharpe-weighted":   None,
}
# sharpe-weighted: weight each class by its own eval Sharpe (down-weight weak eq_eu/eq_ap)
al=pd.DataFrame(base).loc["2016-01-01":]; cls=sorted(set(CLS_OF[s] for s in al.columns)); csd={}
for c in cls:
    mem=[s for s in al.columns if CLS_OF[s]==c]; csd[c]=z((sum(z(al[m].fillna(0.0)) for m in mem)/len(mem)).dropna())
w={c:max(0.2,sh(csd[c])) for c in cls}
variants["sharpe-weighted"]=basket(base, weights=w)

print(f"{'variant':18} {'evalSR':>7} {'2021+':>7} {'maxDD':>7} | FP pass@7% (realistic)")
for nm,b in variants.items():
    e=(1+b.loc['2017-01-01':]).cumprod(); dd=float((e/e.cummax()-1).min()*100)
    s=fp_sim(b.values,0.7,day_safety=1.5,p1=0.08,p2=0.05,dayloss=0.05,maxloss=0.10)
    print(f"{nm:18} {sh(b):+7.3f} {sh(b,'2021-01-01'):+7.2f} {dd:6.1f}% | pass {s['passpct']:.1f}% failDD {s['fail_dd']:.1f}% med {s['med_mo']:.0f}mo")
