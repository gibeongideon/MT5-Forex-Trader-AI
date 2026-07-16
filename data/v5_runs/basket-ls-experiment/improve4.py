"""Robust risk-management levers (no lookahead): portfolio vol-targeting, drawdown
scaler, and the vol-dial frontier. Goal: raise FP pass rate via drawdown control."""
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
def xau_lo():
    h4=load_h4(); D=6; close=h4["close"]; ew=ewmac_fc(close,tuple((f*D,s*D) for f,s in ((16,64),(32,128),(64,256)))); bk=breakout_fc(close,[d*D for d in (10,20,40)])
    fc=(0.5*(conc(np.maximum(ew.clip(lower=0),bk.clip(lower=0)))*0.8+0.15)+0.5*(conc(bk)*0.8+0.15)).clip(0,2)
    return net_daily(h4,fc,ann=ANN_H4,cost_px=(h4["spread_px"]/2+SLIP_USD))
def z(d,tv=0.10): sd=d.std()*np.sqrt(252); return d*(tv/sd) if sd>0 else d
def sh(d,s="2017-01-01"): x=d.loc[s:].dropna(); return float(x.mean()/x.std()*np.sqrt(252)) if x.std()>0 else 0.0
CLS_OF={**DRIFT,"XAUCHAMP":"xau"}
st={}
for sym in DRIFT:
    try: st[sym]=net_daily(load_d1(sym), champ_lo(load_d1(sym)["close"]))
    except: pass
st["XAUCHAMP"]=xau_lo()
al=pd.DataFrame(st).loc["2016-01-01":]; cls=sorted(set(CLS_OF[s] for s in al.columns))
csd={c: z((sum(z(al[m].fillna(0.0)) for m in [s for s in al.columns if CLS_OF[s]==c])/len([s for s in al.columns if CLS_OF[s]==c])).dropna()) for c in cls}
CL=pd.DataFrame(csd).dropna(); BASE=z(sum(z(CL[c]) for c in cls))

def voltarget(b, tv=0.07, hl=20):    # scale to constant trailing vol (no lookahead, shifted)
    rv=b.ewm(halflife=hl,min_periods=20).std()*np.sqrt(252)
    scale=(tv/rv).shift(1).clip(0,3).fillna(1.0); return b*scale
def ddscale(b, floor=0.5):           # de-risk as running drawdown deepens (no lookahead)
    eq=(1+b).cumprod(); dd=eq/eq.cummax()-1
    scale=(1+ dd*3).clip(lower=floor).shift(1).fillna(1.0)   # dd -5% -> scale 0.85, -10% -> 0.7
    return b*scale

variants={"BASE equal-class":BASE, "+ vol-target 7%":voltarget(BASE), "+ dd-scaler":ddscale(BASE),
          "+ voltarget + ddscale":ddscale(voltarget(BASE))}
print(f"{'variant':24} {'evalSR':>7} {'2021+':>7} {'maxDD':>7} | FP pass@7% (realistic)")
for nm,b in variants.items():
    bn=z(b.dropna(),0.07)  # renorm to 7% for fair FP comparison
    e=(1+bn.loc['2017-01-01':]).cumprod(); dd=float((e/e.cummax()-1).min()*100)
    s=fp_sim(bn.values,0.7,day_safety=1.5,p1=0.08,p2=0.05,dayloss=0.05,maxloss=0.10)
    print(f"{nm:24} {sh(b):+7.3f} {sh(b,'2021-01-01'):+7.2f} {dd:6.1f}% | pass {s['passpct']:.1f}% failDD {s['fail_dd']:.1f}% failDay {s['fail_day']:.1f}% med {s['med_mo']:.0f}mo")
print("\n=== vol-dial frontier (BASE, pass vs speed) ===")
for k in (0.5,0.6,0.7,0.8):
    s=fp_sim(BASE.values,k,day_safety=1.5,p1=0.08,p2=0.05,dayloss=0.05,maxloss=0.10)
    print(f"  {k*10:.0f}% vol: pass {s['passpct']:.1f}%  median {s['med_mo']:.1f}mo  p75 {s['q75_mo']:.1f}mo")

print("\n=== CLEAN validation: BASE vs +vol-target, normalized once to 10% then k=0.7 ===")
for nm,b in (("BASE equal-class",BASE),("+ vol-target",voltarget(BASE)),("+ voltarget+dd",ddscale(voltarget(BASE)))):
    bn=z(b.dropna(),0.10)   # normalize ONCE to 10%
    for tag,k in (("7%vol",0.7),("6%vol",0.6)):
        s=fp_sim(bn.values,k,day_safety=1.5,p1=0.08,p2=0.05,dayloss=0.05,maxloss=0.10)
        e=(1+(bn*k).loc['2017-01-01':]).cumprod(); dd=float((e/e.cummax()-1).min()*100)
        print(f"  {nm:16} {tag}: SR {sh(b):+.2f}  pass {s['passpct']:.1f}%  failDD {s['fail_dd']:.1f}%  maxDD {dd:.1f}%  med {s['med_mo']:.1f}mo")
