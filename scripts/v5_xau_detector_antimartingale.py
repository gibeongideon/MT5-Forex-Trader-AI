"""Anti-martingale ('untimartingale') sizing on the 70%-precision bottom detector.

Anti-martingale = scale the stake UP after a WIN (ride hot streaks), reset to base
after a LOSS. Opposite of martingale. Rationale: the bottom-long trades win ~55-63%
and in a drifting market wins may cluster, so pressing winners could compound.

Pipeline (same as scripts/v5_xau_turning_trade.py, strict time split):
  train 60% -> fit HistGBoost bottom model
  val   20% -> pick proba threshold for ~target precision
  test  20% -> generate debounced long trades (one open at a time), then apply
               sizing engines to the ORDERED per-trade returns.

Engines (on per-trade net return r_i, $ equity, bust if equity<=floor):
  flat    constant base stake
  anti    mult = min(mult*step, cap) after a win; reset to 1 after a loss
  marti   (contrast) mult = min(mult*2, cap) after a loss; reset after a win

    python scripts/v5_xau_detector_antimartingale.py --exit hold48
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v5_xau_turning_ml import load, atr, zigzag_swings, features, label_near
from scripts.v5_xau_turning_trade import simulate, precision_at, pick_threshold


def size_engine(rets, engine, base, equity0, floor, step, cap):
    eq = equity0
    curve, stakes = [], []
    mult = 1.0
    busted = -1
    for i, r in enumerate(rets):
        if eq <= floor:
            busted = i; break
        stake = base * mult
        stake = min(stake, max(0.0, eq - floor))     # cannot risk past bust
        eq += stake * r
        curve.append(eq); stakes.append(mult)
        won = r > 0
        if engine == "flat":
            mult = 1.0
        elif engine == "anti":
            mult = min(mult * step, cap) if won else 1.0
        elif engine == "marti":
            mult = 1.0 if won else min(mult * 2.0, cap)
    return np.array(curve), np.array(stakes), busted


def stats(curve, stakes, rets, equity0, months, label, busted):
    if len(curve) == 0:
        print(f"  {label:16} no trades"); return
    r = np.diff(np.concatenate([[equity0], curve])) / np.concatenate([[equity0], curve[:-1]])
    total = curve[-1] / equity0 - 1
    peak = np.maximum.accumulate(curve); dd = (curve / peak - 1).min()
    sh = r.mean() / r.std() * np.sqrt(len(r) / (months / 12)) if r.std() > 0 else 0
    wr = (np.array(rets[:len(curve)]) > 0).mean()
    b = f"BUST@{busted}" if busted >= 0 else "ok"
    print(f"  {label:16} end ${curve[-1]:9,.0f}  total {total*100:+7.1f}%  win {wr*100:4.1f}%  "
          f"Sharpe {sh:5.2f}  maxDD {dd*100:6.1f}%  peakStake {stakes.max():4.0f}x  {b}")


def streak_stats(rets):
    wins = rets > 0
    runs = []
    c = 0
    for w in wins:
        if w:
            c += 1
        else:
            if c: runs.append(c)
            c = 0
    if c: runs.append(c)
    runs = np.array(runs) if runs else np.array([0])
    # autocorrelation of win/loss (does a win predict a win?)
    if len(wins) > 2:
        ac = np.corrcoef(wins[:-1].astype(float), wins[1:].astype(float))[0, 1]
    else:
        ac = np.nan
    print(f"  win/loss lag-1 autocorr {ac:+.3f}  (>0 => wins cluster, favours anti-martingale)")
    print(f"  win-streak runs: n={len(runs)} mean={runs.mean():.2f} max={runs.max()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--theta", type=float, default=1.5)
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--target-prec", type=float, default=0.70)
    ap.add_argument("--exit", default="hold48", help="hold6|hold12|hold24|hold48|atr21|atr315")
    ap.add_argument("--spread", type=float, default=0.34)
    ap.add_argument("--base", type=float, default=1000.0)
    ap.add_argument("--equity", type=float, default=10000.0)
    args = ap.parse_args()

    d = load(args.tf)
    n = len(d)
    theta = args.theta * atr(d)
    _, buys = zigzag_swings(d, args.order, theta)
    f = features(d)
    y = label_near(buys, n, args.tol)
    X = f.values
    ok = ~np.isnan(X).any(axis=1)
    i_tr, i_val = int(n * 0.6), int(n * 0.8)
    tr_m = ok.copy(); tr_m[i_tr:] = False
    va_m = ok.copy(); va_m[:i_tr] = False; va_m[i_val:] = False
    te_m = ok.copy(); te_m[:i_val] = False

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                         l2_regularization=1.0, random_state=7)
    clf.fit(X[tr_m], y[tr_m])
    thr = pick_threshold(clf.predict_proba(X[va_m])[:, 1], y[va_m], args.target_prec)
    p_te = clf.predict_proba(X[te_m])[:, 1]
    pte, _ = precision_at(p_te, y[te_m], thr)
    d_test = d.iloc[np.where(te_m)[0]]
    months = (d_test.index[-1] - d_test.index[0]).days / 30.44
    bh = d_test.close.iloc[-1] / d_test.close.iloc[0] - 1
    sig = np.zeros(len(d_test), dtype=bool); sig[p_te >= thr] = True

    ex = args.exit
    if ex.startswith("hold"):
        tr = simulate(d_test, sig, "hold", int(ex[4:]), 0, 0, args.spread, 0)
    elif ex.startswith("atr"):
        tp, sl = int(ex[3]) , int(ex[4:]) / 10 if len(ex) > 4 else 1.0
        tr = simulate(d_test, sig, "atr", 0, tp, sl, args.spread, 96)
    rets = tr["ret"].values

    print(f"[detector anti-martingale] tf={args.tf} exit={ex} thr={thr:.3f} "
          f"TEST precision {pte*100:.1f}%  window {d_test.index[0].date()}->{d_test.index[-1].date()} "
          f"({months:.1f} mo)")
    print(f"  trades {len(tr)} ({len(tr)/months:.1f}/mo)  base win {(rets>0).mean()*100:.1f}%  "
          f"buy&hold gold {bh*100:+.1f}%")
    streak_stats(rets)
    print()
    for eng, label in (("flat", "flat"),):
        c, s, b = size_engine(rets, eng, args.base, args.equity, 0.0, 0, 0)
        stats(c, s, rets, args.equity, months, label, b)
    for step in (1.5, 2.0):
        for cap in (4.0, 8.0):
            c, s, b = size_engine(rets, "anti", args.base, args.equity, 0.0, step, cap)
            stats(c, s, rets, args.equity, months, f"anti s{step} c{cap:.0f}", b)
    c, s, b = size_engine(rets, "marti", args.base, args.equity, 0.0, 0, 16.0)
    stats(c, s, rets, args.equity, months, "marti(contrast)", b)


if __name__ == "__main__":
    main()
