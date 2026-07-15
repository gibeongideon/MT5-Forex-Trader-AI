"""XAUUSD turning-point detection — push precision toward 70%.

Two explorations on top of scripts/v5_xau_turning_points.py's ground truth:

  (A) RULE CONFLUENCE sweep — how far does precision rise as we tighten the
      z-score threshold and stack confirmations (RSI + wick rejection + regime)?
      Pure precision/recall trade-off, fully interpretable, no training.

  (B) SUPERVISED ceiling — train a gradient-boosted classifier on a causal
      feature set to predict "is this bar within +/-tol of a swing" (per side),
      evaluated OUT-OF-SAMPLE (time-ordered split). Read the precision-recall
      curve: is 70% precision reachable, and at what recall?

Ground truth = same ZigZag swings (argrelextrema, alternated, amplitude>=theta*ATR).
Labels/features per SIDE (buy=bottom, sell=top). All features causal.

    python scripts/v5_xau_turning_ml.py --tf H1 --side buy
    python scripts/v5_xau_turning_ml.py --tf H1 --side sell --theta 3 --tol 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(tf):
    d = pd.read_csv(f"data/XAUUSD_{tf}_long.csv", parse_dates=["time"])
    return d.set_index("time").sort_index()


def atr(d, n=14):
    tr = pd.concat([(d.high - d.low), (d.high - d.close.shift()).abs(),
                    (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def zigzag_swings(d, order, theta):
    hi, lo = d.high.values, d.low.values
    imax = set(argrelextrema(hi, np.greater_equal, order=order)[0])
    imin = set(argrelextrema(lo, np.less_equal, order=order)[0])
    cand = sorted([(i, "H") for i in imax] + [(i, "L") for i in imin])
    piv = []
    for i, t in cand:
        if piv and piv[-1][1] == t:
            j, _ = piv[-1]
            if (hi[i] > hi[j]) if t == "H" else (lo[i] < lo[j]):
                piv[-1] = (i, t)
        else:
            piv.append((i, t))
    kept = []
    for i, t in piv:
        p = hi[i] if t == "H" else lo[i]
        if not kept:
            kept.append((i, t, p)); continue
        j, tj, pj = kept[-1]
        if abs(p - pj) >= theta.iloc[i]:
            kept.append((i, t, p))
        elif t == tj and ((t == "H" and p > pj) or (t == "L" and p < pj)):
            kept[-1] = (i, t, p)
    sells = np.array([i for i, t, _ in kept if t == "H"], int)
    buys = np.array([i for i, t, _ in kept if t == "L"], int)
    return sells, buys


def label_near(truth, n, tol):
    y = np.zeros(n, dtype=int)
    for i in truth:
        y[max(0, i - tol):min(n, i + tol + 1)] = 1
    return y


def features(d):
    """Causal feature matrix aimed at reversal/exhaustion detection."""
    c, h, l = d.close, d.high, d.low
    a = atr(d)
    rng = (h - l).replace(0, np.nan)
    f = pd.DataFrame(index=d.index)
    for w in (10, 20, 50):
        m = c.rolling(w).mean(); sd = c.rolling(w).std()
        f[f"z{w}"] = (c - m) / sd                          # stretch from mean
        f[f"pctl{w}"] = c.rolling(w).apply(lambda x: (x[-1] > x).mean(), raw=True)  # rank in window
    for n in (7, 14):
        f[f"rsi{n}"] = rsi(c, n)
    f["ema_dist"] = (c - c.ewm(span=50).mean()) / a         # ATR-normalised trend gap
    f["mom"] = (c - c.shift(5)) / a                          # 5-bar momentum (ATR units)
    f["accel"] = f["mom"] - (c.shift(5) - c.shift(10)) / a   # deceleration -> turn
    f["upwick"] = (h - np.maximum(c, d.open)) / rng          # rejection wicks
    f["lowick"] = (np.minimum(c, d.open) - l) / rng
    f["clpos"] = (c - l) / rng                               # close in bar range
    f["atr_z"] = (a - a.rolling(100).mean()) / a.rolling(100).std()   # vol regime
    # streak of consecutive same-direction closes
    up = (c.diff() > 0).astype(int); dn = (c.diff() < 0).astype(int)
    f["run_up"] = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
    f["run_dn"] = dn * (dn.groupby((dn != dn.shift()).cumsum()).cumcount() + 1)
    # RSI divergence: new price extreme without RSI extreme over 20 bars
    r14 = f["rsi14"]
    f["bull_div"] = ((l <= l.rolling(20).min()) & (r14 > r14.rolling(20).min() + 3)).astype(int)
    f["bear_div"] = ((h >= h.rolling(20).max()) & (r14 < r14.rolling(20).max() - 3)).astype(int)
    f["hour"] = d.index.hour
    return f


def rule_confluence(d, f, sells, buys, side, n, tol):
    """Interpretable precision as we stack conditions (buy side shown symmetric)."""
    truth = buys if side == "buy" else sells
    cov = label_near(truth, n, tol).mean()
    print(f"\n(A) RULE CONFLUENCE — side={side}  random precision={cov*100:.1f}%")
    print(f"{'rule':46} {'signals':>8} {'prec':>7} {'recall':>7} {'lift':>6}")
    z, rsi14 = f["z20"], f["rsi14"]
    if side == "buy":
        base = z < -2
        rules = [
            ("z20<-2", base),
            ("z20<-2.5", z < -2.5),
            ("z20<-3", z < -3),
            ("z20<-2 & rsi14<30", base & (rsi14 < 30)),
            ("z20<-2 & rsi14<25", base & (rsi14 < 25)),
            ("z20<-2 & lowick>0.4", base & (f["lowick"] > 0.4)),
            ("z20<-2 & rsi14<30 & lowick>0.4", base & (rsi14 < 30) & (f["lowick"] > 0.4)),
            ("z20<-2.5 & rsi14<25 & lowick>0.3 & atr_z>0", (z < -2.5) & (rsi14 < 25) & (f["lowick"] > 0.3) & (f["atr_z"] > 0)),
            ("bull_div & z20<-1.5", (f["bull_div"] == 1) & (z < -1.5)),
        ]
    else:
        base = z > 2
        rules = [
            ("z20>2", base),
            ("z20>2.5", z > 2.5),
            ("z20>3", z > 3),
            ("z20>2 & rsi14>70", base & (rsi14 > 70)),
            ("z20>2 & rsi14>75", base & (rsi14 > 75)),
            ("z20>2 & upwick>0.4", base & (f["upwick"] > 0.4)),
            ("z20>2 & rsi14>70 & upwick>0.4", base & (rsi14 > 70) & (f["upwick"] > 0.4)),
            ("z20>2.5 & rsi14>75 & upwick>0.3 & atr_z>0", (z > 2.5) & (rsi14 > 75) & (f["upwick"] > 0.3) & (f["atr_z"] > 0)),
            ("bear_div & z20>1.5", (f["bear_div"] == 1) & (z > 1.5)),
        ]
    ytruth = label_near(truth, n, tol)
    for name, mask in rules:
        pos = np.where(mask.fillna(False).values)[0]
        if len(pos) == 0:
            print(f"{name:46} {0:8d}     n/a"); continue
        prec = ytruth[pos].mean()
        # recall: fraction of swings with a signal within tol
        det = np.sort(pos); rec_hits = 0
        for tr in truth:
            j = np.searchsorted(det, tr)
            if any(0 <= c < len(det) and abs(det[c] - tr) <= tol for c in (j - 1, j)):
                rec_hits += 1
        rec = rec_hits / max(1, len(truth))
        print(f"{name:46} {len(pos):8d} {prec*100:6.1f}% {rec*100:6.1f}% {prec/cov:5.2f}x")


def supervised(f, truth, n, tol, side):
    y = label_near(truth, n, tol)
    X = f.values
    ok = ~np.isnan(X).any(axis=1)
    X, y, idx = X[ok], y[ok], np.where(ok)[0]
    split = int(len(X) * 0.7)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         max_depth=4, l2_regularization=1.0,
                                         validation_fraction=0.15, random_state=7)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    base = yte.mean()
    ap = average_precision_score(yte, p)
    prec, rec, thr = precision_recall_curve(yte, p)
    print(f"\n(B) SUPERVISED (HistGBoost, OUT-OF-SAMPLE last 30%) — side={side}")
    print(f"  test bars {len(yte)}   base rate (random precision) {base*100:.1f}%   PR-AUC {ap:.3f}")
    # precision achievable at recall grid
    print("  precision @ recall:   ", end="")
    for target_rec in (0.5, 0.3, 0.2, 0.1, 0.05):
        # highest precision with recall >= target
        mask = rec[:-1] >= target_rec
        pv = prec[:-1][mask].max() if mask.any() else float("nan")
        print(f"R{int(target_rec*100)}%→P{pv*100:.0f}%  ", end="")
    print()
    # can we hit 70% precision? find max recall where precision>=0.70
    for tp in (0.70, 0.65, 0.60):
        m = prec[:-1] >= tp
        rmax = rec[:-1][m].max() if m.any() else 0.0
        verdict = f"recall {rmax*100:.1f}%" if rmax > 0 else "NOT reachable"
        print(f"  precision>={int(tp*100)}%  ->  {verdict}")
    # feature importance via permutation-free proxy: use built-in not available; show top by simple corr
    imp = np.abs([np.corrcoef(np.nan_to_num(f.iloc[:split][col].values[ok[:split]]), ytr)[0, 1]
                  if False else 0 for col in f.columns])  # placeholder guard
    cols = list(f.columns)
    corrs = []
    ftr = f.values[ok][:split]
    for ci, col in enumerate(cols):
        v = ftr[:, ci]
        if np.nanstd(v) > 0:
            corrs.append((col, abs(np.corrcoef(np.nan_to_num(v), ytr)[0, 1])))
    corrs.sort(key=lambda x: -x[1])
    print("  top features (|corr| w/ label): " + ", ".join(f"{c}={v:.2f}" for c, v in corrs[:6]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--side", default="both", choices=["buy", "sell", "both"])
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--theta", type=float, default=1.5)
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--eval-start", default=None)
    args = ap.parse_args()

    d = load(args.tf)
    if args.eval_start:
        d = d[d.index >= pd.Timestamp(args.eval_start)]
    n = len(d)
    theta = args.theta * atr(d)
    sells, buys = zigzag_swings(d, args.order, theta)
    f = features(d)
    print(f"[XAU turning ML] tf={args.tf} bars={n} order={args.order} theta={args.theta}*ATR tol=±{args.tol}")
    print(f"  swings: {len(sells)} sells + {len(buys)} buys")
    sides = ["buy", "sell"] if args.side == "both" else [args.side]
    for side in sides:
        truth = buys if side == "buy" else sells
        rule_confluence(d, f, sells, buys, side, n, args.tol)
        supervised(f, truth, n, args.tol, side)


if __name__ == "__main__":
    main()
