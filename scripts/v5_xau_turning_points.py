"""XAUUSD turning-point DETECTION accuracy harness (classification, not returns).

Question: how well can we flag, in real time and without lookahead, that price is
at/near a SELL turning point (swing high) or a BUY turning point (swing low)?

Ground truth (labels, allowed to use future — it's the answer key):
    ZigZag swings from local extrema (scipy argrelextrema, order=k), then filtered
    to alternate H/L and to keep only swings whose amplitude >= theta = MULT * ATR.
    Swing HIGH -> a SELL turning point; swing LOW -> a BUY turning point.

Detectors (CAUSAL — only data up to and including bar t):
    rsi     Wilder RSI: <lo -> buy, >hi -> sell
    zscore  close z vs rolling mean (Bollinger-style): z<-Z -> buy, z>+Z -> sell
    fade    candle shape: close near bar LOW -> buy, near HIGH -> sell
    fractal causal local extreme: t is the min of last k bars -> buy (symmetric sell)
    combo   majority vote of the above (>=2 agree)

Scoring — a detection at bar t of side S counts as a TRUE POSITIVE if a ground-truth
swing of side S lies within +/- TOL bars of t (temporal tolerance). Then:
    precision = TP / (all detections of side S)         "when it fires, is it right?"
    recall    = matched swings / (all swings of side S)  "does it catch the turns?"
    F1, plus median LEAD (bars the detection precedes the true extremum; +=anticipates)
Baseline: COVERAGE = fraction of bars within +/-TOL of any true swing of that side =
the precision a RANDOM detector firing at the same rate would get. LIFT = prec/coverage.
A detector is only real if precision > coverage (lift > 1) with F1 above the random F1.

    python scripts/v5_xau_turning_points.py --tf H1
    python scripts/v5_xau_turning_points.py --tf M15 --theta 2.0 --tol 4 --order 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def load(tf: str) -> pd.DataFrame:
    d = pd.read_csv(f"data/XAUUSD_{tf}_long.csv", parse_dates=["time"])
    return d.set_index("time").sort_index()


def atr(d: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([(d.high - d.low),
                    (d.high - d.close.shift()).abs(),
                    (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ---------------------------------------------------------------- ground truth
def zigzag_swings(d: pd.DataFrame, order: int, theta: pd.Series):
    """Return DataFrame index positions + side ('sell'=high, 'buy'=low), amplitude-filtered."""
    hi = d.high.values
    lo = d.low.values
    imax = set(argrelextrema(hi, np.greater_equal, order=order)[0])
    imin = set(argrelextrema(lo, np.less_equal, order=order)[0])
    cand = sorted([(i, "H") for i in imax] + [(i, "L") for i in imin])
    # collapse duplicates (a bar flagged both) and enforce strict H/L alternation,
    # keeping the more-extreme pivot within a run of same type
    piv = []
    for i, t in cand:
        if piv and piv[-1][1] == t:
            j, _ = piv[-1]
            better = hi[i] > hi[j] if t == "H" else lo[i] < lo[j]
            if better:
                piv[-1] = (i, t)
        else:
            piv.append((i, t))
    # amplitude filter: keep pivot only if move from previous kept pivot >= theta
    kept = []
    for i, t in piv:
        p = hi[i] if t == "H" else lo[i]
        if not kept:
            kept.append((i, t, p)); continue
        j, tj, pj = kept[-1]
        if abs(p - pj) >= theta.iloc[i]:
            kept.append((i, t, p))
        else:
            # too small: replace last if this one is more extreme same dir, else skip
            if t == tj:
                if (t == "H" and p > pj) or (t == "L" and p < pj):
                    kept[-1] = (i, t, p)
    sells = np.array([i for i, t, _ in kept if t == "H"], dtype=int)
    buys = np.array([i for i, t, _ in kept if t == "L"], dtype=int)
    return sells, buys


# ---------------------------------------------------------------- detectors (causal)
def rsi(d, n=14):
    delta = d.close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def signals(d: pd.DataFrame, name: str, cfg: dict) -> pd.Series:
    """+1 = buy(bottom) detection, -1 = sell(top) detection, 0 = none. Causal."""
    s = pd.Series(0, index=d.index, dtype=int)
    if name == "rsi":
        r = rsi(d, cfg["rsi_n"])
        s[r < cfg["rsi_lo"]] = 1
        s[r > cfg["rsi_hi"]] = -1
    elif name == "zscore":
        m = d.close.rolling(cfg["z_n"]).mean()
        sd = d.close.rolling(cfg["z_n"]).std()
        z = (d.close - m) / sd
        s[z < -cfg["z_k"]] = 1
        s[z > cfg["z_k"]] = -1
    elif name == "fade":
        rng = (d.high - d.low).replace(0, np.nan)
        cp = (d.close - d.low) / rng
        s[cp < cfg["fade_lo"]] = 1
        s[cp > cfg["fade_hi"]] = -1
    elif name == "fractal":
        k = cfg["frac_k"]
        # causal: bar t is the lowest low / highest high of the trailing k+1 bars
        roll_lo = d.low.rolling(k + 1).min()
        roll_hi = d.high.rolling(k + 1).max()
        s[d.low <= roll_lo] = 1
        s[d.high >= roll_hi] = -1
    elif name == "combo":
        sub = np.zeros(len(d))
        for nm in ("rsi", "zscore", "fade", "fractal"):
            sub = sub + signals(d, nm, cfg).values
        s[sub >= 2] = 1
        s[sub <= -2] = -1
    return s


# ---------------------------------------------------------------- scoring
def coverage(truth: np.ndarray, n: int, tol: int) -> float:
    if len(truth) == 0:
        return 0.0
    covered = np.zeros(n, dtype=bool)
    for i in truth:
        covered[max(0, i - tol):min(n, i + tol + 1)] = True
    return covered.mean()


def score_side(det_pos: np.ndarray, truth: np.ndarray, n: int, tol: int):
    """precision, recall, F1, median lead (bars detection precedes true extreme)."""
    if len(det_pos) == 0 or len(truth) == 0:
        return dict(prec=0.0, rec=0.0, f1=0.0, lead=np.nan, nsig=len(det_pos))
    truth_sorted = np.sort(truth)
    # precision: each detection matched to nearest truth within tol
    tp = 0
    leads = []
    for t in det_pos:
        j = np.searchsorted(truth_sorted, t)
        best = None
        for cand in (j - 1, j):
            if 0 <= cand < len(truth_sorted):
                dd = truth_sorted[cand] - t
                if abs(dd) <= tol and (best is None or abs(dd) < abs(best)):
                    best = dd
        if best is not None:
            tp += 1
            leads.append(best)          # >0 => detection came before the extreme
    prec = tp / len(det_pos)
    # recall: each truth matched if any detection within tol
    det_sorted = np.sort(det_pos)
    matched = 0
    for tr in truth:
        j = np.searchsorted(det_sorted, tr)
        hit = any(0 <= c < len(det_sorted) and abs(det_sorted[c] - tr) <= tol
                  for c in (j - 1, j))
        matched += hit
    rec = matched / len(truth)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(prec=prec, rec=rec, f1=f1,
                lead=float(np.median(leads)) if leads else np.nan, nsig=len(det_pos))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--order", type=int, default=5, help="argrelextrema window (bars each side)")
    ap.add_argument("--theta", type=float, default=1.5, help="min swing amplitude in ATR")
    ap.add_argument("--tol", type=int, default=3, help="temporal tolerance (bars)")
    ap.add_argument("--eval-start", default=None)
    # detector params
    ap.add_argument("--rsi-n", type=int, default=14)
    ap.add_argument("--rsi-lo", type=float, default=30)
    ap.add_argument("--rsi-hi", type=float, default=70)
    ap.add_argument("--z-n", type=int, default=20)
    ap.add_argument("--z-k", type=float, default=2.0)
    ap.add_argument("--fade-lo", type=float, default=0.15)
    ap.add_argument("--fade-hi", type=float, default=0.85)
    ap.add_argument("--frac-k", type=int, default=5)
    args = ap.parse_args()

    d = load(args.tf)
    if args.eval_start:
        d = d[d.index >= pd.Timestamp(args.eval_start)]
    a = atr(d)
    theta = args.theta * a
    n = len(d)
    sells, buys = zigzag_swings(d, args.order, theta)

    print(f"[XAU turning-points] tf={args.tf} bars={n} ({(d.index[-1]-d.index[0]).days/365.25:.1f} yrs) "
          f"order={args.order} theta={args.theta}*ATR tol=±{args.tol}")
    print(f"  ground-truth swings: {len(sells)} sells(highs) + {len(buys)} buys(lows) "
          f"= 1 every {n/max(1,len(sells)+len(buys)):.0f} bars")
    cov_s, cov_b = coverage(sells, n, args.tol), coverage(buys, n, args.tol)
    print(f"  random-chance precision (coverage): sell {cov_s*100:.1f}%  buy {cov_b*100:.1f}%\n")

    cfg = dict(rsi_n=args.rsi_n, rsi_lo=args.rsi_lo, rsi_hi=args.rsi_hi,
               z_n=args.z_n, z_k=args.z_k, fade_lo=args.fade_lo, fade_hi=args.fade_hi,
               frac_k=args.frac_k)
    hdr = f"{'detector':9} {'side':4} {'signals':>8} {'prec':>7} {'recall':>7} {'F1':>6} {'lift':>6} {'lead':>6}"
    print(hdr); print("-" * len(hdr))
    for name in ("rsi", "zscore", "fade", "fractal", "combo"):
        sig = signals(d, name, cfg).values
        for side, pos_val, truth, cov in (("sell", -1, sells, cov_s), ("buy", 1, buys, cov_b)):
            det = np.where(sig == pos_val)[0]
            r = score_side(det, truth, n, args.tol)
            lift = r["prec"] / cov if cov > 0 else 0.0
            lead = f"{r['lead']:+.1f}" if not np.isnan(r["lead"]) else "  n/a"
            print(f"{name:9} {side:4} {r['nsig']:8d} {r['prec']*100:6.1f}% {r['rec']*100:6.1f}% "
                  f"{r['f1']*100:5.1f}% {lift:5.2f}x {lead:>6}")
        print()


if __name__ == "__main__":
    main()
