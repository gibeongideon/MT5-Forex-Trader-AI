"""Trade-simulate the 70%-precision XAUUSD bottom detector (long-only).

Pipeline (honest, no lookahead in features; strict time split):
  train 60%  -> fit HistGBoost bottom model
  val   20%  -> pick the probability threshold that yields ~target precision
  test  20%  -> COUNT signals/month and SIMULATE long trades net of spread

Trade rule: on a bottom flag (proba>=thr, debounced so one open trade at a time),
enter LONG at next bar OPEN; exit by one of:
    hold{H}   fixed horizon H bars (exit at that bar close)
    atr       bracket: TP=+tp*ATR, SL=-sl*ATR (first touch; else timeout at maxbars)
Cost = full $0.34 spread paid once per round trip. Benchmark = buy&hold over test.

    python scripts/v5_xau_turning_trade.py --tf H1 --target-prec 0.70
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


def precision_at(proba, y, thr):
    m = proba >= thr
    return (y[m].mean() if m.any() else 0.0), int(m.sum())


def pick_threshold(proba, y, target):
    best = None
    for thr in np.linspace(0.30, 0.95, 66):
        p, nsig = precision_at(proba, y, thr)
        if p >= target and nsig >= 20:
            best = thr; break
    return best if best is not None else 0.9


def simulate(d_test, sig_bars, exit_mode, H, tp, sl, spread_usd, maxbars):
    """sig_bars: boolean array over test bars. Debounced long trades."""
    o = d_test.open.values; c = d_test.close.values
    h = d_test.high.values; l = d_test.low.values
    a = atr(d_test).values
    idx = d_test.index
    n = len(d_test)
    trades = []
    i = 0
    while i < n - 1:
        if not sig_bars[i]:
            i += 1; continue
        entry_i = i + 1                      # next bar open
        if entry_i >= n:
            break
        e = o[entry_i]
        exit_i, x = None, None
        if exit_mode == "hold":
            exit_i = min(entry_i + H, n - 1)
            x = c[exit_i]
        else:  # atr bracket
            tp_px = e + tp * a[entry_i]
            sl_px = e - sl * a[entry_i]
            for j in range(entry_i, min(entry_i + maxbars, n)):
                if l[j] <= sl_px:
                    exit_i, x = j, sl_px; break
                if h[j] >= tp_px:
                    exit_i, x = j, tp_px; break
            if exit_i is None:
                exit_i = min(entry_i + maxbars, n - 1); x = c[exit_i]
        ret = (x - e) / e - spread_usd / e   # long, spread paid once
        trades.append(dict(t_in=idx[entry_i], t_out=idx[exit_i], ret=ret,
                           bars=exit_i - entry_i))
        i = exit_i + 1                        # no overlap
    return pd.DataFrame(trades)


def report(tr, months, label, bh_ret):
    if len(tr) == 0:
        print(f"  {label:14} no trades"); return
    r = tr["ret"].values
    eq = np.cumprod(1 + r)
    total = eq[-1] - 1
    wr = (r > 0).mean()
    shp = r.mean() / r.std() * np.sqrt(len(r) / (months / 12)) if r.std() > 0 else 0
    peak = np.maximum.accumulate(eq); dd = (eq / peak - 1).min()
    print(f"  {label:14} trades {len(tr):4d} ({len(tr)/months:4.1f}/mo)  "
          f"win {wr*100:4.1f}%  avg {r.mean()*100:+.2f}%  total {total*100:+6.1f}%  "
          f"Sharpe {shp:4.2f}  maxDD {dd*100:5.1f}%  hold {tr['bars'].mean():4.1f}b")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--theta", type=float, default=1.5)
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--target-prec", type=float, default=0.70)
    ap.add_argument("--spread", type=float, default=0.34)
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
    p_val = clf.predict_proba(X[va_m])[:, 1]
    thr = pick_threshold(p_val, y[va_m], args.target_prec)
    pv, _ = precision_at(p_val, y[va_m], thr)

    p_te = clf.predict_proba(X[te_m])[:, 1]
    pte, nte = precision_at(p_te, y[te_m], thr)
    d_test = d.iloc[np.where(te_m)[0]]
    months = (d_test.index[-1] - d_test.index[0]).days / 30.44
    bh_ret = d_test.close.iloc[-1] / d_test.close.iloc[0] - 1

    print(f"[bottom-detector trade sim] tf={args.tf} tol=±{args.tol} theta={args.theta}*ATR spread=${args.spread}")
    print(f"  threshold {thr:.3f} (picked on val, precision {pv*100:.1f}%)  "
          f"-> TEST precision {pte*100:.1f}%  raw signal-bars {nte}")
    print(f"  TEST window: {d_test.index[0].date()} -> {d_test.index[-1].date()} "
          f"({months:.1f} months, {len(d_test)} bars)")
    print(f"  raw signal-bars/month {nte/months:.1f}  |  buy&hold gold over window {bh_ret*100:+.1f}%\n")

    # signal on full test index
    sig = np.zeros(len(d_test), dtype=bool)
    sig[p_te >= thr] = True

    print("  EXIT RULE           trades           win    avg      total    Sharpe   maxDD   hold")
    for H in (6, 12, 24, 48):
        tr = simulate(d_test, sig, "hold", H, 0, 0, args.spread, 0)
        report(tr, months, f"hold {H}b", bh_ret)
    for tp, sl in ((2, 1), (3, 1.5), (1.5, 1.5)):
        tr = simulate(d_test, sig, "atr", 0, tp, sl, args.spread, 96)
        report(tr, months, f"atr {tp}/{sl}", bh_ret)


if __name__ == "__main__":
    main()
