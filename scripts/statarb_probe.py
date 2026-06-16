"""
statarb_probe.py — Cross-pair stat-arb probe (relative-value mean-reversion).

Directional prediction is exhausted (M15/H1/H4 all dead or overfit). This tests a
fundamentally different edge: EURUSD and GBPUSD are highly correlated (both USD-quote
European majors); their spread mean-reverts. We trade the z-score of the spread.

Leak-free: hedge ratio (rolling OLS beta), spread mean and std are all computed on a
TRAILING window only (shift(1)), so every signal uses only past information. Costs =
real per-bar spread on BOTH legs + commission.

Spread:   s_t = log(A_t) - beta_t * log(B_t),  beta_t = rolling OLS(logA~logB) on past WIN bars
z_t   :   (s_t - mean(s, past Z)) / std(s, past Z)
Entry :   z > +ENTRY  → short spread (sell A, buy B);  z < -ENTRY → long spread
Exit  :   |z| < EXIT  (mean reverted)  or opposite signal
PnL   :   leg returns from entry→exit minus 2-leg spread+commission costs, in spread units

Usage:
    python scripts/statarb_probe.py --a EURUSD --b GBPUSD --tf H1
    python scripts/statarb_probe.py --a EURUSD --b GBPUSD --tf H4 --entry 2.0 --exit 0.5
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PIP = {"EURUSD": 1e-4, "USDJPY": 1e-2, "GBPUSD": 1e-4, "XAUUSD": 1e-1}
COMM_PIPS = 0.5


def _load(sym, tf):
    f = ROOT / "data" / f"{sym}_{tf}_long.csv"
    d = pd.read_csv(f, index_col=0, parse_dates=True)
    d.columns = [c.lower() for c in d.columns]
    return d


def _rolling_beta(la, lb, win):
    """Trailing OLS slope of la on lb over `win` bars (lookahead-free)."""
    cov = la.rolling(win).cov(lb)
    var = lb.rolling(win).var()
    return (cov / var)


def run(a, b, tf, win, zwin, entry, exit_z, ann, dfrom=None, dto=None, tag=""):
    A, B = _load(a, tf), _load(b, tf)
    idx = A.index.intersection(B.index)
    if dfrom: idx = idx[idx >= pd.Timestamp(dfrom)]
    if dto:   idx = idx[idx < pd.Timestamp(dto)]
    A, B = A.loc[idx], B.loc[idx]
    la, lb = np.log(A["close"]), np.log(B["close"])

    beta = _rolling_beta(la, lb, win).shift(1)          # past-only hedge ratio
    spread = la - beta * lb
    mu = spread.rolling(zwin).mean().shift(1)
    sd = spread.rolling(zwin).std().shift(1)
    z = (spread - mu) / sd                                # uses spread_t vs PAST mean/std

    # costs per round-trip in spread (log) units: spread_pips*pip on each leg, ×2 (entry+exit)
    costA = (A["spread"].fillna(1.0) * PIP[a] + COMM_PIPS * PIP[a]) / A["close"]
    costB = (B["spread"].fillna(1.0) * PIP[b] + COMM_PIPS * PIP[b]) / B["close"]
    rt_cost = 2.0 * (costA + abs(beta.fillna(1.0)) * costB)   # 2 legs, both ways

    pos = 0          # +1 long spread, -1 short spread
    la_e = lb_e = beta_e = entry_cost = 0.0
    rets = []        # realized per-trade returns (net), in return units of a $1 A-leg
    times = []
    zv = z.values; lav = la.values; lbv = lb.values; bv = beta.values; cv = rt_cost.values
    for i in range(len(idx)):
        zi = zv[i]; bi = bv[i]
        if not np.isfinite(zi) or not np.isfinite(bi):
            continue
        if pos == 0:
            if zi > entry:   pos = -1
            elif zi < -entry: pos = 1
            if pos != 0:
                la_e, lb_e, beta_e, entry_cost = lav[i], lbv[i], bi, cv[i]
        else:
            if abs(zi) < exit_z or (pos == 1 and zi > entry) or (pos == -1 and zi < -entry):
                # proper 2-leg P&L with beta FIXED at entry (log-return approx):
                # long spread = +1 unit A, -beta_e units B
                leg = (lav[i] - la_e) - beta_e * (lbv[i] - lb_e)
                pnl = pos * leg - entry_cost
                rets.append(pnl); times.append(idx[i]); pos = 0

    if len(rets) < 20:
        print(f"  {a}-{b} {tf}: only {len(rets)} trades — insufficient"); return
    r = pd.Series(rets, index=pd.DatetimeIndex(times))
    span = (r.index[-1] - r.index[0]).days / 365.25
    tpy = len(r) / span if span > 0 else len(r)
    sh = float(r.mean() / r.std(ddof=1) * np.sqrt(tpy)) if r.std(ddof=1) > 1e-12 else float("nan")
    # bootstrap CI
    rng = np.random.default_rng(42); a_ = r.values
    bs = [s.mean()/s.std(ddof=1)*np.sqrt(tpy) for s in
          (rng.choice(a_, len(a_), replace=True) for _ in range(1000)) if s.std(ddof=1) > 1e-12]
    lo, hi = (np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if bs else (float("nan"),)*2
    eq = (1 + r.cumsum()*0).copy()  # spread returns are additive log units
    cum = r.cumsum(); dd = float((cum.cummax() - cum).max())
    wr = float((r > 0).mean())
    print(f"  {tag}{a}-{b} {tf}  WIN={win} Z={zwin} entry={entry} exit={exit_z}: "
          f"Sharpe={sh:+.3f} (95%CI [{lo:+.2f},{hi:+.2f}])  win={wr:.1%}  "
          f"trades={len(r)} (~{tpy:.0f}/yr)  mean={r.mean()*1e4:+.2f}bp  maxDD={dd*1e4:.0f}bp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="EURUSD"); ap.add_argument("--b", default="GBPUSD")
    ap.add_argument("--tf", default="H1", choices=["H1", "H4", "M15"])
    ap.add_argument("--win", type=int, default=60)     # hedge-ratio lookback
    ap.add_argument("--zwin", type=int, default=100)   # z-score lookback
    ap.add_argument("--entry", type=float, default=2.0)
    ap.add_argument("--exit", type=float, default=0.5)
    ap.add_argument("--dc", action="store_true", help="discover/confirm split at 2022-01-01")
    args = ap.parse_args()
    print(f"\n=== STAT-ARB PROBE {args.a}-{args.b} {args.tf} ===")
    ann = {"H1": 24*252, "H4": 6*252, "M15": 96*252}[args.tf]
    if args.dc:
        run(args.a, args.b, args.tf, args.win, args.zwin, args.entry, args.exit, ann,
            dto="2022-01-01", tag="[DISCOVER 15-21] ")
        run(args.a, args.b, args.tf, args.win, args.zwin, args.entry, args.exit, ann,
            dfrom="2022-01-01", tag="[CONFIRM 22-26]  ")
    else:
        run(args.a, args.b, args.tf, args.win, args.zwin, args.entry, args.exit, ann)


if __name__ == "__main__":
    main()
