"""
vol_timing_probe.py — Volatility-timing probe (compression → breakout).

Volatility is the forecastable quantity (vol clustering / mean-reversion). In spot FX
you can't trade vol directly, so the testable edge is: when volatility is COMPRESSED
(low ATR percentile) and price breaks the recent range, the subsequent expansion gives
the breakout follow-through. Direction comes from the break, not a forecast.

Bug-safe by construction: the signal generator only produces a `side` series (+1/-1/NaN);
ALL P&L/exits go through the already-validated `side_barrier_meta_label` (ATR barriers,
real per-bar spread, R-units). No bespoke P&L math.

Lookahead-free: ATR (shift-safe), vol percentile over trailing window, breakout vs the
range of PRIOR bars (shift(1)). Entry at bar close; outcome on subsequent bars.

Usage:
    python scripts/vol_timing_probe.py --symbol EURUSD --tf H1 --dc
    python scripts/vol_timing_probe.py --symbol GBPUSD --tf H4 --mode revert --dc
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features.indicators import atr as raw_atr
from src.features.meta_labels import side_barrier_meta_label

PIP = {"EURUSD": 1e-4, "USDJPY": 1e-2, "GBPUSD": 1e-4, "XAUUSD": 1e-1}
COMM_PIPS = 0.5
TP_MULT = SL_MULT = 1.5
HORIZON = 16


def _load(sym, tf):
    d = pd.read_csv(ROOT / "data" / f"{sym}_{tf}_long.csv", index_col=0, parse_dates=True)
    d.columns = [c.lower() for c in d.columns]
    return d


def _signal(d, atr, vol_win, comp_pct, brk_win, mode):
    """side: compression-breakout (mode=breakout) or expansion-fade (mode=revert)."""
    # volatility percentile over trailing window (current atr known at t — safe)
    vol_rank = atr.rolling(vol_win).apply(lambda x: (x[-1] >= x).mean(), raw=True)
    compressed = vol_rank < comp_pct
    expanded   = vol_rank > (1 - comp_pct)
    # breakout vs the range of PRIOR brk_win bars (exclude current → shift 1)
    hi = d["high"].rolling(brk_win).max().shift(1)
    lo = d["low"].rolling(brk_win).min().shift(1)
    up = d["close"] > hi
    dn = d["close"] < lo
    side = pd.Series(np.nan, index=d.index)
    if mode == "breakout":     # trade the break, only when vol was compressed
        side[compressed & up] = 1.0
        side[compressed & dn] = -1.0
    else:                       # revert: fade the break when vol is already expanded
        side[expanded & up] = -1.0
        side[expanded & dn] = 1.0
    return side


def _sharpe_ci(R, span_days):
    R = np.asarray(R)
    if len(R) < 20:
        return float("nan"), (float("nan"), float("nan"))
    tpy = len(R) / span_days * 365.25 if span_days > 0 else len(R)
    sd = R.std(ddof=1)
    sh = float(R.mean()/sd*np.sqrt(tpy)) if sd > 1e-12 else float("nan")
    rng = np.random.default_rng(42)
    bs = [s.mean()/s.std(ddof=1)*np.sqrt(tpy) for s in
          (rng.choice(R, len(R), replace=True) for _ in range(1000)) if s.std(ddof=1) > 1e-12]
    return sh, ((np.percentile(bs,2.5), np.percentile(bs,97.5)) if bs else (float("nan"),)*2)


def run(sym, tf, vol_win, comp_pct, brk_win, mode, dfrom=None, dto=None, tag=""):
    d = _load(sym, tf)
    if dfrom: d = d[d.index >= pd.Timestamp(dfrom)]
    if dto:   d = d[d.index < pd.Timestamp(dto)]
    pip = PIP[sym]
    a = raw_atr(d, 14)
    side = _signal(d, a, vol_win, comp_pct, brk_win, mode)
    lab = side_barrier_meta_label(d["high"], d["low"], d["close"], side, a,
                                  TP_MULT, SL_MULT, HORIZON, pip)
    if len(lab) < 20:
        print(f"  {tag}{sym} {tf} {mode}: {len(lab)} trades — insufficient"); return
    spr = d["spread"].reindex(lab.index).fillna(1.0)
    net = lab["pips"].values - spr.values - COMM_PIPS
    R = net / lab["sl_pips"].values
    span = (lab.index[-1] - lab.index[0]).days
    sh, (lo, hi) = _sharpe_ci(R, span)
    wr = float((net > 0).mean())
    print(f"  {tag}{sym} {tf} {mode} (volwin={vol_win} comp={comp_pct} brk={brk_win}): "
          f"Sharpe={sh:+.3f} (95%CI [{lo:+.2f},{hi:+.2f}])  win={wr:.1%}  "
          f"trades={len(R)} (~{len(R)/max(span/365.25,1e-9):.0f}/yr)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD", choices=list(PIP))
    ap.add_argument("--tf", default="H1", choices=["H1", "H4", "M15"])
    ap.add_argument("--mode", default="breakout", choices=["breakout", "revert"])
    ap.add_argument("--vol-win", type=int, default=100)
    ap.add_argument("--comp-pct", type=float, default=0.25)
    ap.add_argument("--brk-win", type=int, default=20)
    ap.add_argument("--dc", action="store_true")
    args = ap.parse_args()
    print(f"\n=== VOL-TIMING {args.symbol} {args.tf} {args.mode} ===")
    if args.dc:
        run(args.symbol, args.tf, args.vol_win, args.comp_pct, args.brk_win, args.mode,
            dto="2022-01-01", tag="[DISC 15-21] ")
        run(args.symbol, args.tf, args.vol_win, args.comp_pct, args.brk_win, args.mode,
            dfrom="2022-01-01", tag="[CONF 22-26] ")
    else:
        run(args.symbol, args.tf, args.vol_win, args.comp_pct, args.brk_win, args.mode)


if __name__ == "__main__":
    main()
