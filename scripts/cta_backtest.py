"""CTA daily momentum backtest — portfolio Sharpe with discover/confirm + diagnostics.

Headline = portfolio NET Sharpe. Single-instrument numbers are diagnostics only.
GO bar: confirm net Sharpe ≥ +0.5 AND bootstrap CI lower bound > 0 AND positive in
both discover sub-halves. Sharpe ≫1 ⇒ STOP and audit (cardinal rule).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.cta.universe import UNIVERSE, FX_PAIRS
from src.cta.panel import build_panels, daily_returns, pip_series, asset_classes
from src.cta.signals import tsmom, xsmom, combine, fx_carry, ewmac
from src.cta.portfolio import inv_vol_weights, vol_target, cluster_risk_weights
from src.cta.strategy import rebalance_hold as _rebalance, buffer_band as _buffer, TREND_SPEEDS
from src.cta.pnl import portfolio_pnl
from src.cta.bootstrap import block_bootstrap_sharpe

DATA = ROOT / "data"

ANN = np.sqrt(252)


def _sharpe(net: pd.Series) -> float:
    r = net.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * ANN) if sd > 1e-12 else float("nan")


def _maxdd(net: pd.Series) -> float:
    eq = (1 + net.fillna(0)).cumprod()
    return float(((eq.cummax() - eq) / eq.cummax()).max() * 100)


def _report(name, net, pos, returns, classes, gross=None):
    r = net.dropna()
    if len(r) < 60:
        print(f"  [{name}] too few days ({len(r)})"); return
    sh = _sharpe(r); lo, hi = block_bootstrap_sharpe(r.values)
    vol = r.std(ddof=1) * ANN * 100
    dd = _maxdd(r);
    g = _sharpe(gross.dropna()) if gross is not None else float("nan")
    print(f"  [{name}] netSharpe={sh:+.3f} (95%CI [{lo:+.2f},{hi:+.2f}])  "
          f"grossSharpe={g:+.3f}  vol={vol:.1f}%  maxDD={dd:.1f}%  days={len(r)}")
    # per-asset-class contribution (gross, pos held next day)
    contrib = (pos.shift(1) * returns)
    by_cls = {}
    for inst, cls in classes.items():
        if inst in contrib.columns:
            by_cls.setdefault(cls, []).append(inst)
    parts = []
    for cls, insts in sorted(by_cls.items()):
        csh = _sharpe(contrib[insts].sum(axis=1).reindex(r.index))
        parts.append(f"{cls}={csh:+.2f}")
    print(f"       class Sharpe: {'  '.join(parts)}")
    return sh, lo, hi


def run(sleeve, target_vol, with_costs, rebalance, risk, buffer_frac, instruments=None,
        trend_speeds="fast"):
    aliases = instruments if instruments else list(UNIVERSE)
    close, spread, kept = build_panels(aliases, "D1")
    print(f"  universe: {len(kept)} instruments, {close.index[0].date()} → {close.index[-1].date()}")
    returns = daily_returns(close)
    classes = asset_classes(kept)
    pips = pip_series(kept)

    # trend component: EWMAC continuous forecast (combined/momcarry use it now)
    trend = ewmac(close, speeds=TREND_SPEEDS[trend_speeds])
    mom = combine(trend, xsmom(close))
    if sleeve in ("carry", "momcarry"):
        rates = pd.read_csv(DATA / "rates_3m.csv", index_col=0, parse_dates=True)
        carry = fx_carry(close.index, rates, FX_PAIRS, kept)

    def _book(sig):
        raw = (cluster_risk_weights(sig, returns, classes, target_vol) if risk == "cluster"
               else inv_vol_weights(sig, returns, target_vol))
        return vol_target(raw, returns, target=target_vol)

    if sleeve == "tsmom":      pos = _book(tsmom(close))      # binary baseline
    elif sleeve == "ewmac":    pos = _book(trend)             # continuous trend
    elif sleeve == "xsmom":    pos = _book(xsmom(close))
    elif sleeve == "combined": pos = _book(mom)               # EWMAC + xsmom
    elif sleeve == "ml":                                       # NEW: ridge factor-combine (OOS)
        from src.cta.ml_combine import ml_forecast
        pos = _book(ml_forecast(close, returns))
    elif sleeve == "carry":    pos = _book(carry)
    else:  # momcarry: trend+xsmom+carry summed, then vol-targeted as one book
        _w = (lambda s: cluster_risk_weights(s, returns, classes, target_vol)) if risk == "cluster" \
             else (lambda s: inv_vol_weights(s, returns, target_vol))
        pos = vol_target(_w(mom) + _w(carry), returns, target=target_vol)
    pos = _rebalance(pos, rebalance)
    pos = _buffer(pos, buffer_frac)
    pnl = portfolio_pnl(pos, returns, spread if with_costs else spread * 0, pips, close)
    net, gross = pnl["net"], pnl["gross"]

    # beta to equal-weight long-everything basket (prove it's momentum, not beta)
    long_all = returns.mean(axis=1)
    beta_corr = net.corr(long_all)

    print(f"\n=== CTA {sleeve.upper()}  target_vol={target_vol:.0%}  costs={'on' if with_costs else 'OFF'} ===")
    splits = [("DISCOVER 2010-2021", "2008-01-01", "2022-01-01"),
              ("  sub 2010-2015", "2008-01-01", "2016-01-01"),
              ("  sub 2016-2021", "2016-01-01", "2022-01-01"),
              ("CONFIRM 2022-2026", "2022-01-01", "2027-01-01"),
              ("FULL", "2008-01-01", "2027-01-01")]
    for nm, a, b in splits:
        m = (net.index >= a) & (net.index < b)
        _report(nm, net[m], pos.loc[m], returns.loc[m], classes, gross[m])
    print(f"  turnover (ann, % NAV/yr): {pnl['turnover'].mean()*252*100:.0f}%   "
          f"corr-to-long-everything: {beta_corr:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleeve", default="ewmac",
                    choices=["tsmom", "ewmac", "xsmom", "combined", "ml", "carry", "momcarry"])
    ap.add_argument("--target-vol", type=float, default=0.10)
    ap.add_argument("--gross", action="store_true", help="disable costs")
    ap.add_argument("--rebalance", default="monthly", choices=["daily", "weekly", "monthly"])
    ap.add_argument("--risk", default="diag", choices=["diag", "cluster"],
                    help="diag=per-instrument inv-vol; cluster=equal risk across asset classes")
    ap.add_argument("--buffer", type=float, default=0.0,
                    help="position no-trade band as fraction of avg position (e.g. 0.1)")
    ap.add_argument("--instruments", default=None,
                    help="comma-separated alias subset (e.g. GOLD,SPX,UST10Y); default=full universe")
    ap.add_argument("--trend-speeds", default="fast", choices=list(TREND_SPEEDS),
                    help="EWMAC speed set: fast(4-speed default)/slow/slowest — slower = less turnover")
    args = ap.parse_args()
    instruments = [s.strip() for s in args.instruments.split(",")] if args.instruments else None
    print(f"  [risk={args.risk} rebalance={args.rebalance} buffer={args.buffer} speeds={args.trend_speeds}"
          f"{' instruments=' + ','.join(instruments) if instruments else ''}]")
    run(args.sleeve, args.target_vol, with_costs=not args.gross, rebalance=args.rebalance,
        risk=args.risk, buffer_frac=args.buffer, instruments=instruments, trend_speeds=args.trend_speeds)
    print("Done.")


if __name__ == "__main__":
    main()
