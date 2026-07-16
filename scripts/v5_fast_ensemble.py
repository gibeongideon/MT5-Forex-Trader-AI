"""v5_fast_ensemble.py — FAST (non-trend) diversifier sleeve to sit ALONGSIDE the
trend book. Two independent, ~uncorrelated daily signals, combined equal-risk:

  OVERNIGHT : hold each instrument close->open (captures the overnight risk premium).
              OOS Sharpe ~1.19 — BUT hinges on the close->open gap being tradeable;
              gated behind --overnight (default OFF until verified on demo).
  REVERSAL  : fade the prior day's move (short after up-day, long after down-day),
              exit next close. OOS ~0.57, cleanly tradeable close->close, ~0 corr
              to the trend book. Always on.

All net of spread. OOS = 2021+. See data/v5_runs/fast-strategies/REPORT.md.

    python scripts/v5_fast_ensemble.py --backtest              # reversal-only (safe)
    python scripts/v5_fast_ensemble.py --backtest --overnight  # + overnight (unverified)
    python scripts/v5_fast_ensemble.py --targets --overnight   # live per-instrument targets
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# instruments with a genuine session (real overnight gap) + liquid for reversal.
# NOTE: crypto EXCLUDED — 24h, no overnight gap (backtest confirmed net-negative).
INSTRUMENTS = ["SPX", "NDX", "DJI", "DAX", "NIKKEI", "GOLD", "SILVER", "NATGAS"]
REVERSAL_SET = ["SPX", "NDX", "ASX", "SILVER"]   # subset with a real 1-day reversal edge
START = "2016-01-01"


def load(sym):
    df = pd.read_csv(f"{ROOT}/data/{sym}_D1_long.csv", parse_dates=["time"],
                     index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["spread_px"] = df["spread"].clip(lower=df["spread"].median())
    return df


def overnight_stream(sym):
    """Net daily return of holding close[t-1]->open[t]. Long-only overnight premium."""
    df = load(sym)
    c = df["close"]
    return ((df["open"] - c.shift(1)) / c.shift(1) - df["spread_px"] / c).dropna()


def reversal_stream(sym):
    """Fade prior day: pos = -sign(yesterday return), held one day, net of spread."""
    df = load(sym)
    c = df["close"]
    ret = c.pct_change()
    cost = df["spread_px"] / c
    pos = -np.sign(ret).shift(1)
    return (pos * ret - pos.diff().abs().fillna(0) * cost).dropna()


def _z(d, tv=0.10):
    sd = d.std() * np.sqrt(252)
    return d * (tv / sd) if sd > 0 else d


def build(overnight=False):
    """Equal-risk fast ensemble. Reversal always in; overnight optional."""
    comps = {f"REV_{s}": reversal_stream(s) for s in REVERSAL_SET}
    if overnight:
        comps.update({f"ON_{s}": overnight_stream(s) for s in INSTRUMENTS})
    al = pd.DataFrame({k: v.loc[START:] for k, v in comps.items()})
    book = _z(sum(_z(al[c].fillna(0.0)) for c in al.columns) / len(al.columns))
    return book, comps


def _sh(d, s="2017-01-01"):
    x = d.loc[s:].dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0


def cmd_backtest(overnight):
    book, comps = build(overnight)
    for tag, a, b in (("in-sample 2017-20", "2017-01-01", "2020-12-31"),
                      ("OUT-OF-SAMPLE 21+", "2021-01-01", "2027-01-01")):
        seg = book.loc[a:b]
        eq = (1 + seg).cumprod()
        dd = float((eq / eq.cummax() - 1).min() * 100)
        print(f"  {tag}: Sharpe {seg.mean()/seg.std()*np.sqrt(252):+.2f}  maxDD {dd:.1f}%")
    yr = book.loc["2021-01-01":].groupby(book.loc["2021-01-01":].index.year).apply(_sh)
    print("  OOS yearly:", "  ".join(f"{y}:{v:+.1f}" for y, v in yr.items()))
    # correlation to the trend basket (if importable)
    try:
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        import v5_basket_challenge as E
        _, tb, _ = E.build(dial=0.7)
        j = pd.concat([book, tb], axis=1).dropna()
        corr = j.iloc[:, 0].corr(j.iloc[:, 1])
        comb = _z((_z(book) + _z(tb.loc[book.index])) / 2)
        print(f"  corr to TREND book {corr:+.2f}  |  TREND+FAST 50/50 OOS Sharpe "
              f"{_sh(comb,'2021-01-01'):+.2f} (trend alone {_sh(tb,'2021-01-01'):+.2f})")
    except Exception as exc:  # noqa
        print(f"  (trend-corr skipped: {exc})")


def cmd_targets(overnight):
    """Live signal: per-instrument direction for tomorrow. +1 long / -1 short / 0 flat."""
    print(f"{'sleeve':10} {'instrument':10} {'signal':>7}")
    for s in REVERSAL_SET:
        df = load(s)
        last = np.sign(df["close"].pct_change().iloc[-1])
        print(f"{'reversal':10} {s:10} {-last:+7.0f}   (fade yesterday's move)")
    if overnight:
        for s in INSTRUMENTS:
            print(f"{'overnight':10} {s:10} {'+1':>7}   (hold close->open; VERIFY FILLS)")
    else:
        print("  (overnight sleeve OFF — enable with --overnight after fill verification)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overnight", action="store_true",
                    help="include the overnight sleeve (unverified fills)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--backtest", action="store_true")
    g.add_argument("--targets", action="store_true")
    args = ap.parse_args()
    label = "REVERSAL + OVERNIGHT" if args.overnight else "REVERSAL-only (safe)"
    print(f"=== FAST ensemble [{label}] ===")
    if args.backtest:
        cmd_backtest(args.overnight)
    else:
        cmd_targets(args.overnight)


if __name__ == "__main__":
    main()
