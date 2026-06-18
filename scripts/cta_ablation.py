"""cta_ablation.py — marginal contribution of each CTA building block.

Holds the locked champion config fixed and toggles ONE block per cell, on the 5-symbol basket
and/or the full universe, then prints a summary table (FULL/CONFIRM net Sharpe + CI + maxDD +
vol + turnover + beta). Reuses scripts/cta_backtest.run() so every number matches the backtester.

  Block 1  trend representation : tsmom vs ewmac(fast) vs ewmac(slow)
  Block 2  cross-sectional mom  : ewmac-only vs xsmom-only vs combined   (FULL only — needs breadth)
  Block 3  volatility targeting : dynamic vs OFF (watch realized vol + maxDD)
  Block 4  regime filtering     : none vs trend vs vol vs trend_vol
  Block 5  risk budgeting       : equal vs diag vs cluster               (FULL — clustering needs breadth)

Usage:
    python scripts/cta_ablation.py                 # full matrix
    python scripts/cta_ablation.py --only B5        # one block
    python scripts/cta_ablation.py --csv data/cta_ablation.csv
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.cta_backtest import run
from src.cta.strategy import BASKET

B = ",".join(BASKET)
# locked champion defaults; each cell overrides one knob
DEF = dict(sleeve="combined", target_vol=0.10, with_costs=True, rebalance="monthly",
           risk="cluster", buffer_frac=0.4, instruments=BASKET, trend_speeds="slow",
           regime="none", vol_target_mode="dynamic")

# (block, label, overrides)
CELLS = [
    # Block 1 — time-series momentum representation
    ("B1", "basket tsmom",        dict(sleeve="tsmom")),
    ("B1", "basket ewmac-fast",   dict(sleeve="ewmac", trend_speeds="fast")),
    ("B1", "basket ewmac-slow",   dict(sleeve="ewmac", trend_speeds="slow")),
    ("B1", "full   tsmom",        dict(sleeve="tsmom", instruments=None)),
    ("B1", "full   ewmac-fast",   dict(sleeve="ewmac", trend_speeds="fast", instruments=None)),
    ("B1", "full   ewmac-slow",   dict(sleeve="ewmac", trend_speeds="slow", instruments=None)),
    # Block 2 — cross-sectional momentum marginal (FULL universe; xsmom needs breadth)
    ("B2", "full   ewmac-only",   dict(sleeve="ewmac", instruments=None)),
    ("B2", "full   xsmom-only",   dict(sleeve="xsmom", instruments=None)),
    ("B2", "full   combined",     dict(sleeve="combined", instruments=None)),
    # Block 3 — volatility targeting on/off
    ("B3", "basket vt-on",        dict(vol_target_mode="dynamic")),
    ("B3", "basket vt-off",       dict(vol_target_mode="off")),
    ("B3", "full   vt-on",        dict(instruments=None)),
    ("B3", "full   vt-off",       dict(instruments=None, vol_target_mode="off")),
    # Block 4 — regime filtering
    ("B4", "basket regime-none",  dict(regime="none")),
    ("B4", "basket regime-trend", dict(regime="trend")),
    ("B4", "basket regime-vol",   dict(regime="vol")),
    ("B4", "basket regime-trvol", dict(regime="trend_vol")),
    ("B4", "full   regime-none",  dict(regime="none", instruments=None)),
    ("B4", "full   regime-trend", dict(regime="trend", instruments=None)),
    ("B4", "full   regime-vol",   dict(regime="vol", instruments=None)),
    ("B4", "full   regime-trvol", dict(regime="trend_vol", instruments=None)),
    # Block 5 — risk budgeting (FULL universe; clustering needs breadth)
    ("B5", "full   equal",        dict(risk="equal", instruments=None)),
    ("B5", "full   diag",         dict(risk="diag", instruments=None)),
    ("B5", "full   cluster",      dict(risk="cluster", instruments=None)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run only cells whose block/label contains this")
    ap.add_argument("--csv", default=None, help="save the summary table to this path")
    args = ap.parse_args()

    cells = [c for c in CELLS if not args.only or args.only.lower() in f"{c[0]} {c[1]}".lower()]
    rows = []
    for block, label, ov in cells:
        kw = {**DEF, **ov}
        print(f"\n{'#'*80}\n# {block}  {label}\n{'#'*80}")
        r = run(**kw)
        f = lambda v: round(v, 3) if isinstance(v, float) else v
        rows.append({
            "block": block, "cell": label, "n": r["n_instruments"],
            "FULL_Sh": f(r["full_sharpe"]), "FULL_CI": f"[{r['full_lo']:+.2f},{r['full_hi']:+.2f}]",
            "FULL_DD%": f(r["full_dd"]), "vol%": f(r["full_vol"]),
            "CONF_Sh": f(r["confirm_sharpe"]), "turn%/yr": round(r["turnover"]), "beta": f(r["beta"]),
        })
    df = pd.DataFrame(rows)
    print(f"\n{'='*100}\n  CTA BUILDING-BLOCK ABLATION — marginal contribution summary\n{'='*100}")
    print(df.to_string(index=False))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nsaved → {args.csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()
