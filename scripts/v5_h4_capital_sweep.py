"""V5 H4 minimal-capital sweep — discrete-lot replay across equity levels.

Pre-registered grid (declared before running; report ALL cells):
  equity     : 500, 1000, 2000, 3000, 5000, 10000 USD
  vol target : 0.10, 0.20
  specs      : 0.01 lot min/step, FX contract 100k, XAUUSD 100 oz
  acceptance : discrete net Sharpe >= 0.8 x continuous net Sharpe

    python scripts/v5_h4_capital_sweep.py --run-id h4-capital-sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cta.bootstrap import block_bootstrap_sharpe
from src.evaluation.metrics import max_drawdown
from src.v5.artifacts import V5ArtifactWriter
from src.v5.h4_cta import CONFIG, PERIODS_PER_YEAR, h4_pnl, h4_positions, load_h4_panel
from src.v5.h4_discrete import discrete_replay

EVAL_START = "2017-01-01"
EQUITY_GRID = (500.0, 1000.0, 2000.0, 3000.0, 5000.0, 10000.0)
VOL_TARGETS = (0.10, 0.20)


def _sharpe(returns) -> float:
    r = returns.dropna()
    return float(r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR)) if r.std() > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="h4-capital-sweep")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    close, spread = load_h4_panel(args.data_dir)
    rows, equity_curves = [], {}

    for tv in VOL_TARGETS:
        cfg = {**CONFIG, "target_vol": tv}
        positions = h4_positions(close, cfg)
        cont = h4_pnl(positions, close, spread).loc[EVAL_START:]
        cont_sharpe = _sharpe(cont["net"])
        cont_eq = (1 + cont["net"].fillna(0)).cumprod()
        print(f"\n== vol target {tv:.0%}: continuous Sharpe {cont_sharpe:.3f}, "
              f"maxDD {max_drawdown(cont_eq * 10000):.1f}% ==")

        pos_eval = positions.loc[EVAL_START:]
        close_eval = close.loc[EVAL_START:]
        spread_eval = spread.loc[EVAL_START:]

        for eq0 in EQUITY_GRID:
            res = discrete_replay(pos_eval, close_eval, spread_eval, eq0)
            dr = res["equity"].pct_change(fill_method=None)
            ds = _sharpe(dr)
            ret_pct = (res["equity"].iloc[-1] / eq0 - 1) * 100
            mdd = max_drawdown(res["equity"])
            daily = res["equity"].resample("D").last().pct_change(fill_method=None).dropna()
            ci_lo, ci_hi = block_bootstrap_sharpe(daily.values)
            retention = ds / cont_sharpe if cont_sharpe else float("nan")
            verdict = "VIABLE" if retention >= 0.8 else "distorted"
            zero_worst = max(res["rounded_zero_frac"].items(), key=lambda kv: kv[1])
            rows.append(dict(vol_target=tv, equity=eq0, cont_sharpe=round(cont_sharpe, 3),
                             disc_sharpe=round(ds, 3), retention=round(retention, 3),
                             return_pct=round(float(ret_pct), 2),
                             max_dd_pct=round(mdd, 2),
                             sharpe_ci=[round(ci_lo, 3), round(ci_hi, 3)],
                             lot_changes_per_year=round(res["lot_changes"] / 9.45, 1),
                             worst_zero_leg=f"{zero_worst[0]}:{zero_worst[1]:.0%}",
                             verdict=verdict))
            print(f"  ${eq0:>7,.0f}: discrete Sharpe {ds:+.3f} ({retention:.0%} of cont), "
                  f"ret {ret_pct:+.1f}%, maxDD {mdd:.1f}%, "
                  f"worst zero-leg {zero_worst[0]} {zero_worst[1]:.0%}  -> {verdict}")
            equity_curves[f"tv{int(tv*100)}_eq{int(eq0)}"] = res["equity"]

    writer = V5ArtifactWriter()
    best = [r for r in rows if r["verdict"] == "VIABLE"]
    stats = {"grid": rows,
             "min_viable": min(best, key=lambda r: (r["equity"], -r["retention"]))
             if best else None}
    curve = equity_curves.get("tv10_eq3000")
    if curve is None:
        curve = next(iter(equity_curves.values()))
    run_dir = writer.write_run(
        run_id=args.run_id,
        settings={"strategy": "h4_cta_discrete_lots", "eval_start": EVAL_START,
                  "equity_grid": list(EQUITY_GRID), "vol_targets": list(VOL_TARGETS),
                  "vol_min": 0.01, "vol_step": 0.01,
                  "acceptance": "discrete Sharpe >= 0.8x continuous"},
        trades=[], equity=curve, stats=stats,
        reconciliation={"status": "research_replay",
                        "note": "lot-quantization stress of h4-cta-v5"})
    print(f"\nrun_dir: {run_dir}")


if __name__ == "__main__":
    main()
