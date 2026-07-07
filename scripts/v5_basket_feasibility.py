"""Small-account feasibility study for the promoted V5 continuous basket.

The promoted champion (basket10-btc, expected Sharpe 0.935) needs ~$66.7k
min-viable equity — infeasible on the small demo. This searches curated
sub-baskets and vol targets for the best VALIDATED Sharpe that also FITS a given
account, and emits a tiered recommendation.

Two facts drive the design:
  * Diversification comes from asset CLASSES, not instrument count (V5_PLAN
    "Universe-Widening Ablation": basket5/5-classes beats full48/8-classes).
    So candidates pick representatives across the six classes
    {METAL, RATES, EQ_INDEX, ENERGY, FX_USD, CRYPTO}.
  * Sharpe is (cost-adjusted) INVARIANT to target_vol — scaling the vol target
    scales net return and its std together. So target_vol is a pure
    DEPLOYABILITY knob: raising it shrinks min_viable_equity without changing
    Sharpe, until per-instrument leverage gets imprudent. We therefore compute
    Sharpe once per basket and sweep target_vol only for min-viable / realized-
    vol / round-to-zero.

Everything reuses the validated causal path (`lever_positions` +
`mtm_pnl_price_units`) so numbers are comparable to the promoted run. Research-
only: no config/live change. Contract sizes for the CFDs are best-effort and
MUST be verified against the live terminal (symbol_info) before any deployment.

    conda run -n envmt5 python scripts/v5_basket_feasibility.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cta.bootstrap import block_bootstrap_sharpe
from src.cta.panel import asset_classes, build_panels
from src.cta.sizing import min_viable_equity, target_lots
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv
from src.v5.artifacts import V5ArtifactWriter
from src.v5.h4_cta import mtm_pnl_price_units
from src.v5.levers import ANN, lever_positions

# HFM demo specs VERIFIED live 2026-07-07 (symbol_select + symbol_info).
# quote: "cfd"/"metal" -> USD notional per lot = contract * price;
#        "fx_quote" (USD is quote ccy, e.g. EURUSD) -> contract * price;
#        "fx_base"  (USD is base ccy, e.g. USDJPY)  -> contract (price cancels).
# Note: HFM has NO 30y bond (US30.F is the Dow index); only US10YR.F for rates.
HFM_SPEC = {
    "GOLD":   dict(symbol="XAUUSD",   contract=100.0,    vmin=0.01, quote="metal"),
    "SILVER": dict(symbol="XAGUSD",   contract=1000.0,   vmin=0.01, quote="metal"),
    "UST10Y": dict(symbol="US10YR.F", contract=100.0,    vmin=1.00, quote="cfd"),
    "SPX":    dict(symbol="US500.F",  contract=1.0,      vmin=0.10, quote="cfd"),
    "DAX":    dict(symbol="GER40",    contract=1.0,      vmin=0.01, quote="cfd"),
    "WTI":    dict(symbol="USOIL",    contract=100.0,    vmin=0.10, quote="cfd"),
    "BRENT":  dict(symbol="UKOIL",    contract=100.0,    vmin=0.10, quote="cfd"),
    "EURUSD": dict(symbol="EURUSD",   contract=100000.0, vmin=0.01, quote="fx_quote"),
    "USDJPY": dict(symbol="USDJPY",   contract=100000.0, vmin=0.01, quote="fx_base"),
    "BTC":    dict(symbol="#BTCUSD",  contract=1.0,      vmin=0.01, quote="cfd"),
}

# Curated candidates, restricted to HFM-tradable instruments (UST30Y dropped —
# not offered). Diversify across classes; prefer GRANULAR representatives
# (SILVER over GOLD, SPX/DAX over rates, USDJPY over EURUSD) so legs stop
# rounding to zero on a small account. Rates (UST10Y) has vmin=1.0 so it only
# fits larger accounts — hence the *_no_rates small-account variants.
CANDIDATES: dict[str, list[str]] = {
    "full10":            ["GOLD", "SILVER", "UST10Y", "SPX", "DAX",
                          "WTI", "BRENT", "EURUSD", "USDJPY", "BTC"],
    "basket5_v4":        ["GOLD", "UST10Y", "SPX", "WTI", "EURUSD"],
    "six_class":         ["SILVER", "UST10Y", "SPX", "WTI", "USDJPY", "BTC"],
    "five_no_crypto":    ["SILVER", "UST10Y", "SPX", "WTI", "USDJPY"],
    "five_no_rates":     ["SILVER", "SPX", "WTI", "USDJPY", "BTC"],
    "four_no_rates":     ["SILVER", "SPX", "WTI", "USDJPY"],
    "four_core":         ["SILVER", "UST10Y", "SPX", "USDJPY"],
    "three_no_rates":    ["SILVER", "SPX", "USDJPY"],
    "three_classic":     ["GOLD", "UST10Y", "SPX"],
    "two_metal_eq":      ["SILVER", "SPX"],
}

LEVER_BASE = dict(speeds="slow", sleeve="combined", rebalance="monthly",
                  buffer_frac=0.4, regime="none", carry=False, ml_combine=False)


def backtest(aliases: list[str], eval_start: str, target_vol: float):
    close, spread, kept = build_panels(aliases, tf="D1")
    classes = asset_classes(kept)
    cfg = dict(LEVER_BASE, target_vol=target_vol)
    pos = lever_positions(close, kept, classes, cfg)
    pnl = mtm_pnl_price_units(pos, close, spread).loc[eval_start:]
    pos = pos.loc[eval_start:]
    return kept, close, pos, pnl


def _usd_per_lot(a: str, price: float) -> float:
    """USD notional of 1.0 lot, handling FX quote convention."""
    s = HFM_SPEC[a]
    if s["quote"] == "fx_base":        # USD is the base ccy (e.g. USDJPY)
        return s["contract"]
    return s["contract"] * price        # cfd / metal / fx_quote (USD is quote)


def specs_for(kept, close):
    """Specs whose (contract_size * price) equals the correct USD notional."""
    out = {}
    for a in kept:
        s = HFM_SPEC[a]
        price = float(close[a].dropna().iloc[-1])
        per_lot = _usd_per_lot(a, price)
        # feed sizing.target_lots an effective contract so contract*price == per_lot
        out[a] = dict(symbol=s["symbol"], contract_size=per_lot / price,
                      price=price, vol_min=s["vmin"], vol_step=s["vmin"],
                      vol_max=1e6)
    return out


def deployability(kept, close, pos, specs, tiers):
    """Use each leg's TYPICAL (median |weight|) target as the sizing unit."""
    units = {a: float(np.sign(pos[a].iloc[-1] or 1.0) * pos[a].abs().median())
             for a in kept}
    mve = min_viable_equity(units, specs)
    per_tier = {}
    for tier in tiers:
        res = target_lots(units, tier, specs)
        zeros = [a for a, r in res.items() if r["rounded_zero"]]
        gross = sum(abs(r["actual_notional"]) for r in res.values())
        per_tier[tier] = {"n_round_zero": len(zeros), "round_zero": zeros,
                          "gross_notional": round(gross, 0),
                          "leverage": round(gross / tier, 2)}
    return mve, per_tier, units


def stats(pnl, trial_daily_sr):
    net = pnl["net"].fillna(0.0)
    equity = (1.0 + net).cumprod()
    daily = equity.pct_change(fill_method=None).dropna()
    sd = net.std()
    sharpe = float(net.mean() / sd * ANN) if sd > 0 else 0.0
    ci = block_bootstrap_sharpe(daily.values)
    realized_vol = float(net.std() * ANN)
    dsr = deflated_sharpe_ratio(daily.values, np.array(trial_daily_sr)) \
        if trial_daily_sr else {"dsr": float("nan")}
    years = max((pnl.index[-1] - pnl.index[0]).days / 365.25, 1e-9)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0)
    peak = equity.cummax(); dd = float(((peak - equity) / peak).max())
    return {"sharpe": round(sharpe, 3), "ci95": [round(ci[0], 3), round(ci[1], 3)],
            "dsr": round(dsr["dsr"], 3) if np.isfinite(dsr["dsr"]) else None,
            "cagr_pct": round(cagr * 100, 2), "maxdd_pct": round(dd * 100, 1),
            "realized_vol_pct": round(realized_vol * 100, 1), "daily": daily}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-start", default="2010-01-01")
    ap.add_argument("--tiers", default="1000,10000,100000")
    ap.add_argument("--vol-sweep", default="0.10,0.20,0.30,0.40")
    ap.add_argument("--run-id", default="basket-feasibility")
    args = ap.parse_args()
    tiers = [float(x) for x in args.tiers.split(",")]
    vols = [float(x) for x in args.vol_sweep.split(",")]

    # First pass at base vol: validated stats per candidate (+ PBO across them).
    base_daily, base_stats, base_meta = {}, {}, {}
    for name, aliases in CANDIDATES.items():
        kept, close, pos, pnl = backtest(aliases, args.eval_start, vols[0])
        base_meta[name] = (kept, close, pos, pnl)
        base_daily[name] = None  # filled after we have trial SRs
    # trial daily-SRs (non-annualized) for the Deflated Sharpe benchmark
    trial_sr = []
    for name, (kept, close, pos, pnl) in base_meta.items():
        net = pnl["net"].fillna(0.0)
        eq = (1.0 + net).cumprod(); d = eq.pct_change(fill_method=None).dropna()
        base_daily[name] = d
        trial_sr.append(float(d.mean() / d.std()) if d.std() > 0 else 0.0)

    rows = []
    for name, (kept, close, pos, pnl) in base_meta.items():
        s = stats(pnl, trial_sr)
        specs = specs_for(kept, close)
        # min-viable scales ~1/target_vol; report the vol needed to fit each tier
        results_by_vol = {}
        for v in vols:
            _, _, posv, _ = backtest(CANDIDATES[name], args.eval_start, v)
            mve, per_tier, _ = deployability(kept, close, posv, specs, tiers)
            results_by_vol[v] = {"min_viable": round(mve, 0), "tiers": per_tier}
        rows.append({"basket": name, "n": len(kept), "kept": kept,
                     "sharpe": s["sharpe"], "ci95": s["ci95"], "dsr": s["dsr"],
                     "cagr_pct": s["cagr_pct"], "maxdd_pct": s["maxdd_pct"],
                     "by_vol": results_by_vol})

    # PBO across candidates (overfitting of the selection itself)
    common = None
    for d in base_daily.values():
        common = d.index if common is None else common.intersection(d.index)
    M = np.column_stack([base_daily[n].reindex(common).fillna(0).values
                         for n in CANDIDATES])
    pbo = pbo_cscv(M, n_partitions=10) if len(common) > 40 else None

    # tiered recommendation: best Sharpe whose min_viable (at best allowed vol
    # <= 0.40) fits the tier, preferring CI-low > 0.
    recs = {}
    for tier in tiers:
        best = None
        for r in rows:
            fit_vol = next((v for v in vols
                            if r["by_vol"][v]["min_viable"] <= tier), None)
            if fit_vol is None:
                continue
            key = (r["ci95"][0] > 0, r["sharpe"])
            if best is None or key > best[0]:
                best = (key, {"basket": r["basket"], "sharpe": r["sharpe"],
                              "ci95": r["ci95"], "target_vol": fit_vol,
                              "min_viable": r["by_vol"][fit_vol]["min_viable"],
                              "round_zero": r["by_vol"][fit_vol]["tiers"][tier]["round_zero"],
                              "kept": r["kept"]})
        recs[tier] = best[1] if best else None

    # ── print ────────────────────────────────────────────────────────────────
    print(f"\nValidated basket candidates (eval {args.eval_start}+, base vol {vols[0]}):")
    print(f"{'basket':20s} {'n':>2s} {'Sharpe':>7s} {'CI95':>16s} {'DSR':>6s} "
          f"{'CAGR%':>6s} {'DD%':>6s}  min-viable@vol")
    for r in sorted(rows, key=lambda x: -x["sharpe"]):
        mv = "  ".join(f"{v:.2f}:${int(r['by_vol'][v]['min_viable']):,}" for v in vols)
        print(f"{r['basket']:20s} {r['n']:>2d} {r['sharpe']:+7.3f} "
              f"[{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}] {str(r['dsr']):>6s} "
              f"{r['cagr_pct']:6.1f} {r['maxdd_pct']:6.1f}  {mv}")
    if pbo is not None:
        print(f"\nPBO across {len(CANDIDATES)} candidates (CSCV): {pbo.pbo:.3f}")
    print("\nTiered recommendation (best validated Sharpe that FITS the account):")
    for tier, rec in recs.items():
        if rec is None:
            print(f"  ${int(tier):>7,}: NONE fit even at vol {max(vols):.2f}")
        else:
            print(f"  ${int(tier):>7,}: {rec['basket']:18s} Sharpe {rec['sharpe']:+.3f} "
                  f"CI {rec['ci95']} @vol {rec['target_vol']:.2f} "
                  f"(min-viable ${int(rec['min_viable']):,}"
                  f"{', drops '+','.join(rec['round_zero']) if rec['round_zero'] else ''})")

    out = {"eval_start": args.eval_start, "tiers": tiers, "vol_sweep": vols,
           "pbo_across_candidates": round(pbo.pbo, 3) if pbo else None,
           "candidates": [{k: v for k, v in r.items() if k != "by_vol"} |
                          {"by_vol": {str(v): r["by_vol"][v] for v in vols}}
                          for r in rows],
           "recommendation_by_tier": {str(int(t)): recs[t] for t in tiers},
           "contract_sizes_are_best_effort": True,
           "note": "Sharpe invariant to target_vol; vol is a deployability knob. "
                   "VERIFY contract sizes in terminal before deploying."}
    eq_ref = (1.0 + base_meta["full11"][3]["net"].fillna(0.0)).cumprod()
    V5ArtifactWriter().write_run(
        run_id=args.run_id,
        settings={"strategy": "basket_small_account_feasibility",
                  "candidates": CANDIDATES, "lever_base": LEVER_BASE},
        trades=[], equity=eq_ref, stats=out,
        reconciliation={"status": "research_only"})
    print(f"\nrun_dir: data/v5_runs/{args.run_id}")


if __name__ == "__main__":
    main()
