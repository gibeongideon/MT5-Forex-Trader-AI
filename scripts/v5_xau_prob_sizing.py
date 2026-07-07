"""XAUUSD probabilistic-sizing comparison harness — V5 Tracks 1 & 2.

Runs, on IDENTICAL folds and an identical R-based equity metric, every sizing
policy against the promoted engine's own trades and reports Sharpe + bootstrap
CI + Deflated Sharpe + PBO, plus the stress battery. Nothing here changes the
live engine — sizing multipliers are applied post-hoc to realized R-multiples
(same methodology as the rejected 2026-07-05 meta experiment), so the A/B is
apples-to-apples and research-only.

Variants:
  baseline            conf-risk only (mult = 1)                       [reference]
  vol_target          risk x ex-ante inverse-vol (Track 2)
  meta_full           risk x P(win), features = XAU + exogenous       (Track 1)
  meta_xauonly        risk x P(win), features = XAU only              [control]
  const_ctrl          constant multiplier = mean(meta_full mult)      [control]
  meta_gate           skip trades with P(win) < gate threshold        (Track 1)

Promotion (checked, not auto-applied): a variant must beat baseline on Sharpe
AND CI-low, survive spread x2 and +1 delay, and (meta_*) clear fold-mean OOS
AUC > 0.53 above the XAU-only control.

    conda run -n envmt5 python scripts/v5_xau_prob_sizing.py --max-folds 3   # smoke
    conda run -n envmt5 python scripts/v5_xau_prob_sizing.py                 # full
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
sys.path.insert(0, str(ROOT / "scripts"))

from v5_xau_meta import bar_features, rolling_prior_r  # reuse prior harness
from src.cta.bootstrap import block_bootstrap_sharpe
from src.evaluation.metrics import max_drawdown
from src.evaluation.dsr_pbo import deflated_sharpe_ratio, pbo_cscv
from src.features.vol_forecast import ewma_vol
from src.features.xau_exog import add_xau_exog_features, available_exog
from src.v5.artifacts import V5ArtifactWriter
from src.v5.prob_sizing import prob_gate, prob_to_risk_mult, vol_target_scale
from src.v5.xau_meta_oos import MetaOOSConfig, generate_meta_oos
from src.v5.xau_trend import run_trades

DATA = ROOT / "data" / "XAUUSD_H4_long.csv"
EVAL_START = pd.Timestamp("2018-01-01")
CONF_RISK_MULT = {"low": 0.5, "med": 1.0, "high": 1.5}
CONF_RISK_FRAC = {"low": 0.005, "med": 0.010, "high": 0.015}


def engine_trades(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    p = {"conf_risk_scale": CONF_RISK_MULT, **(params or {})}
    t = run_trades(df, exit_mode="trail", flip_mode="confidence", params=p)["trades"]
    t = t.copy()
    t["open_time"] = pd.to_datetime(t["open_time"])
    t["close_time"] = pd.to_datetime(t["close_time"])
    t["risk_frac"] = t["confidence"].map(CONF_RISK_FRAC)
    return t.dropna(subset=["r_multiple"]).reset_index(drop=True)


def trade_returns(trades: pd.DataFrame, mult: pd.Series) -> pd.Series:
    """Per-trade fractional return under a risk multiplier, indexed by close_time."""
    ret = trades["r_multiple"] * trades["risk_frac"] * mult
    return pd.Series(ret.values, index=trades["close_time"].values)


def daily_returns(trades: pd.DataFrame, mult: pd.Series) -> pd.Series:
    sel = trades["open_time"] >= EVAL_START
    r = trade_returns(trades[sel], mult[sel])
    eq = (1.0 + r).cumprod().groupby(level=0).last()
    return eq.resample("D").last().ffill().pct_change(fill_method=None).dropna()


def evaluate(trades: pd.DataFrame, mult: pd.Series, label: str,
             trial_sharpes_daily: list[float]) -> dict:
    sel = trades["open_time"] >= EVAL_START
    t = trades[sel]
    r = trade_returns(t, mult[sel])
    eq = (1.0 + r).cumprod().groupby(level=0).last()
    daily = eq.resample("D").last().ffill().pct_change(fill_method=None).dropna()
    sd = daily.std()
    sharpe = float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    ci = block_bootstrap_sharpe(daily.values)
    dsr = deflated_sharpe_ratio(daily.values, np.array(trial_sharpes_daily)) \
        if trial_sharpes_daily else {"dsr": float("nan"), "sr_benchmark": 0.0, "n_trials": 0}
    taken = t[mult[sel].values > 0]
    return {
        "variant": label,
        "sharpe": round(sharpe, 3),
        "sharpe_ci95": [round(ci[0], 3), round(ci[1], 3)],
        "dsr": round(dsr["dsr"], 3) if np.isfinite(dsr["dsr"]) else None,
        "total_return_pct": round((eq.iloc[-1] - 1) * 100, 1) if len(eq) else 0.0,
        "max_dd_pct": round(max_drawdown(eq), 2) if len(eq) else 0.0,
        "n_trades": int(len(t)), "n_taken": int(len(taken)),
        "win_rate_pct": round((taken["r_multiple"] > 0).mean() * 100, 1) if len(taken) else 0.0,
    }


def build_features(df: pd.DataFrame, trades: pd.DataFrame, data_dir: Path):
    """XAU-only and XAU+exog trade-level feature matrices (past-only)."""
    bars = bar_features(df)                       # prior XAU-only features (shifted)
    exog = add_xau_exog_features(df, data_dir=data_dir)
    open_t = trades["open_time"]
    xau_only = bars.loc[open_t].reset_index(drop=True)
    xau_only["prior_r5"] = rolling_prior_r(trades).values
    full = pd.concat([xau_only.reset_index(drop=True),
                      exog.loc[open_t].reset_index(drop=True)], axis=1)
    xau_only.index = trades.index
    full.index = trades.index
    return xau_only.fillna(0.0), full.fillna(0.0)


def meta_probs(X: pd.DataFrame, trades: pd.DataFrame, cfg: MetaOOSConfig):
    y = (trades["r_multiple"] > 0).astype(int)
    y.index = trades.index
    res = generate_meta_oos(X, y, trades["close_time"], trades["open_time"], cfg)
    return res


def vol_mult(df: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    sigma = ewma_vol(df["close"])
    target = float(sigma.loc[sigma.index >= EVAL_START].median())
    scale = vol_target_scale(sigma, target)
    m = scale.reindex(trades["open_time"]).values
    return pd.Series(np.nan_to_num(m, nan=1.0), index=trades.index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--max-folds", type=int, default=None,
                    help="Limit meta OOS folds (start_year..start_year+n) for smoke runs")
    ap.add_argument("--gate", type=float, default=0.5, help="meta_gate P(win) threshold")
    ap.add_argument("--run-id", default="xau-prob-sizing")
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    trades = engine_trades(df)
    print(f"engine trades: {len(trades)} "
          f"({trades['open_time'].min().date()} -> {trades['close_time'].max().date()})")
    print(f"exogenous proxy legs loaded: {available_exog(args.data_dir)}")

    ones = pd.Series(1.0, index=trades.index)
    xau_only, full = build_features(df, trades, Path(args.data_dir))

    cfg = MetaOOSConfig(max_folds=args.max_folds)
    res_full = meta_probs(full, trades, cfg)
    res_ctrl = meta_probs(xau_only, trades, cfg)
    print(f"OOS AUC  meta_full={res_full.mean_auc:.3f} "
          f"({[round(a,3) for a in res_full.fold_aucs]})")
    print(f"OOS AUC  meta_xauonly={res_ctrl.mean_auc:.3f} "
          f"({[round(a,3) for a in res_ctrl.fold_aucs]})")

    p_full, p_ctrl = res_full.probs, res_ctrl.probs
    valid = p_full.notna()
    m_meta_full = ones.where(~valid, prob_to_risk_mult(p_full))
    m_meta_ctrl = ones.where(~p_ctrl.notna(), prob_to_risk_mult(p_ctrl))
    m_vol = vol_mult(df, trades)
    m_gate = ones.where(~valid, prob_gate(p_full, args.gate))
    const = float(m_meta_full[valid].mean()) if valid.any() else 1.0
    m_const = ones.where(~valid, const)

    variants = {
        "baseline": ones, "vol_target": m_vol, "meta_full": m_meta_full,
        "meta_xauonly": m_meta_ctrl, "const_ctrl": m_const, "meta_gate": m_gate,
    }

    # trial Sharpes (non-annualized daily) for the Deflated Sharpe benchmark
    trial_daily_sr = []
    for mult in variants.values():
        d = daily_returns(trades, mult)
        sd = d.std()
        trial_daily_sr.append(float(d.mean() / sd) if sd > 0 else 0.0)

    results = [evaluate(trades, mult, name, trial_daily_sr)
               for name, mult in variants.items()]

    # PBO across variants: per-day return matrix (aligned, common dates)
    daily_map = {name: daily_returns(trades, mult) for name, mult in variants.items()}
    common = None
    for d in daily_map.values():
        common = d.index if common is None else common.intersection(d.index)
    pbo = pbo_cscv(np.column_stack([daily_map[n].reindex(common).fillna(0).values
                                    for n in variants]), n_partitions=10) \
        if common is not None and len(common) > 40 else None

    # stress battery on the meta_full and vol_target cells
    stress = {}
    for label, params in {"spread_x2": {"spread_cost_mult": 2.0},
                          "delay_2": {"entry_delay_bars": 2}}.items():
        st = engine_trades(df, params)
        _, full_s = build_features(df, st, Path(args.data_dir))
        rp = meta_probs(full_s, st, cfg).probs
        v = rp.notna()
        one_s = pd.Series(1.0, index=st.index)
        stress[label] = {
            "meta_full": evaluate(st, one_s.where(~v, prob_to_risk_mult(rp)),
                                  f"meta_full/{label}", []),
            "vol_target": evaluate(st, vol_mult(df, st), f"vol_target/{label}", []),
            "baseline": evaluate(st, one_s, f"baseline/{label}", []),
        }

    header = (f"{'variant':14s} {'Sharpe':>7s} {'CI95':>16s} {'DSR':>6s} "
              f"{'ret%':>7s} {'DD%':>6s} {'taken':>10s} {'win%':>6s}")
    print("\n" + header)
    for r in results:
        print(f"{r['variant']:14s} {r['sharpe']:+7.3f} "
              f"[{r['sharpe_ci95'][0]:+.2f},{r['sharpe_ci95'][1]:+.2f}] "
              f"{str(r['dsr']):>6s} {r['total_return_pct']:+7.1f} "
              f"{r['max_dd_pct']:6.1f} {r['n_taken']:>4d}/{r['n_trades']:<4d} "
              f"{r['win_rate_pct']:6.1f}")
    if pbo is not None:
        print(f"\nPBO (CSCV, {pbo.n_splits} splits): {pbo.pbo:.3f}")
    print("\nstress battery:")
    for lab, cells in stress.items():
        for name, s in cells.items():
            print(f"  {name:22s} Sharpe {s['sharpe']:+.3f} CI {s['sharpe_ci95']} "
                  f"ret {s['total_return_pct']:+.1f}%")

    stats = {
        "results": results,
        "oos_auc": {"meta_full": round(res_full.mean_auc, 3),
                    "meta_xauonly": round(res_ctrl.mean_auc, 3),
                    "fold_aucs_full": [round(a, 3) for a in res_full.fold_aucs]},
        "auc_uplift_vs_control": round(res_full.mean_auc - res_ctrl.mean_auc, 3)
        if np.isfinite(res_full.mean_auc) and np.isfinite(res_ctrl.mean_auc) else None,
        "const_multiplier": round(const, 3),
        "pbo": round(pbo.pbo, 3) if pbo is not None else None,
        "stress": stress,
        "exog_legs": available_exog(args.data_dir),
        "kill_criterion": "abandon meta if AUC<=0.53 AND const_ctrl matches meta_full within CI",
        "note": "identical R-based equity metric for all rows; engine untouched (research-only)",
    }
    _sel = trades["open_time"] >= EVAL_START
    eq = (1.0 + trade_returns(trades[_sel], ones[_sel])).cumprod()
    V5ArtifactWriter().write_run(
        run_id=args.run_id,
        settings={"strategy": "xau_prob_sizing", "pre_registration": "V5_PLAN.MD",
                  "meta_cfg": {"start_year": cfg.start_year, "models": list(cfg.models)}},
        trades=trades.to_dict("records"), equity=eq, stats=stats,
        reconciliation={"status": "research_replay"})
    print(f"\nrun_dir: data/v5_runs/{args.run_id}")


if __name__ == "__main__":
    main()
