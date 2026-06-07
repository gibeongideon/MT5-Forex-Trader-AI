"""
SMC / ICT Signals vs Champion Baseline — A/B/C comparison.

Tests whether porting Smart Money Concepts indicators from trader_reference
improves on the +3.13 Sharpe champion (XGBoost + enc8, M15, 39 features).

Configs
-------
  Config A — Baseline (39 features)
             31 base + 8 latent — champion +3.13 Sharpe

  Config B — + SMC Priority-1 signals (~45 features)
             + ob_bull, ob_bear              (Order Blocks)
             + fvg_bull, fvg_bear, fvg_size  (Fair Value Gaps)
             + pdh_dist, pdl_dist, pd_pos    (Previous Day Levels)

  Config C — + All SMC signals (~56 features)
             Config B signals + Andean Oscillator (4) + SuperTrend (2) + Heiken Ashi (2)

All configs:
  - XGBoost + supervised enc8, 30 epochs
  - 49k M15 EURUSD bars (2024-05-14 → 2026-05-18)
  - Expanding walk-forward 180d / 30d (19 folds)
  - Separate cache dirs to avoid cross-contamination

Key design: signals are injected via FeaturePipeline(extra_spec=...) and
registered in _OHLCV_FUNS so they receive the full DataFrame. The enc8
encoder takes raw 50-bar OHLCV windows — completely unaffected by extra_spec.

Usage
-----
  conda run -n envmt5 python scripts/compare_smc_signals.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.pipeline import PredictorPipeline, PipelineConfig
from src.features.feature_pipeline import FeaturePipeline
import src.features.feature_pipeline as _fp_module

# ── Import SMC signal functions ────────────────────────────────────────────────
from src.features.smc_signals import (
    order_blocks,
    fair_value_gaps,
    prev_day_levels,
    andean_oscillator,
    supertrend,
    heiken_ashi,
)

# Register all new functions as OHLCV-type so FeaturePipeline passes
# the full DataFrame (not just df["close"]) to each function.
_fp_module._OHLCV_FUNS |= {
    order_blocks,
    fair_value_gaps,
    prev_day_levels,
    andean_oscillator,
    supertrend,
    heiken_ashi,
}

# ── Signal specs ───────────────────────────────────────────────────────────────

# Priority-1: structurally orthogonal SMC signals
SMC_P1_SPEC = [
    (("ob_bull", "ob_bear"),                    order_blocks,    {}),
    (("fvg_bull", "fvg_bear", "fvg_size"),       fair_value_gaps, {}),
    (("pdh_dist", "pdl_dist", "pd_pos"),         prev_day_levels, {}),
]

# Priority-2+: additional indicator-based signals
SMC_P2_SPEC = [
    (("andean_bull", "andean_bear",
      "andean_signal", "andean_diff"),            andean_oscillator, {"length": 50}),
    (("supertrend_dir", "supertrend_dist"),       supertrend,        {"period": 10, "multiplier": 3.0}),
    (("ha_dir", "ha_body_norm"),                  heiken_ashi,       {}),
]

FILTER_START = pd.Timestamp("2024-05-14 11:15:00")
CHAMPION = dict(sharpe=3.13, maxdd=13.3, ret=3.583, trades=524,
                label="Champion  31+8=39 feat  no SMC")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next(c for c in df.columns if "time" in c)
    df[time_col] = pd.to_datetime(df[time_col])
    return df.set_index(time_col).sort_index()


def _load_prices(path: str) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    prices.columns = [c.lower() for c in prices.columns]
    return prices.sort_index()


def _make_base_cfg(full_cfg: dict, cache_tag: str) -> PipelineConfig:
    """Build champion-equivalent PipelineConfig with a unique cache dir."""
    cfg = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )
    cfg.model_type               = "xgboost"
    cfg.encoder_mode             = "supervised"
    cfg.encoder_latent_dim       = 8
    cfg.encoder_epochs           = 30
    cfg.candle_tokenizer_enabled = False
    cfg.wf_cache_dir             = str(ROOT / f"data/models/wf_cache_smc_{cache_tag}")
    return cfg


def _run(label: str, extra_spec: list, cfg: PipelineConfig,
         df_raw: pd.DataFrame, prices: pd.DataFrame) -> dict:
    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"  extra signals: {len(extra_spec)} spec groups")
    print(f"  Dataset: {df_raw.index[0].date()} → {df_raw.index[-1].date()}"
          f"  ({len(df_raw):,} bars)")
    print(f"{'='*68}")
    t0 = time.time()

    pipe = PredictorPipeline(cfg)

    # Inject extra signal specs into the FeaturePipeline without touching
    # the base 31-feature spec. extra_spec=[] means baseline config.
    pipe._fp = FeaturePipeline(
        label_horizon   = cfg.label_horizon,
        label_threshold = cfg.label_threshold,
        scale           = cfg.scale,
        extra_spec      = extra_spec if extra_spec else None,
    )

    X, y = pipe.build_features(df_raw)
    n_feat = X.shape[1]
    print(f"  Feature count: {n_feat}", flush=True)

    r = pipe.walk_forward(X, y, prices)

    eq      = r.equity
    ret     = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    elapsed = time.time() - t0

    print(f"\n  Sharpe : {r.sharpe:+.2f}")
    print(f"  MaxDD  : {r.drawdown:.1f}%")
    print(f"  Return : {ret:+.1%}")
    print(f"  Trades : {len(r.trades)}")
    print(f"  Time   : {elapsed/60:.1f} min")

    return dict(label=label, n_feat=n_feat,
                sharpe=r.sharpe, maxdd=r.drawdown,
                ret=ret, trades=len(r.trades))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(ROOT / "config.yaml") as f:
        full_cfg = yaml.safe_load(f)

    csv_path = full_cfg["pipeline"]["data_path"]
    prices   = _load_prices(csv_path)
    df_all   = _load_raw(csv_path)
    df_49k   = df_all[df_all.index >= FILTER_START].copy()

    print(f"\nDataset : {len(df_49k):,} bars "
          f"({df_49k.index[0].date()} → {df_49k.index[-1].date()})")
    print(f"Champion: +{CHAMPION['sharpe']:.2f} Sharpe / "
          f"{CHAMPION['maxdd']:.1f}% MaxDD (39 features)")

    results = []

    # ── Config A: Baseline (no SMC) ──────────────────────────────────────────
    cfg_a = _make_base_cfg(full_cfg, "baseline")
    results.append(_run(
        "Config A  baseline        31 base + 8 latent = 39 feat",
        [],          # no extra signals
        cfg_a, df_49k, prices,
    ))

    # ── Config B: + SMC P1 (OB + FVG + DailyLevels) ─────────────────────────
    cfg_b = _make_base_cfg(full_cfg, "smc_p1")
    results.append(_run(
        "Config B  +SMC-P1         + OB(2) + FVG(3) + DailyHL(3) = ~45 feat",
        SMC_P1_SPEC,
        cfg_b, df_49k, prices,
    ))

    # ── Config C: + All SMC (P1 + Andean + SuperTrend + HA) ─────────────────
    cfg_c = _make_base_cfg(full_cfg, "smc_all")
    results.append(_run(
        "Config C  +SMC-All        P1 + Andean(4) + ST(2) + HA(2) = ~56 feat",
        SMC_P1_SPEC + SMC_P2_SPEC,
        cfg_c, df_49k, prices,
    ))

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 86
    print(f"\n\n{'='*W}")
    print(f"  SMC / ICT Signals vs Baseline — enc8, 49k M15 (May 2024 → May 2026)")
    print(f"{'='*W}")
    hdr = f"  {'Config':<54} {'Feat':>4} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}"
    print(hdr)
    print(f"  {'-'*(W-4)}")

    ch = CHAMPION
    print(f"  {ch['label']:<54} {'39':>4} {ch['sharpe']:>+7.2f} "
          f"{ch['maxdd']:>6.1f}% {ch['ret']:>+7.1%} {ch['trades']:>7}")
    print(f"  {'-'*(W-4)}")

    best = max(results, key=lambda r: r["sharpe"])
    for r in results:
        delta = r["sharpe"] - ch["sharpe"]
        flag  = "  ✓ BEATS CHAMPION!" if delta > 0 else f"  ({delta:+.2f})"
        if r is best and delta > 0:
            flag += "  ← NEW CHAMPION"
        print(f"  {r['label']:<54} {r['n_feat']:>4} {r['sharpe']:>+7.2f} "
              f"{r['maxdd']:>6.1f}% {r['ret']:>+7.1%} {r['trades']:>7}{flag}")

    print(f"  {'-'*(W-4)}")
    print(f"{'='*W}\n")

    best_delta = best["sharpe"] - ch["sharpe"]
    if best_delta > 0:
        print(f"  RESULT: SMC signals improve champion!")
        print(f"  Best config: {best['label'].strip()}")
        print(f"  New Sharpe: {best['sharpe']:+.2f}  ({best_delta:+.2f} over baseline)")
        print(f"  MaxDD: {best['maxdd']:.1f}%  (champion: {ch['maxdd']:.1f}%)")
        if best["maxdd"] <= 20.0:
            print(f"  → PROMOTE to Phase 21 champion. Update WIN-RESEARCH.MD.")
        else:
            print(f"  → MaxDD > 20% threshold. Investigate before promoting.")
    else:
        print(f"  RESULT: Champion +{ch['sharpe']:.2f} holds.")
        print(f"  Best new config: {best['sharpe']:+.2f} ({best_delta:+.2f} vs champion).")
        print(f"  Saturation principle confirmed — SMC signals do not add new information")
        print(f"  beyond the 31 base + 8 latent combination.")
        print(f"  → Proceed to Option A (deployment) or Option B (LLM integration).")
    print()


if __name__ == "__main__":
    main()
