"""
Phase 25: Regime Detection + Dedicated Models — A/B walk-forward comparison.

Tests whether training separate specialist XGBoost models per market regime
improves on the single universal champion model.

The key insight: markets behave differently in trending vs ranging conditions.
One model trained on everything learns an average that fits nothing well.
RegimeRouter trains 4 specialists (trending-up, trending-down, ranging,
high-vol) and routes each bar to the right specialist at inference.

Configs
-------
  Config A — XGBoost + enc8 (champion baseline, 39 features)
             Single model for all market conditions.
             Reuses wf_cache_smc_baseline if available.

  Config B — RegimeRouter + enc8 (39 features)
             KMeans detects 4 regimes from [ATR-ratio, ADX, RSI].
             Separate XGBoost specialist per regime.
             Falls back to global model when regime has < 200 bars.

Usage:
    conda run -n envmt5 --no-capture-output python scripts/compare_regime.py
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

FILTER_START = pd.Timestamp("2024-05-14 11:15:00")
CHAMPION = dict(sharpe=3.13, maxdd=13.3, ret=3.583, trades=524,
                label="Champion  XGBoost+enc8  39 feat  (cached run)")


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


def _make_base_cfg(full_cfg: dict, model_type: str, cache_tag: str) -> PipelineConfig:
    cfg = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )
    cfg.model_type               = model_type
    cfg.encoder_mode             = "supervised"
    cfg.encoder_latent_dim       = 8
    cfg.encoder_epochs           = 30
    cfg.candle_tokenizer_enabled = False
    cfg.wf_cache_dir             = str(ROOT / f"data/models/wf_cache_{cache_tag}")
    return cfg


def _run(label: str, cfg: PipelineConfig,
         df_raw: pd.DataFrame, prices: pd.DataFrame) -> dict:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Dataset: {df_raw.index[0].date()} → {df_raw.index[-1].date()}"
          f"  ({len(df_raw):,} bars)")
    print(f"{'='*70}")
    t0 = time.time()

    pipe = PredictorPipeline(cfg)
    X, y = pipe.build_features(df_raw)
    n_feat = X.shape[1]
    print(f"  Feature count: {n_feat}", flush=True)

    r       = pipe.walk_forward(X, y, prices)
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
          f"{CHAMPION['maxdd']:.1f}% MaxDD")
    print(f"\nPhase 25: Regime-conditioned models vs single universal model")

    results = []

    # ── Config A: Champion XGBoost + enc8 (single model) ─────────────────────
    cfg_a = _make_base_cfg(full_cfg, "xgboost", "regime_baseline")
    smc_cache = ROOT / "data/models/wf_cache_smc_baseline"
    if smc_cache.exists():
        cfg_a.wf_cache_dir = str(smc_cache)
        print(f"\n[A] Reusing SMC baseline cache: {smc_cache}")

    results.append(_run(
        "Config A  XGBoost         39 feat  single model (universal)",
        cfg_a, df_49k, prices,
    ))

    # ── Config B: RegimeRouter + enc8 (4 specialists) ─────────────────────────
    cfg_b = _make_base_cfg(full_cfg, "regime_router", "regime_router")
    results.append(_run(
        "Config B  RegimeRouter    39 feat  4 specialists (trending/ranging/vol)",
        cfg_b, df_49k, prices,
    ))

    # ── Summary table ──────────────────────────────────────────────────────────
    W = 88
    print(f"\n\n{'='*W}")
    print(f"  Phase 25: Regime Detection vs Single Model — enc8, 49k M15 bars")
    print(f"{'='*W}")
    hdr = f"  {'Config':<54} {'Feat':>4} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}"
    print(hdr)
    print(f"  {'-'*(W-4)}")

    ch = CHAMPION
    print(f"  {ch['label']:<54} {'39':>4} {ch['sharpe']:>+7.2f} "
          f"{ch['maxdd']:>6.1f}% {ch['ret']:>+7.1%} {ch['trades']:>7}")
    print(f"  {'-'*(W-4)}")

    fresh_a = results[0]["sharpe"]
    for r in results:
        delta = r["sharpe"] - fresh_a if r is not results[0] else 0
        flag  = f"  ({delta:+.2f} vs A)" if r is not results[0] else "  ← fresh baseline"
        print(f"  {r['label']:<54} {r['n_feat']:>4} {r['sharpe']:>+7.2f} "
              f"{r['maxdd']:>6.1f}% {r['ret']:>+7.1%} {r['trades']:>7}{flag}")

    print(f"  {'-'*(W-4)}")
    print(f"{'='*W}\n")

    if len(results) > 1:
        delta_b = results[1]["sharpe"] - fresh_a
        if delta_b > 0.05 and results[1]["maxdd"] <= 20.0:
            print(f"  RESULT: RegimeRouter BEATS single XGBoost!")
            print(f"  Δ Sharpe: {delta_b:+.2f}  MaxDD: {results[1]['maxdd']:.1f}%")
            print(f"  → Promote RegimeRouter. Specialist models win — deploy per-regime approach.")
        elif delta_b > 0:
            print(f"  RESULT: Marginal improvement ({delta_b:+.2f}) — below +0.05 threshold.")
            print(f"  → Single model holds. Regime boundaries may need tuning.")
        else:
            print(f"  RESULT: Single model wins ({delta_b:+.2f}).")
            print(f"  Possible reason: not enough per-regime data with 18-fold expanding window.")
            print(f"  → Keep champion. Try Phase 26 (meta-labeling).")
    print()


if __name__ == "__main__":
    main()
