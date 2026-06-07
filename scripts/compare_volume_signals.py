"""
Phase 22-A: Volume Anomaly Features vs Champion Baseline — A/B comparison.

Tests whether tick_volume-derived features improve on the champion
(XGBoost + enc8, +3.13 Sharpe, 39 features).

Hypothesis: tick_volume was never used beyond ATR computation. Volume spikes
indicate institutional activity and may carry information orthogonal to enc8.

Configs
-------
  Config A — Baseline (39 features)
             31 base + 8 latent — champion config
             Reuses wf_cache_smc_baseline if available (saves ~60 min)

  Config B — + Volume anomaly (42 features)
             + vol_ratio, vol_zscore, vol_fast_slow

Saturation guard:
  If Config B also hurts Sharpe → accept full saturation, deploy champion as-is.
  If Config B improves → promote and test LLM signals (Phase 22-B).

Usage:
    conda run -n envmt5 --no-capture-output python scripts/compare_volume_signals.py
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
from src.features.volume_signals import volume_signals

# Register volume_signals as OHLCV-type (receives full DataFrame, not just close)
_fp_module._OHLCV_FUNS |= {volume_signals}

VOLUME_SPEC = [
    (("vol_ratio", "vol_zscore", "vol_fast_slow"), volume_signals, {}),
]

FILTER_START = pd.Timestamp("2024-05-14 11:15:00")
CHAMPION = dict(sharpe=3.13, maxdd=13.3, ret=3.583, trades=524,
                label="Champion  31+8=39 feat  no vol signals")


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
    cfg = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )
    cfg.model_type               = "xgboost"
    cfg.encoder_mode             = "supervised"
    cfg.encoder_latent_dim       = 8
    cfg.encoder_epochs           = 30
    cfg.candle_tokenizer_enabled = False
    cfg.wf_cache_dir             = str(ROOT / f"data/models/wf_cache_vol_{cache_tag}")
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
    print(f"\nPhase 22-A: Volume anomaly features (vol_ratio, vol_zscore, vol_fast_slow)")

    results = []

    # ── Config A: Baseline ────────────────────────────────────────────────────
    cfg_a = _make_base_cfg(full_cfg, "baseline")
    # Reuse SMC baseline cache if it exists (same config, saves ~60 min)
    smc_cache = ROOT / "data/models/wf_cache_smc_baseline"
    if smc_cache.exists():
        cfg_a.wf_cache_dir = str(smc_cache)
        print(f"\n[A] Reusing SMC baseline cache: {smc_cache}")

    results.append(_run(
        "Config A  baseline        31 base + 8 latent = 39 feat",
        [],
        cfg_a, df_49k, prices,
    ))

    # ── Config B: + Volume signals ────────────────────────────────────────────
    cfg_b = _make_base_cfg(full_cfg, "vol")
    results.append(_run(
        "Config B  +volume         + vol_ratio + vol_zscore + vol_fast_slow = 42 feat",
        VOLUME_SPEC,
        cfg_b, df_49k, prices,
    ))

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 86
    print(f"\n\n{'='*W}")
    print(f"  Phase 22-A: Volume Anomaly Signals vs Baseline — enc8, 49k M15 bars")
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
            flag += "  ← PROMOTE"
        print(f"  {r['label']:<54} {r['n_feat']:>4} {r['sharpe']:>+7.2f} "
              f"{r['maxdd']:>6.1f}% {r['ret']:>+7.1%} {r['trades']:>7}{flag}")

    print(f"  {'-'*(W-4)}")
    print(f"{'='*W}\n")

    best_delta = best["sharpe"] - ch["sharpe"]
    if best_delta > 0:
        print(f"  RESULT: Volume signals IMPROVE champion!")
        print(f"  Δ Sharpe: {best_delta:+.2f}  MaxDD: {best['maxdd']:.1f}%")
        if best["maxdd"] <= 20.0:
            print(f"  → Promote to Phase 22-A champion. Run Phase 22-B (LLM signals).")
        else:
            print(f"  → MaxDD > 20% threshold. Investigate before promoting.")
    else:
        print(f"  RESULT: Champion +{ch['sharpe']:.2f} holds.")
        print(f"  Volume signals: {best['sharpe']:+.2f} ({best_delta:+.2f} vs champion).")
        print(f"  Saturation principle confirmed — volume features do not add new information.")
        print(f"  → Accept full saturation. Deploy champion. No more signal experiments.")
    print()


if __name__ == "__main__":
    main()
