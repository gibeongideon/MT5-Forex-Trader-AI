"""
Phase 26: Meta-Labeling (Triple-Barrier) — A/B walk-forward comparison.

Tests whether triple-barrier labels improve on standard forward-return labels.

Standard labels (Config A):
  y = 1  if close[t+4] / close[t] - 1 > 0.03%
  y = -1 if close[t+4] / close[t] - 1 < -0.03%
  y = 0  otherwise

Triple-barrier labels (Config B):
  y = 1  if a LONG at bar[t] hits TP (60p) before SL (30p) within 96 bars
  y = -1 if a SHORT at bar[t] hits TP (60p) before SL (30p) within 96 bars
  y = 0  if neither trade hits TP before SL (time barrier = 24 hours)

The triple-barrier label directly models what we care about in trading:
"will this specific trade setup be profitable?" rather than "which direction
does price drift in 1 hour?". Labels match the actual TP/SL geometry.

Source: Marcos López de Prado — triple-barrier method.

Usage:
    conda run -n envmt5 --no-capture-output python scripts/compare_metalabels.py
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
from src.features.meta_labels import triple_barrier_labels_fast, label_stats

FILTER_START = pd.Timestamp("2024-05-14 11:15:00")
CHAMPION = dict(sharpe=3.13, maxdd=13.3, ret=3.583, trades=524,
                label="Champion  XGBoost+enc8  39 feat  (cached run)")

# Triple-barrier parameters — match backtester config
TB_TP_PIPS  = 60.0
TB_SL_PIPS  = 30.0
TB_HORIZON  = 96     # 96 × M15 = 24 hours max hold
TB_PIP_SIZE = 0.0001


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
    cfg.wf_cache_dir             = str(ROOT / f"data/models/wf_cache_{cache_tag}")
    return cfg


def _run_standard(label: str, cfg: PipelineConfig,
                  df_raw: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Config A — standard labels via PredictorPipeline (unchanged)."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Dataset: {df_raw.index[0].date()} → {df_raw.index[-1].date()}"
          f"  ({len(df_raw):,} bars)")
    print(f"{'='*70}")
    t0 = time.time()

    pipe   = PredictorPipeline(cfg)
    X, y   = pipe.build_features(df_raw)
    n_feat = X.shape[1]
    label_stats(y, "Standard labels")
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


def _run_triple_barrier(label: str, cfg: PipelineConfig,
                        df_raw: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Config B — triple-barrier labels replace standard labels."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Dataset: {df_raw.index[0].date()} → {df_raw.index[-1].date()}"
          f"  ({len(df_raw):,} bars)")
    print(f"{'='*70}")
    t0 = time.time()

    # Build features with standard pipeline (same 39 features)
    pipe = PredictorPipeline(cfg)
    X, _ = pipe.build_features(df_raw)   # ignore standard labels (_)
    n_feat = X.shape[1]

    # Replace labels with triple-barrier labels
    print(f"  Computing triple-barrier labels "
          f"(TP={TB_TP_PIPS}p SL={TB_SL_PIPS}p horizon={TB_HORIZON} bars)...",
          flush=True)
    t_label = time.time()
    y_tb = triple_barrier_labels_fast(
        df_raw["close"],
        tp_pips  = TB_TP_PIPS,
        sl_pips  = TB_SL_PIPS,
        horizon  = TB_HORIZON,
        pip_size = TB_PIP_SIZE,
    )
    print(f"  Label computation: {time.time()-t_label:.1f}s")

    # Align triple-barrier labels to feature matrix index
    y = y_tb.reindex(X.index).fillna(0).astype(int)
    label_stats(y, "Triple-barrier labels")
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
    print(f"\nPhase 26: Triple-barrier labels vs standard forward-return labels")
    print(f"  TP={TB_TP_PIPS}p  SL={TB_SL_PIPS}p  horizon={TB_HORIZON} bars ({TB_HORIZON//4}h)")

    results = []

    # ── Config A: Standard labels ─────────────────────────────────────────────
    cfg_a = _make_base_cfg(full_cfg, "metalabel_baseline")
    smc_cache = ROOT / "data/models/wf_cache_smc_baseline"
    if smc_cache.exists():
        cfg_a.wf_cache_dir = str(smc_cache)
        print(f"\n[A] Reusing SMC baseline cache: {smc_cache}")

    results.append(_run_standard(
        "Config A  Standard labels  forward-return > 0.03% in 4 bars",
        cfg_a, df_49k, prices,
    ))

    # ── Config B: Triple-barrier labels ───────────────────────────────────────
    cfg_b = _make_base_cfg(full_cfg, "metalabel_triple")
    results.append(_run_triple_barrier(
        f"Config B  Triple-barrier  P(TP={TB_TP_PIPS}p before SL={TB_SL_PIPS}p) in {TB_HORIZON} bars",
        cfg_b, df_49k, prices,
    ))

    # ── Summary table ──────────────────────────────────────────────────────────
    W = 88
    print(f"\n\n{'='*W}")
    print(f"  Phase 26: Meta-Labeling (Triple-Barrier) vs Standard — enc8, 49k M15 bars")
    print(f"{'='*W}")
    hdr = f"  {'Config':<56} {'Feat':>4} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}"
    print(hdr)
    print(f"  {'-'*(W-4)}")

    ch = CHAMPION
    print(f"  {ch['label']:<56} {'39':>4} {ch['sharpe']:>+7.2f} "
          f"{ch['maxdd']:>6.1f}% {ch['ret']:>+7.1%} {ch['trades']:>7}")
    print(f"  {'-'*(W-4)}")

    fresh_a = results[0]["sharpe"]
    for r in results:
        delta = r["sharpe"] - fresh_a if r is not results[0] else 0
        flag  = f"  ({delta:+.2f} vs A)" if r is not results[0] else "  ← fresh baseline"
        print(f"  {r['label']:<56} {r['n_feat']:>4} {r['sharpe']:>+7.2f} "
              f"{r['maxdd']:>6.1f}% {r['ret']:>+7.1%} {r['trades']:>7}{flag}")

    print(f"  {'-'*(W-4)}")
    print(f"{'='*W}\n")

    if len(results) > 1:
        delta_b = results[1]["sharpe"] - fresh_a
        if delta_b > 0.05 and results[1]["maxdd"] <= 20.0:
            print(f"  RESULT: Triple-barrier labels IMPROVE champion!")
            print(f"  Δ Sharpe: {delta_b:+.2f}  MaxDD: {results[1]['maxdd']:.1f}%")
            print(f"  → Promote. Relabel full dataset, retrain champion with TB labels.")
        elif delta_b > 0:
            print(f"  RESULT: Marginal improvement ({delta_b:+.2f}).")
            print(f"  → Investigate horizon/TP/SL sensitivity before promoting.")
        else:
            print(f"  RESULT: Standard labels win ({delta_b:+.2f}).")
            print(f"  Possible reason: standard labels are noisy but have 10× more positives.")
            print(f"  → Champion holds. Deploy.")
    print()


if __name__ == "__main__":
    main()
