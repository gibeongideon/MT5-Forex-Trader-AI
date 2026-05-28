"""
Session features + K-Means Candle Clustering — A/B/C comparison.

Tests two new feature groups against the +3.13 M15 champion:

  Config A — baseline (31 base + 8 latent = 39 features)     [CHAMPION +3.13]
  Config B — + session/time features (37 base + 8 latent = 45 features)
             Adds: hour_sin, hour_cos, is_london_open, is_ny_open,
                   is_london_ny_overlap, is_asia
  Config C — + session + K-Means candle cluster (45 + 1 = 46 features)
             Adds: candle_cluster (KMeans k=32 on bar shape)

All configs use the same pipeline:
  - XGBoost + supervised enc8 (30 epochs)
  - 49k M15 rows (May 2024 → May 2026)
  - Expanding walk-forward 180d/30d

Each config builds its own feature matrix from the raw CSV (encoder retrains
on new feature set). Cache dirs are separate.

Usage
-----
  conda run -n envmt5 python scripts/compare_new_features.py
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
BASELINE     = dict(sharpe=3.13, maxdd=13.3, ret=3.583, trades=524,
                    label="Baseline  31+8=39 feat  no session  no cluster")


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


def _make_cfg(full_cfg: dict, session: bool, cluster: bool,
              cache_tag: str) -> PipelineConfig:
    cfg = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )
    cfg.model_type               = "xgboost"
    cfg.encoder_mode             = "supervised"
    cfg.encoder_latent_dim       = 8
    cfg.encoder_epochs           = 30
    cfg.candle_tokenizer_enabled = cluster
    cfg.candle_tokenizer_clusters = 32
    cfg.wf_cache_dir             = str(ROOT / f"data/models/wf_cache_feat_{cache_tag}")
    return cfg


def _run(label: str, cfg: PipelineConfig, df_raw: pd.DataFrame,
         prices: pd.DataFrame) -> dict:
    print(f"\n{'='*66}")
    print(f"  {label}")
    print(f"  session={cfg.candle_tokenizer_enabled or True}  "
          f"cluster={cfg.candle_tokenizer_enabled}")
    print(f"  Dataset: {df_raw.index[0].date()} → {df_raw.index[-1].date()}"
          f"  ({len(df_raw):,} bars)")
    print(f"{'='*66}")
    t0 = time.time()

    pipe = PredictorPipeline(cfg)
    X, y = pipe.build_features(df_raw)
    print(f"  Feature count: {X.shape[1]}")
    r    = pipe.walk_forward(X, y, prices)

    eq      = r.equity
    ret     = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    elapsed = time.time() - t0

    print(f"\n  Sharpe : {r.sharpe:+.2f}")
    print(f"  MaxDD  : {r.drawdown:.1f}%")
    print(f"  Return : {ret:+.1%}")
    print(f"  Trades : {len(r.trades)}")
    print(f"  Time   : {elapsed/60:.1f} min")

    return dict(label=label, n_feat=X.shape[1],
                sharpe=r.sharpe, maxdd=r.drawdown,
                ret=ret, trades=len(r.trades))


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        full_cfg = yaml.safe_load(f)

    csv_path = full_cfg["pipeline"]["data_path"]
    prices   = _load_prices(csv_path)
    df_all   = _load_raw(csv_path)
    df_49k   = df_all[df_all.index >= FILTER_START].copy()

    print(f"\nDataset: {len(df_49k):,} bars "
          f"({df_49k.index[0].date()} → {df_49k.index[-1].date()})")
    print(f"Baseline: +{BASELINE['sharpe']:.2f} Sharpe (39 features)")

    results = []

    # ── Config A: baseline — same pipeline as champion ────────────────────
    cfg_a = _make_cfg(full_cfg, session=False, cluster=False, cache_tag="baseline")
    results.append(_run(
        "Config A  baseline        31 base + 8 latent = 39 feat",
        cfg_a, df_49k, prices,
    ))

    # ── Config B: + session features ──────────────────────────────────────
    cfg_b = _make_cfg(full_cfg, session=True, cluster=False, cache_tag="session")
    results.append(_run(
        "Config B  + session/time  37 base + 8 latent = 45 feat",
        cfg_b, df_49k, prices,
    ))

    # ── Config C: + session + candle cluster ──────────────────────────────
    cfg_c = _make_cfg(full_cfg, session=True, cluster=True, cache_tag="session_cluster")
    results.append(_run(
        "Config C  + session+cluster  37 base + 8 latent + 1 cluster = 46 feat",
        cfg_c, df_49k, prices,
    ))

    # ── Summary ───────────────────────────────────────────────────────────
    W = 84
    print(f"\n\n{'='*W}")
    print(f"  Session + K-Means Candle Cluster vs Baseline — enc8, 49k M15 (May 2024+)")
    print(f"{'='*W}")
    print(f"  {'Config':<52} {'Feat':>4} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}")
    print(f"  {'-'*(W-4)}")

    b = BASELINE
    print(f"  {b['label']:<52} {'39':>4} {b['sharpe']:>+7.2f} "
          f"{b['maxdd']:>6.1f}% {b['ret']:>+7.1%} {b['trades']:>7}")
    print(f"  {'-'*(W-4)}")

    best = max(results, key=lambda r: r["sharpe"])
    for r in results:
        delta = r["sharpe"] - b["sharpe"]
        flag  = "  ✓ BEATS!" if delta > 0 else f"  ({delta:+.2f})"
        if r is best:
            flag += "  ← BEST"
        print(f"  {r['label']:<52} {r['n_feat']:>4} {r['sharpe']:>+7.2f} "
              f"{r['maxdd']:>6.1f}% {r['ret']:>+7.1%} {r['trades']:>7}{flag}")

    print(f"  {'-'*(W-4)}")
    print(f"{'='*W}\n")

    br = best
    delta = br["sharpe"] - b["sharpe"]
    if delta > 0:
        print(f"  RESULT: New features help! Best: {br['sharpe']:+.2f}  "
              f"({delta:+.2f} over baseline)")
    else:
        abs_best = max(results, key=lambda r: r["sharpe"])
        print(f"  RESULT: Baseline +{b['sharpe']:.2f} holds. "
              f"Best new config: {abs_best['sharpe']:+.2f} ({delta:+.2f}).")
        if br["maxdd"] < b["maxdd"]:
            print(f"  However, {br['label'].strip()} has lower MaxDD "
                  f"({br['maxdd']:.1f}% vs {b['maxdd']:.1f}%).")
    print()


if __name__ == "__main__":
    main()
