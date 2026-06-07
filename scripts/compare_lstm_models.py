"""
Phase 23: LSTM Experiments — A/B/C walk-forward comparison.

Three configs tested against the same 49k-bar EURUSD M15 dataset:

  Config A — XGBoost + enc8 (champion baseline, 39 features)
             Reuses wf_cache_smc_baseline if available (saves ~60 min)

  Config B — LSTMModel on 39 features
             Same FeaturePipeline output, sequential model instead of XGBoost.
             Tests: "does recurrence over the feature matrix beat gradient boosting?"

  Config C — E2ELSTMModel on raw OHLCV (5 columns, no feature engineering)
             Bypasses FeaturePipeline entirely. LSTM learns from 50-bar OHLCV
             windows with per-window z-score normalization.
             Tests: "can end-to-end learning beat the full enc8+XGBoost stack?"

Decision rules (after completion):
  If B or C beats A by > +0.05 Sharpe AND MaxDD <= 20%:
      → Promote winner, move to Phase 24 (cross-pair features on new champion)
  Else:
      → Deploy XGBoost+enc8 champion. Move to cross-pair as separate add-on.

Usage:
    conda run -n envmt5 --no-capture-output python scripts/compare_lstm_models.py
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
from src.evaluation.walk_forward import WalkForwardConfig, WalkForwardValidator
from src.evaluation.backtester import BacktestConfig

FILTER_START = pd.Timestamp("2024-05-14 11:15:00")
CHAMPION = dict(sharpe=3.13, maxdd=13.3, ret=3.583, trades=524,
                label="Champion  XGBoost+enc8  39 feat  (cached run)")


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def _make_labels(close: pd.Series, horizon: int = 4, threshold: float = 0.0003) -> pd.Series:
    """Same formula as FeaturePipeline._make_labels."""
    future  = close.shift(-horizon)
    fwd_ret = (future - close) / close
    labels  = pd.Series(0, index=close.index, dtype=int)
    labels[fwd_ret >  threshold] =  1
    labels[fwd_ret < -threshold] = -1
    return labels.iloc[:-horizon]


# ── Config A / B runner (via PredictorPipeline) ────────────────────────────────

def _run_pipeline(label: str, cfg: PipelineConfig,
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


# ── Config C runner (E2E LSTM bypasses FeaturePipeline) ───────────────────────

def _run_e2e(label: str, full_cfg: dict,
             df_raw: pd.DataFrame, prices: pd.DataFrame) -> dict:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Dataset: {df_raw.index[0].date()} → {df_raw.index[-1].date()}"
          f"  ({len(df_raw):,} bars)")
    print(f"{'='*70}")
    t0 = time.time()

    pipe_cfg = full_cfg.get("pipeline", {})
    horizon   = pipe_cfg.get("label_horizon",   4)
    threshold = pipe_cfg.get("label_threshold", 0.0003)
    bt_cfg    = pipe_cfg.get("backtest", {})

    y = _make_labels(df_raw["close"], horizon=horizon, threshold=threshold)
    X = df_raw.loc[y.index]   # align raw OHLCV to label index
    print(f"  OHLCV rows after label alignment: {len(X):,}", flush=True)
    print(f"  Input columns: {list(X.columns)}")

    wf_cfg = WalkForwardConfig(
        model_type  = "e2e_lstm",
        window_type = "expanding",
        train_days  = 180,
        test_days   = 30,
        backtest    = BacktestConfig(
            threshold       = bt_cfg.get("threshold",       0.40),
            sl_pips         = bt_cfg.get("sl_pips",         30.0),
            tp_pips         = bt_cfg.get("tp_pips",         60.0),
            spread_pips     = bt_cfg.get("spread_pips",     1.0),
            initial_balance = bt_cfg.get("initial_balance", 10_000.0),
        ),
        cache_dir = str(ROOT / "data/models/wf_cache_e2e_lstm"),
    )

    r       = WalkForwardValidator(verbose=True).run(X, y, prices, wf_cfg)
    eq      = r.equity
    ret     = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    elapsed = time.time() - t0
    n_feat  = 5  # OHLCV cols

    print(f"\n  Sharpe : {r.sharpe:+.2f}")
    print(f"  MaxDD  : {r.drawdown:.1f}%")
    print(f"  Return : {ret:+.1%}")
    print(f"  Trades : {len(r.trades)}")
    print(f"  Time   : {elapsed/60:.1f} min")

    return dict(label=label, n_feat=n_feat,
                sharpe=r.sharpe, maxdd=r.drawdown,
                ret=ret, trades=len(r.trades))


# ── Main ───────────────────────────────────────────────────────────────────────

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
          f"{CHAMPION['maxdd']:.1f}% MaxDD (39 features, cached run)")
    print(f"\nPhase 23: LSTM experiments — 39-feature LSTM vs E2E LSTM from raw OHLCV")

    results = []

    # ── Config A: Champion XGBoost + enc8 (baseline) ──────────────────────────
    cfg_a = _make_base_cfg(full_cfg, "lstm_baseline")
    smc_cache = ROOT / "data/models/wf_cache_smc_baseline"
    if smc_cache.exists():
        cfg_a.wf_cache_dir = str(smc_cache)
        print(f"\n[A] Reusing SMC baseline cache: {smc_cache}")

    results.append(_run_pipeline(
        "Config A  XGBoost+enc8   31 base + 8 latent = 39 feat",
        cfg_a, df_49k, prices,
    ))

    # ── Config B: LSTMModel on 39 features ────────────────────────────────────
    cfg_b = _make_base_cfg(full_cfg, "lstm_39feat")
    cfg_b.model_type   = "lstm"
    cfg_b.encoder_mode = "supervised"   # keep enc8 latent features

    results.append(_run_pipeline(
        "Config B  LSTMModel      31 base + 8 latent = 39 feat  seq_len=20",
        cfg_b, df_49k, prices,
    ))

    # ── Config C: E2E LSTM from raw OHLCV ─────────────────────────────────────
    results.append(_run_e2e(
        "Config C  E2ELSTMModel   5 OHLCV cols  window=50  hidden=64",
        full_cfg, df_49k, prices,
    ))

    # ── Summary table ──────────────────────────────────────────────────────────
    W = 90
    print(f"\n\n{'='*W}")
    print(f"  Phase 23: LSTM Experiments vs Baseline — enc8, 49k M15 bars")
    print(f"{'='*W}")
    hdr = f"  {'Config':<56} {'Feat':>4} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}"
    print(hdr)
    print(f"  {'-'*(W-4)}")

    ch = CHAMPION
    print(f"  {ch['label']:<56} {'39':>4} {ch['sharpe']:>+7.2f} "
          f"{ch['maxdd']:>6.1f}% {ch['ret']:>+7.1%} {ch['trades']:>7}")
    print(f"  {'-'*(W-4)}")

    best = max(results, key=lambda r: r["sharpe"])
    for r in results:
        delta = r["sharpe"] - results[0]["sharpe"]   # vs fresh A (not cached champion)
        flag  = ""
        if r is not results[0]:
            flag = f"  ({delta:+.2f} vs A)"
        print(f"  {r['label']:<56} {r['n_feat']:>4} {r['sharpe']:>+7.2f} "
              f"{r['maxdd']:>6.1f}% {r['ret']:>+7.1%} {r['trades']:>7}{flag}")

    print(f"  {'-'*(W-4)}")
    print(f"{'='*W}\n")

    # Decision logic — compare B/C vs fresh Config A (stochastic variance considered)
    fresh_a = results[0]["sharpe"]
    winners = [r for r in results[1:] if r["sharpe"] > fresh_a + 0.05 and r["maxdd"] <= 20.0]

    if winners:
        best_w = max(winners, key=lambda r: r["sharpe"])
        delta  = best_w["sharpe"] - fresh_a
        print(f"  RESULT: {best_w['label'].strip()} BEATS baseline!")
        print(f"  Δ Sharpe: {delta:+.2f}  MaxDD: {best_w['maxdd']:.1f}%")
        print(f"  → Promote winner as new champion. Proceed to Phase 24 (cross-pair).")
    else:
        print(f"  RESULT: XGBoost+enc8 champion holds (fresh Sharpe: {fresh_a:+.2f}).")
        print(f"  LSTM variants did not beat the baseline by > +0.05 Sharpe.")
        print(f"  → Deploy champion as-is. Proceed to Phase 24 (cross-pair) as add-on.")
    print()


if __name__ == "__main__":
    main()
