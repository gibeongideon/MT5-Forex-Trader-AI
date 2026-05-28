"""
H1 with properly recalibrated labels — 3 stop variants.

Previous H1 test used M15 labels (3-pip threshold, 4-bar = 1h horizon).
This script recalibrates for H1 scale:
    label_horizon  = 8   (8 H1 bars = 8 hours ahead)
    label_threshold = 0.001 (10 pips — matches H1 typical move)

Then tests 3 SL/TP configurations with the SAME trained models (fold
models are cached and reused across stop variants):
    Config A: SL=30p  TP=60p   (tight, 2:1 RR)
    Config B: SL=40p  TP=80p   (medium, 2:1 RR — ~1x H1 ATR)
    Config C: SL=50p  TP=100p  (wide, 2:1 RR — ~1.25x H1 ATR)

All vs M15 champion: XGBoost + enc8, 30p/60p, Sharpe +3.13

Usage
-----
  conda run -n envmt5 python scripts/compare_h1_calibrated.py
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
from src.evaluation.walk_forward import WalkForwardConfig, WalkForwardValidator
from src.evaluation.backtester import BacktestConfig
from src.risk_manager import RiskManager, RiskConfig

# ── Recalibrated H1 labels ────────────────────────────────────────────────────
H1_HORIZON    = 8      # 8 H1 bars = 8 hours ahead
H1_THRESHOLD  = 0.001  # 10 pips — H1-appropriate threshold

FILTER_START  = pd.Timestamp("2024-05-14")
CACHE_DIR     = str(ROOT / "data/models/wf_cache_h1_cal")   # shared → model reuse

# ── Stop configurations ───────────────────────────────────────────────────────
STOP_CONFIGS = [
    (30.0,  60.0,  "SL=30p  TP=60p   (tight, same as M15)"),
    (40.0,  80.0,  "SL=40p  TP=80p   (medium, ~1x H1 ATR)"),
    (50.0,  100.0, "SL=50p  TP=100p  (wide,  ~1.25x H1 ATR)"),
]

# ── Reference baselines ───────────────────────────────────────────────────────
M15_CHAMPION = dict(label="M15 XGBoost+enc8  30p/60p  [CHAMPION]",
                    sharpe=3.13, maxdd=13.3, ret=3.583, trades=524)
H1_UNCAL     = dict(label="H1  XGBoost+enc8  30p/60p  [uncalibrated labels]",
                    sharpe=1.47, maxdd=11.6, ret=0.312, trades=465)


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


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        full_cfg = yaml.safe_load(f)

    rm_cfg = full_cfg.get("risk_manager", {})

    # ── Load H1 data, filter to same period as M15 champion ───────────────────
    df_h1   = _load_raw(str(ROOT / "data/EURUSD_H1.csv"))
    df_h1   = df_h1[df_h1.index >= FILTER_START].copy()
    prices  = _load_prices(str(ROOT / full_cfg["pipeline"]["data_path"]))

    print(f"\nH1 dataset  : {len(df_h1):,} bars "
          f"({df_h1.index[0].date()} → {df_h1.index[-1].date()})")
    print(f"Label params: horizon={H1_HORIZON} bars ({H1_HORIZON}h)  "
          f"threshold={H1_THRESHOLD} ({H1_THRESHOLD*10000:.0f} pips)")
    print(f"Encoder     : supervised enc8, 30 epochs")
    print(f"Model       : XGBoost (same as M15 champion)")

    # ── Step 1: build features ONCE with recalibrated labels ──────────────────
    print(f"\n{'='*60}")
    print("  Building H1 features with recalibrated labels …")
    print(f"{'='*60}")
    t0 = time.time()

    cfg_build = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}), rm_cfg=rm_cfg,
    )
    cfg_build.model_type         = "xgboost"
    cfg_build.encoder_mode       = "supervised"
    cfg_build.encoder_latent_dim = 8
    cfg_build.encoder_epochs     = 30
    cfg_build.label_horizon      = H1_HORIZON
    cfg_build.label_threshold    = H1_THRESHOLD

    pipe     = PredictorPipeline(cfg_build)
    X, y     = pipe.build_features(df_h1)
    enc_time = time.time() - t0

    print(f"\n  Feature matrix : {X.shape}   ({X.index[0].date()} → {X.index[-1].date()})")
    print(f"  Labels         : buy={int((y==1).sum())}  hold={int((y==0).sum())}  sell={int((y==-1).sum())}")
    print(f"  Build time     : {enc_time/60:.1f} min")

    # ── Step 2: walk-forward with 3 stop configs (fold models shared) ─────────
    risk_manager = RiskManager(RiskConfig())
    results = []

    for sl, tp, stop_label in STOP_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  {stop_label}")
        print(f"{'='*60}")
        t0 = time.time()

        bt_cfg = BacktestConfig(
            initial_balance   = 10_000.0,
            sl_pips           = sl,
            tp_pips           = tp,
            spread_pips       = 1.0,
            risk_pct          = 0.01,
            max_slippage_pips = 0.0,
            use_regime_filter = False,
            risk_manager      = risk_manager,
        )
        wf_cfg = WalkForwardConfig(
            model_type  = "xgboost",
            window_type = "expanding",
            train_days  = 180,
            test_days   = 30,
            backtest    = bt_cfg,
            cache_dir   = CACHE_DIR,   # ← shared: fold models reused after first run
        )
        result = WalkForwardValidator(verbose=True).run(X, y, prices, wf_cfg)

        eq      = result.equity
        ret     = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
        elapsed = time.time() - t0

        print(f"\n  Sharpe : {result.sharpe:+.2f}")
        print(f"  MaxDD  : {result.drawdown:.1f}%")
        print(f"  Return : {ret:+.1%}")
        print(f"  Trades : {len(result.trades)}")
        print(f"  Time   : {elapsed/60:.1f} min")

        results.append(dict(
            label=f"H1 cal  {stop_label[:20].strip()}",
            sharpe=result.sharpe, maxdd=result.drawdown,
            ret=ret, trades=len(result.trades),
        ))

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 80
    print(f"\n\n{'='*W}")
    print(f"  H1 CALIBRATED LABELS (horizon=8h, threshold=10pip) — Stop Comparison")
    print(f"{'='*W}")
    print(f"  {'Config':<46} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}")
    print(f"  {'-'*(W-4)}")

    def _row(label, sharpe, maxdd, ret, trades, flag=""):
        print(f"  {label:<46} {sharpe:>+7.2f} {maxdd:>6.1f}% {ret:>+7.1%} {trades:>7}{flag}")

    # Reference rows
    c = M15_CHAMPION
    _row(c["label"], c["sharpe"], c["maxdd"], c["ret"], c["trades"])
    u = H1_UNCAL
    _row(u["label"], u["sharpe"], u["maxdd"], u["ret"], u["trades"])
    print(f"  {'-'*(W-4)}")

    best = max(results, key=lambda r: r["sharpe"])
    for r in results:
        delta = r["sharpe"] - M15_CHAMPION["sharpe"]
        flag  = "  ✓ BEATS M15!" if delta > 0 else f"  ({delta:+.2f} vs M15)"
        if r is best:
            flag += "  ← BEST H1"
        _row(r["label"], r["sharpe"], r["maxdd"], r["ret"], r["trades"], flag)

    print(f"  {'-'*(W-4)}")
    print(f"{'='*W}")
    print()

    b = best
    print(f"  Best H1 calibrated: {b['sharpe']:+.2f} Sharpe / {b['maxdd']:.1f}% MaxDD  →  {b['label']}")
    delta = b["sharpe"] - M15_CHAMPION["sharpe"]
    if delta > 0:
        print(f"  H1 CALIBRATED BEATS M15 champion by {delta:+.2f} Sharpe!")
    elif b["sharpe"] > H1_UNCAL["sharpe"]:
        print(f"  Calibration improved H1: {b['sharpe']:+.2f} vs uncalibrated {H1_UNCAL['sharpe']:+.2f}")
    else:
        print(f"  M15 still leads: +{M15_CHAMPION['sharpe']:.2f} vs H1 best {b['sharpe']:+.2f}")
    print()


if __name__ == "__main__":
    main()
