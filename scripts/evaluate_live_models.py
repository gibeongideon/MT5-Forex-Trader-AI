"""
Evaluate live-ready models for EURUSD and USDJPY.

Runs a proper hold-out walk-forward (same method as champion validation)
on each pair to produce Sharpe, MaxDD, Return, Trades before paper trading.

Uses the last 12 months as out-of-sample test across multiple folds.
The model is trained fresh per fold on data BEFORE each test window —
this is genuine out-of-sample performance, not in-sample.

Usage:
    conda run -n envmt5 --no-capture-output python scripts/evaluate_live_models.py \\
        > logs/evaluate_live.log 2>&1 &
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
from src.evaluation.walk_forward import WalkForwardValidator, WalkForwardConfig
from src.evaluation.backtester import BacktestConfig

PAIRS = {
    "EURUSD": dict(
        csv      = "data/EURUSD_M15.csv",
        pip_size = 0.0001,
        sl       = 30,
        tp       = 60,
        spread   = 1.0,
        cache    = "wf_cache_eval_EURUSD",
    ),
    "USDJPY": dict(
        csv      = "data/USDJPY_M15.csv",
        pip_size = 0.01,
        sl       = 30,
        tp       = 60,
        spread   = 1.0,
        cache    = "wf_cache_eval_USDJPY",
    ),
}

# Same period as champion +2.31 baseline
FILTER_START = pd.Timestamp("2024-05-14 11:15:00")


def _load_df(path: str) -> pd.DataFrame:
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


def evaluate_pair(symbol: str, pair: dict, full_cfg: dict) -> dict:
    print(f"\n{'='*68}")
    print(f"  EVALUATING  {symbol}  — walk-forward hold-out")
    print(f"{'='*68}")

    csv    = str(ROOT / pair["csv"])
    df_all = _load_df(csv)
    df     = df_all[df_all.index >= FILTER_START].copy()
    prices = _load_prices(csv)
    prices = prices[prices.index >= FILTER_START]

    print(f"  Bars   : {len(df):,}  ({df.index[0].date()} → {df.index[-1].date()})")
    print(f"  Config : XGBoost + enc8  epochs=100  patience=15", flush=True)

    t0 = time.time()

    # Champion config with better patience for evaluation
    cfg = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )
    cfg.model_type               = "xgboost"
    cfg.encoder_mode             = "supervised"
    cfg.encoder_latent_dim       = 8
    cfg.encoder_epochs           = 100
    cfg.encoder_patience         = 15
    cfg.candle_tokenizer_enabled = False
    cfg.wf_cache_dir             = str(ROOT / "data/models" / pair["cache"])

    pipe   = PredictorPipeline(cfg)
    X, y   = pipe.build_features(df)
    n_feat = X.shape[1]

    print(f"  Features : {n_feat}   Labels : {y.value_counts().sort_index().to_dict()}", flush=True)

    bt_cfg = BacktestConfig(
        threshold       = 0.40,
        pip_size        = pair["pip_size"],
        sl_pips         = pair["sl"],
        tp_pips         = pair["tp"],
        spread_pips     = pair["spread"],
        commission_pips = 0.5,
        initial_balance = 10_000.0,
        risk_pct        = 0.01,
    )
    wf_cfg = WalkForwardConfig(
        model_type  = "xgboost",
        window_type = "expanding",
        train_days  = 180,
        test_days   = 30,
        backtest    = bt_cfg,
        cache_dir   = cfg.wf_cache_dir,
    )

    r       = WalkForwardValidator(verbose=True).run(X, y, prices, wf_cfg)
    eq      = r.equity
    ret     = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    elapsed = time.time() - t0

    print(f"\n  ── {symbol} Results ──────────────────────────────────────")
    print(f"  Sharpe  : {r.sharpe:+.2f}")
    print(f"  MaxDD   : {r.drawdown:.1f}%")
    print(f"  Return  : {ret:+.1%}")
    print(f"  Trades  : {len(r.trades)}")
    print(f"  Time    : {elapsed/60:.1f} min")

    return dict(symbol=symbol, n_feat=n_feat, sharpe=r.sharpe,
                maxdd=r.drawdown, ret=ret, trades=len(r.trades))


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        full_cfg = yaml.safe_load(f)

    print(f"\nEvaluating live models — walk-forward out-of-sample")
    print(f"Period  : {FILTER_START.date()} → 2026-06-05  (same as champion)")
    print(f"Method  : expanding window  180d train / 30d test")
    print(f"Config  : XGBoost + enc8  epochs=100  patience=15\n")

    results = []
    for symbol, pair in PAIRS.items():
        results.append(evaluate_pair(symbol, pair, full_cfg))

    # ── Summary ───────────────────────────────────────────────────────────────
    W = 72
    print(f"\n\n{'='*W}")
    print(f"  Live Model Evaluation — XGBoost + enc8  M15")
    print(f"{'='*W}")
    hdr = f"  {'Pair':<8} {'Feat':>4} {'Sharpe':>8} {'MaxDD':>7} {'Return':>9} {'Trades':>7}"
    print(hdr)
    print(f"  {'-'*(W-4)}")
    print(f"  {'Champion (cached):':<8}              +3.13   13.3%   +358.3%     524  ← reference")
    print(f"  {'Fresh avg:':<8}                      +2.31    8.0%    +43.0%     513  ← reference")
    print(f"  {'-'*(W-4)}")

    for r in results:
        verdict = "DEPLOY" if r["sharpe"] > 0.8 else "INVESTIGATE"
        print(
            f"  {r['symbol']:<8} {r['n_feat']:>4} {r['sharpe']:>+8.2f} "
            f"{r['maxdd']:>6.1f}% {r['ret']:>+8.1%} {r['trades']:>7}  → {verdict}"
        )

    print(f"  {'-'*(W-4)}")
    print(f"{'='*W}")

    deploy = [r["symbol"] for r in results if r["sharpe"] > 0.8]
    print(f"\n  Ready for paper trading : {deploy or 'none — check logs'}")

    if deploy:
        print(f"\n  Start paper trading:")
        for sym in deploy:
            p = PAIRS[sym]
            print(f"\n  {sym}:")
            print(f"    conda run -n envmt5 python src/bots/pipeline_bot.py \\")
            print(f"        --model-dir data/models/pipeline_{sym} \\")
            if sym != "EURUSD":
                print(f"        --symbol {sym} \\")
            print(f"        > logs/bot_{sym}.log 2>&1 &")
    print()


if __name__ == "__main__":
    main()
