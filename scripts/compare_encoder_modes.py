"""
compare_encoder_modes.py — Side-by-side WF benchmark of all encoder modes.

Runs the same sliding-120d/60d walk-forward as train_candle_model.py for each
encoder mode and prints a comparison table.

Modes tested:
  supervised   — current champion (CrossEntropy on 4-bar direction)
  forecast     — NEW: MSELoss on next-8-bar log-returns
  contrastive  — NEW: SimCLR NT-Xent (no labels)
  autoencoder  — unsupervised MSE reconstruction

The test is fair: same CatBoost params, same fold splits, same SL/TP/threshold.
The ONLY thing that changes between runs is the encoder training method.

Usage:
    conda run -n envmt5 python scripts/compare_encoder_modes.py
    conda run -n envmt5 python scripts/compare_encoder_modes.py --symbol EURUSD
    conda run -n envmt5 python scripts/compare_encoder_modes.py --folds 3
    conda run -n envmt5 python scripts/compare_encoder_modes.py --modes supervised forecast
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.pipeline import PredictorPipeline, PipelineConfig
from scripts.train_candle_model import _add_extra_features, SYMBOL_CFG
from src.models.catboost_model import CatBoostModel

# ── Config (matches train_candle_model.py) ─────────────────────────────────────
TRAIN_DAYS      = 120
TEST_DAYS       = 60
THRESHOLD       = 0.60
SL_PIPS         = 10.0
TP_PIPS         = 30.0
SPREAD_PIPS     = 1.0
COMM_PIPS       = 0.5
RISK_PCT        = 0.01
INITIAL_BAL     = 10_000.0
LABEL_HORIZON   = 1
LABEL_THRESHOLD = 0.0005

ALL_MODES = ["supervised", "forecast", "contrastive", "autoencoder"]


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def _get_sliding_folds(index, train_days, test_days, max_folds=None):
    start = index[0]
    end   = index[-1]
    td    = pd.Timedelta(days=train_days)
    te    = pd.Timedelta(days=test_days)
    folds = []
    fold_idx  = 0
    train_end = start + td
    while train_end + te <= end + pd.Timedelta(days=1):
        test_end    = min(train_end + te, end)
        train_start = train_end - td
        folds.append((fold_idx, train_start, train_end, test_end))
        fold_idx  += 1
        train_end += te
        if max_folds and fold_idx >= max_folds:
            break
    return folds


def _simulate_trades(proba, classes, index, prices, pip_size):
    cm = {c: i for i, c in enumerate(classes)}
    pb = proba[:, cm.get(1,  cm.get("buy",  0))]
    ps = proba[:, cm.get(-1, cm.get("sell", 2))]
    trades = []
    for i, ts in enumerate(index):
        if pb[i] >= THRESHOLD and pb[i] > ps[i]:
            direction = "buy";  conf = float(pb[i])
        elif ps[i] >= THRESHOLD and ps[i] > pb[i]:
            direction = "sell"; conf = float(ps[i])
        else:
            continue
        nxt = prices.index[prices.index > ts]
        if not len(nxt):
            continue
        nt    = nxt[0]
        entry = prices.loc[ts,  "close"]
        h_nx  = prices.loc[nt,  "high"]
        l_nx  = prices.loc[nt,  "low"]
        c_nx  = prices.loc[nt,  "close"]
        sl    = SL_PIPS * pip_size
        tp    = TP_PIPS * pip_size
        if direction == "buy":
            if l_nx <= entry - sl:   pips = -SL_PIPS - SPREAD_PIPS - COMM_PIPS
            elif h_nx >= entry + tp: pips =  TP_PIPS - SPREAD_PIPS - COMM_PIPS
            else: pips = (c_nx - entry) / pip_size - SPREAD_PIPS - COMM_PIPS
        else:
            if h_nx >= entry + sl:   pips = -SL_PIPS - SPREAD_PIPS - COMM_PIPS
            elif l_nx <= entry - tp: pips =  TP_PIPS - SPREAD_PIPS - COMM_PIPS
            else: pips = (entry - c_nx) / pip_size - SPREAD_PIPS - COMM_PIPS
        trades.append({"ts": ts, "dir": direction, "conf": conf, "pips": pips})
    return trades


def _annualized_sharpe(trades):
    if len(trades) < 5:
        return float("nan")
    pnl = [t["pips"] for t in trades]
    returns = pd.Series(pnl) / SL_PIPS * RISK_PCT
    if len(trades) > 1:
        span_days = (trades[-1]["ts"] - trades[0]["ts"]).total_seconds() / 86400
        tpy = len(trades) / span_days * 365.25 if span_days > 0 else len(trades)
    else:
        tpy = len(trades)
    mean_r = returns.mean()
    std_r  = returns.std(ddof=1)
    if std_r < 1e-9:
        return float("nan")
    return float(mean_r / std_r * np.sqrt(tpy))


# ── Per-mode WF run ────────────────────────────────────────────────────────────

def run_mode_wf(
    symbol: str,
    mode: str,
    df_raw: pd.DataFrame,
    folds: list,
    pip_size: float,
) -> dict:
    """Run one full WF for a given encoder mode. Returns result dict."""
    all_trades = []

    for fold_idx, train_start, train_end, test_end in folds:
        df_train = df_raw[(df_raw.index >= train_start) & (df_raw.index < train_end)].copy()
        df_fold  = df_raw[(df_raw.index >= train_start) & (df_raw.index < test_end)].copy()
        df_test  = df_raw[(df_raw.index >= train_end)   & (df_raw.index < test_end)].copy()

        if len(df_train) < 500 or len(df_test) < 50:
            continue

        pipe_fold = PredictorPipeline(PipelineConfig(
            label_horizon    = LABEL_HORIZON,
            label_threshold  = LABEL_THRESHOLD,
            encoder_mode     = mode,
            encoder_latent_dim = 8,
            encoder_epochs   = 30,
        ))
        X_train, y_train = pipe_fold.build_features(df_train, train_frac=1.0)
        X_train = _add_extra_features(df_train, X_train)
        feature_cols = list(X_train.columns)

        if len(X_train) < 100:
            continue

        X_base_fold, _ = pipe_fold._fp.build(df_fold, fit=False)
        if pipe_fold._enc is not None:
            lat_fold   = pipe_fold._enc.transform(df_fold)
            shared     = X_base_fold.index.intersection(lat_fold.index)
            X_fold_all = pd.concat([X_base_fold.loc[shared], lat_fold.loc[shared]], axis=1)
        else:
            X_fold_all = X_base_fold
        X_fold_all = _add_extra_features(df_fold, X_fold_all)
        for c in feature_cols:
            if c not in X_fold_all.columns:
                X_fold_all[c] = 0.0
        X_fold_all = X_fold_all[feature_cols]
        X_test = X_fold_all[(X_fold_all.index >= train_end) & (X_fold_all.index < test_end)]

        if len(X_test) < 10:
            continue

        model = CatBoostModel(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, subsample=0.8, calibration_cv=0,
        )
        model.train(X_train, y_train)
        proba   = model.predict_proba(X_test)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        classes     = list(model._classes)
        prices_test = df_raw.reindex(X_test.index)
        fold_trades = _simulate_trades(proba, classes, X_test.index, prices_test, pip_size)
        all_trades.extend(fold_trades)

    if not all_trades:
        return {"mode": mode, "sharpe": float("nan"), "win_rate": 0.0,
                "max_dd": float("nan"), "n_trades": 0, "return_pct": float("nan")}

    n_total = len(all_trades)
    wins    = sum(1 for t in all_trades if t["pips"] > 0)
    wr      = wins / n_total
    balance = INITIAL_BAL
    eq      = [balance]
    for t in all_trades:
        balance += balance * RISK_PCT * (t["pips"] / SL_PIPS)
        eq.append(balance)
    eq_s = pd.Series(eq)
    dd   = float(((eq_s.cummax() - eq_s) / eq_s.cummax()).max() * 100)
    ret  = (eq_s.iloc[-1] / INITIAL_BAL - 1) * 100

    return {
        "mode":       mode,
        "sharpe":     _annualized_sharpe(all_trades),
        "win_rate":   wr,
        "max_dd":     dd,
        "n_trades":   n_total,
        "return_pct": ret,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare encoder modes side-by-side")
    parser.add_argument("--symbol",   default=None, choices=list(SYMBOL_CFG.keys()))
    parser.add_argument("--modes",    nargs="+", default=ALL_MODES,
                        help="Encoder modes to compare (space-separated)")
    parser.add_argument("--folds",    type=int, default=None,
                        help="Limit to first N folds (quick smoke test)")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())
    modes   = args.modes

    print(f"\n{'='*80}")
    print(f"  ENCODER MODE COMPARISON")
    print(f"  Modes: {modes}")
    print(f"  WF: sliding {TRAIN_DAYS}d/{TEST_DAYS}d  |  CatBoost  |  threshold={THRESHOLD}")
    print(f"  SL={SL_PIPS}p  TP={TP_PIPS}p  spread={SPREAD_PIPS}p  comm={COMM_PIPS}p")
    if args.folds:
        print(f"  NOTE: limited to first {args.folds} folds (smoke test)")
    print(f"{'='*80}\n")

    all_results = {}

    for symbol in symbols:
        cfg_s    = SYMBOL_CFG[symbol]
        pip_size = cfg_s["pip_size"]
        df_raw   = _load_raw(cfg_s["data_path"])
        folds    = _get_sliding_folds(df_raw.index, TRAIN_DAYS, TEST_DAYS,
                                       max_folds=args.folds)

        print(f"\n  Symbol: {symbol}  ({len(df_raw):,} bars, {len(folds)} folds)")
        print(f"  {'-'*76}")

        results = []
        for mode in modes:
            t0 = time.time()
            print(f"\n  Running mode={mode!r}...", flush=True)
            res = run_mode_wf(symbol, mode, df_raw, folds, pip_size)
            res["elapsed_min"] = (time.time() - t0) / 60
            results.append(res)
            sharpe_str = f"{res['sharpe']:+.3f}" if not np.isnan(res["sharpe"]) else "   n/a"
            dd_str     = f"{res['max_dd']:.1f}%" if not np.isnan(res["max_dd"]) else "  n/a"
            print(
                f"  [{mode:>12}]  Sharpe={sharpe_str}  Win={res['win_rate']:.1%}"
                f"  MaxDD={dd_str}  Trades={res['n_trades']}  "
                f"Ret={res['return_pct']:+.1f}%  ({res['elapsed_min']:.1f}min)",
                flush=True,
            )

        all_results[symbol] = results

        # Summary table
        print(f"\n{'='*80}")
        print(f"  SUMMARY — {symbol}")
        print(f"  {'Mode':>12}  {'Sharpe':>8}  {'Win%':>6}  {'MaxDD':>7}  {'Trades':>7}  {'Return':>8}")
        print(f"  {'-'*64}")
        for r in sorted(results, key=lambda x: x["sharpe"] if not np.isnan(x["sharpe"]) else -99, reverse=True):
            sh_s = f"{r['sharpe']:+.3f}" if not np.isnan(r["sharpe"]) else "   n/a"
            dd_s = f"{r['max_dd']:.1f}%" if not np.isnan(r["max_dd"]) else "  n/a"
            rt_s = f"{r['return_pct']:+.1f}%" if not np.isnan(r["return_pct"]) else "  n/a"
            print(f"  {r['mode']:>12}  {sh_s:>8}  {r['win_rate']:>5.1%}  {dd_s:>7}"
                  f"  {r['n_trades']:>7}  {rt_s:>8}")
        print(f"{'='*80}\n")

    # Save JSON results
    out_path = ROOT / "logs" / "encoder_mode_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        # Convert nan to None for JSON serialisation
        clean = {}
        for sym, rlist in all_results.items():
            clean[sym] = [
                {k: (None if isinstance(v, float) and np.isnan(v) else v)
                 for k, v in r.items()}
                for r in rlist
            ]
        json.dump(clean, f, indent=2)
    print(f"Results saved → {out_path}\n")


if __name__ == "__main__":
    main()
