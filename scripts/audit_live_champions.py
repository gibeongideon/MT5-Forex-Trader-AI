"""
audit_live_champions.py — Phase 1: Honest re-validation of the live champion models.

Quantifies TWO independent leakage sources that inflate the project's headline
Sharpe / win-rate numbers, and reports the honest, leak-free performance.

LEAK #1 — Encoder lookahead (already established this session)
  The enc8 MLP encoder is fit on the first 80% of ALL data (train_frac=0.80 in
  scripts/train_candle_model.py:179). The sliding-120d walk-forward then reuses
  that one encoder for every fold, so early OOS windows fall INSIDE the encoder's
  training set. Clean per-fold reconstructions go negative.

LEAK #2 — Multi-timeframe EMA lookahead (found this session)
  scripts/train_candle_model.py:_add_extra_features builds 1H/4H EMA features via
      close_1h = df["close"].resample("1h").last().ffill()
      ema_1h_m15 = ema.reindex(df.index, method="ffill")
  resample() labels each 1H bin by its LEFT edge (10:00) but the value is the LAST
  M15 close in the bin (10:45). After ffill, the 10:15 bar is assigned the 10:45
  close — 30 minutes in the FUTURE. The 4H features leak up to 3h45 ahead.
  For a 1-bar-horizon candle model this directly inflates next-bar accuracy.

WHAT THIS SCRIPT DOES
  Runs the candle-model walk-forward (CatBoost, sliding 120d/60d, encoder OFF to
  isolate the MTF leak cheaply) under two feature configs:
    A) MTF leaky   — _add_extra_features as shipped
    B) MTF fixed   — higher-TF series shifted by one completed bin (no lookahead)
  For each: aggregate Sharpe, win rate, and a PnL-distribution breakdown
  (TP hits / SL hits / tiny force-close wins / force-close losses) so we can see
  whether the headline 87-91% win rate is real signal or tiny-win inflation.

Usage:
    python scripts/audit_live_champions.py --symbol EURUSD
    python scripts/audit_live_champions.py --symbol USDJPY
    python scripts/audit_live_champions.py                 # both
"""
from __future__ import annotations

import argparse
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
from src.models.catboost_model import CatBoostModel
from scripts.train_candle_model import SYMBOL_CFG

# Candle-model config (mirrors train_candle_model.py)
TRAIN_DAYS      = 120
TEST_DAYS       = 60
THRESHOLD       = 0.60
SPREAD_PIPS     = 1.0
COMM_PIPS       = 0.5
RISK_PCT        = 0.01
INITIAL_BAL     = 10_000.0
LABEL_HORIZON   = 1
LABEL_THRESHOLD = 0.0005


# ── Feature builders: leaky vs fixed MTF ────────────────────────────────────────

def _session_flags(idx: pd.DatetimeIndex) -> pd.DataFrame:
    hour = idx.hour
    e = pd.DataFrame(index=idx)
    e["session_sydney"]  = ((hour >= 22) | (hour < 7)).astype(float)
    e["session_tokyo"]   = ((hour >= 0)  & (hour < 9)).astype(float)
    e["session_london"]  = ((hour >= 8)  & (hour < 17)).astype(float)
    e["session_ny"]      = ((hour >= 13) & (hour < 22)).astype(float)
    e["session_tok_lon"] = ((hour >= 8)  & (hour < 9)).astype(float)
    e["session_lon_ny"]  = ((hour >= 13) & (hour < 17)).astype(float)
    e["hour_sin"]        = np.sin(2 * np.pi * hour / 24)
    e["hour_cos"]        = np.cos(2 * np.pi * hour / 24)
    return e


def _mtf_emas(df_raw: pd.DataFrame, idx: pd.DatetimeIndex, fix_lookahead: bool) -> pd.DataFrame:
    """1H/4H EMA features. If fix_lookahead, shift higher-TF series by one
    completed bin so bar t only sees fully-closed higher-TF bars."""
    e = pd.DataFrame(index=idx)

    def _tf(span_resample, ema_span, slope_n, ratio_name, slope_name):
        close_tf = df_raw["close"].resample(span_resample).last().ffill()
        ema_tf   = close_tf.ewm(span=ema_span, adjust=False).mean()
        if fix_lookahead:
            ema_tf = ema_tf.shift(1)          # only completed higher-TF bars
        ema_m15  = ema_tf.reindex(df_raw.index, method="ffill")
        e[ratio_name] = ((df_raw["close"] - ema_m15) / df_raw["close"]).reindex(idx).fillna(0)
        e[slope_name] = (ema_m15.diff(slope_n) / df_raw["close"]).reindex(idx).fillna(0)

    _tf("1h", 20,  4, "ema_1h_ratio", "ema_1h_slope")
    _tf("4h", 50, 16, "ema_4h_ratio", "ema_4h_slope")
    return e


def add_extra_features(df_raw: pd.DataFrame, X: pd.DataFrame, fix_lookahead: bool) -> pd.DataFrame:
    idx   = X.index
    sess  = _session_flags(idx)
    mtf   = _mtf_emas(df_raw, idx, fix_lookahead)
    extra = pd.concat([sess, mtf], axis=1).reindex(idx).fillna(0)
    return pd.concat([X, extra], axis=1)


# ── WF plumbing ──────────────────────────────────────────────────────────────────

def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def _sliding_folds(index, train_days, test_days):
    start, end = index[0], index[-1]
    td, te = pd.Timedelta(days=train_days), pd.Timedelta(days=test_days)
    folds, fi, train_end = [], 0, start + td
    while train_end + te <= end + pd.Timedelta(days=1):
        test_end = min(train_end + te, end)
        folds.append((fi, train_end - td, train_end, test_end))
        fi += 1
        train_end += te
    return folds


def _simulate_1bar(proba, classes, index, prices, pip_size, sl_pips, tp_pips):
    """1-bar force-close. Returns trades with an 'exit' tag for PnL-distribution."""
    cm = {c: i for i, c in enumerate(classes)}
    pb = proba[:, cm.get(1,  cm.get("buy",  0))]
    ps = proba[:, cm.get(-1, cm.get("sell", 2))]
    trades = []
    for i, ts in enumerate(index):
        if pb[i] >= THRESHOLD and pb[i] > ps[i]:
            d = "buy"
        elif ps[i] >= THRESHOLD and ps[i] > pb[i]:
            d = "sell"
        else:
            continue
        nxt = prices.index[prices.index > ts]
        if not len(nxt):
            continue
        nt = nxt[0]
        entry = prices.loc[ts, "close"]
        h, l, c = prices.loc[nt, "high"], prices.loc[nt, "low"], prices.loc[nt, "close"]
        sl, tp = sl_pips * pip_size, tp_pips * pip_size
        if d == "buy":
            if l <= entry - sl:      pips, ex = -sl_pips - SPREAD_PIPS - COMM_PIPS, "SL"
            elif h >= entry + tp:    pips, ex =  tp_pips - SPREAD_PIPS - COMM_PIPS, "TP"
            else:                    pips, ex = (c - entry) / pip_size - SPREAD_PIPS - COMM_PIPS, "FC"
        else:
            if h >= entry + sl:      pips, ex = -sl_pips - SPREAD_PIPS - COMM_PIPS, "SL"
            elif l <= entry - tp:    pips, ex =  tp_pips - SPREAD_PIPS - COMM_PIPS, "TP"
            else:                    pips, ex = (entry - c) / pip_size - SPREAD_PIPS - COMM_PIPS, "FC"
        trades.append({"ts": ts, "dir": d, "pips": pips, "exit": ex})
    return trades


def _sharpe(trades, sl_pips):
    if len(trades) < 5:
        return float("nan")
    r = pd.Series([t["pips"] for t in trades]) / sl_pips * RISK_PCT
    span = (trades[-1]["ts"] - trades[0]["ts"]).total_seconds() / 86400
    tpy = len(trades) / span * 365.25 if span > 0 else len(trades)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(tpy)) if sd > 1e-12 else float("nan")


def _pnl_distribution(trades):
    """Break wins/losses into TP / SL / tiny force-close buckets."""
    n = len(trades)
    if n == 0:
        return {}
    tp   = sum(1 for t in trades if t["exit"] == "TP")
    sl   = sum(1 for t in trades if t["exit"] == "SL")
    fc_w = sum(1 for t in trades if t["exit"] == "FC" and t["pips"] > 0)
    fc_l = sum(1 for t in trades if t["exit"] == "FC" and t["pips"] <= 0)
    wins = sum(1 for t in trades if t["pips"] > 0)
    avg_win  = np.mean([t["pips"] for t in trades if t["pips"] > 0]) if wins else 0.0
    avg_loss = np.mean([t["pips"] for t in trades if t["pips"] <= 0]) if (n - wins) else 0.0
    fc_win_pips = [t["pips"] for t in trades if t["exit"] == "FC" and t["pips"] > 0]
    return dict(n=n, win_rate=wins / n,
                tp=tp, sl=sl, fc_w=fc_w, fc_l=fc_l,
                tp_pct=tp / n, sl_pct=sl / n, fc_w_pct=fc_w / n, fc_l_pct=fc_l / n,
                avg_win=avg_win, avg_loss=avg_loss,
                avg_fc_win=np.mean(fc_win_pips) if fc_win_pips else 0.0,
                avg_pips=np.mean([t["pips"] for t in trades]))


def run_config(symbol, df_raw, folds, pip_size, sl_pips, tp_pips, fix_lookahead):
    all_trades = []
    for fi, ts_start, ts_train_end, ts_test_end in folds:
        df_train = df_raw[(df_raw.index >= ts_start) & (df_raw.index < ts_train_end)].copy()
        df_fold  = df_raw[(df_raw.index >= ts_start) & (df_raw.index < ts_test_end)].copy()
        df_test  = df_raw[(df_raw.index >= ts_train_end) & (df_raw.index < ts_test_end)].copy()
        if len(df_train) < 500 or len(df_test) < 50:
            continue

        # encoder OFF — isolate the MTF leak cheaply (encoder leak established separately)
        pipe = PredictorPipeline(PipelineConfig(
            label_horizon=LABEL_HORIZON, label_threshold=LABEL_THRESHOLD,
            encoder_enabled=False,
        ))
        X_train, y_train = pipe.build_features(df_train, train_frac=1.0)
        X_train = add_extra_features(df_train, X_train, fix_lookahead)
        cols = list(X_train.columns)
        if len(X_train) < 100:
            continue

        X_base_fold, _ = pipe._fp.build(df_fold, fit=False)
        X_fold = add_extra_features(df_fold, X_base_fold, fix_lookahead)
        for c in cols:
            if c not in X_fold.columns:
                X_fold[c] = 0.0
        X_fold = X_fold[cols]
        X_test = X_fold[(X_fold.index >= ts_train_end) & (X_fold.index < ts_test_end)]
        if len(X_test) < 10:
            continue

        model = CatBoostModel(n_estimators=300, max_depth=6, learning_rate=0.05,
                              l2_leaf_reg=3.0, subsample=0.8, calibration_cv=0)
        model.train(X_train, y_train)
        proba = model.predict_proba(X_test)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        prices_test = df_raw.reindex(X_test.index)
        all_trades += _simulate_1bar(proba, list(model._classes), X_test.index,
                                     prices_test, pip_size, sl_pips, tp_pips)
    return all_trades


def audit_symbol(symbol):
    cfg_s    = SYMBOL_CFG[symbol]
    pip_size = cfg_s["pip_size"]
    sl_pips, tp_pips = cfg_s["sl_pips"], cfg_s["tp_pips"]
    df_raw   = _load_raw(cfg_s["data_path"])
    folds    = _sliding_folds(df_raw.index, TRAIN_DAYS, TEST_DAYS)

    print(f"\n{'='*74}")
    print(f"  CANDLE MODEL AUDIT — {symbol}   (encoder OFF; isolating MTF-EMA leak)")
    print(f"  {len(df_raw):,} bars  ({df_raw.index[0].date()} → {df_raw.index[-1].date()})")
    print(f"  Sliding {TRAIN_DAYS}d/{TEST_DAYS}d  |  {len(folds)} folds  |  CatBoost  |  thr={THRESHOLD}")
    print(f"  SL={sl_pips}p TP={tp_pips}p  1-bar force-close")
    print(f"{'='*74}")

    results = {}
    for tag, fix in [("MTF LEAKY (as shipped)", False), ("MTF FIXED (no lookahead)", True)]:
        t0 = time.time()
        trades = run_config(symbol, df_raw, folds, pip_size, sl_pips, tp_pips, fix)
        d = _pnl_distribution(trades)
        sh = _sharpe(trades, sl_pips)
        results[tag] = dict(dist=d, sharpe=sh, n=len(trades))
        print(f"\n  ── {tag} ──   ({(time.time()-t0)/60:.1f} min)")
        if not d:
            print("     no trades")
            continue
        print(f"     Sharpe (annualized) : {sh:+.3f}")
        print(f"     Win rate            : {d['win_rate']:.1%}   ({d['n']} trades)")
        print(f"     Exit breakdown      : TP {d['tp_pct']:.0%} ({d['tp']})  "
              f"SL {d['sl_pct']:.0%} ({d['sl']})  "
              f"force-close-win {d['fc_w_pct']:.0%} ({d['fc_w']})  "
              f"force-close-loss {d['fc_l_pct']:.0%} ({d['fc_l']})")
        print(f"     Avg win/loss (pips) : +{d['avg_win']:.1f} / {d['avg_loss']:.1f}   "
              f"avg force-close win: +{d['avg_fc_win']:.1f}p")
        print(f"     Avg pips per trade  : {d['avg_pips']:+.2f}")

    # Verdict
    print(f"\n  {'─'*70}")
    lk = results.get("MTF LEAKY (as shipped)", {})
    fx = results.get("MTF FIXED (no lookahead)", {})
    if lk.get("dist") and fx.get("dist"):
        print(f"  IMPACT OF MTF-EMA LOOKAHEAD ({symbol}):")
        print(f"    Win rate : {lk['dist']['win_rate']:.1%}  →  {fx['dist']['win_rate']:.1%}  "
              f"({(fx['dist']['win_rate']-lk['dist']['win_rate'])*100:+.1f}pp)")
        print(f"    Sharpe   : {lk['sharpe']:+.3f}  →  {fx['sharpe']:+.3f}  "
              f"({fx['sharpe']-lk['sharpe']:+.3f})")
    print(f"  {'─'*70}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, choices=list(SYMBOL_CFG.keys()))
    args = ap.parse_args()
    symbols = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())

    print(f"\n{'#'*74}")
    print(f"  PHASE 1 — HONEST RE-VALIDATION OF LIVE CHAMPIONS")
    print(f"  Diagnosing MTF-EMA lookahead (encoder leak established separately)")
    print(f"{'#'*74}")

    for sym in symbols:
        audit_symbol(sym)
    print("\nDone.")


if __name__ == "__main__":
    main()
