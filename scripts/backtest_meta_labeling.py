"""
backtest_meta_labeling.py — Phase 3: Clean meta-labeling walk-forward.

ARCHITECTURE (López de Prado meta-labeling)
  Primary model  → decides SIDE (long/short) on each bar.
                   --primary rule : deterministic 1H-EMA trend rule (default)
                   --primary xgb  : XGBoost 3-class direction model
  Meta model     → binary P(this trade hits TP before SL) given features + side.
                   TemporalCalibratedXGBoost (isotonic, temporal holdout).
  Trade rule     → take the primary's side only when calibrated P(win) >= threshold.
  Exit           → SAME side-conditioned triple-barrier as the LABEL (ATR-scaled
                   TP/SL, fixed horizon) → label == realized PnL by construction.
  Sizing/Sharpe  → R-units: every trade risks RISK_PCT of balance regardless of
                   volatility (R_t = net_pips / sl_pips_at_entry).

ZERO-LEAKAGE PER FOLD (expanding 180d/30d)
  train window [train_start, train_end) split temporally:
    A = first 60%   → (barriers are pre-registered, not tuned here)
    B = last  40%   → meta-model training set (primary fires; label via barriers)
  PURGE: drop the last `horizon` bars of B whose label window crosses train_end.
  Meta model's own internal 20% temporal holdout fits the isotonic calibrator.
  Test [train_end, test_end) is strictly after B.
  Raw ATR (indicators.atr) and the fixed (lookahead-free) MTF features are used
  throughout — NOT the as-shipped _add_extra_features (which leaks 30min/3h45).

Usage:
    python scripts/backtest_meta_labeling.py --symbol EURUSD --primary rule
    python scripts/backtest_meta_labeling.py --symbol EURUSD --primary rule --sweep
    python scripts/backtest_meta_labeling.py --symbol EURUSD --primary xgb  --sweep
    python scripts/backtest_meta_labeling.py --symbol EURUSD --gate A     # exit-mismatch proof
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
from src.models.xgboost_model import XGBoostModel
from src.features.indicators import atr as raw_atr
from src.features.meta_labels import side_barrier_meta_label
from scripts.train_candle_model import SYMBOL_CFG
# reuse leak-free helpers from the audit + clean baseline
from scripts.audit_live_champions import add_extra_features  # fix_lookahead-aware MTF
from scripts.backtest_champion_baseline import (
    TemporalCalibratedXGBoost, _get_expanding_folds, _load_raw,
)

# ── Config ─────────────────────────────────────────────────────────────────────
MIN_TRAIN_DAYS  = 180
STEP_DAYS       = 30
TEST_DAYS       = 30
LABEL_HORIZON   = 4          # for the engineered-feature direction labels (xgb primary)
LABEL_THRESHOLD = 0.0003
# Pre-registered barrier geometry (smoke-tested: ~48% win, ~87% resolved on EURUSD)
TP_MULT         = 1.5
SL_MULT         = 1.5
BARRIER_HORIZON = 16         # 16 × M15 = 4h max hold
A_FRAC          = 0.60       # primary/meta temporal split inside train window
META_THRESHOLD  = 0.55
SPREAD_PIPS     = 1.0
COMM_PIPS       = 0.5
RISK_PCT        = 0.01
INITIAL_BAL     = 10_000.0


# ── Feature matrix (leak-free, encoder OFF, fixed MTF) ───────────────────────────

def _build_X(pipe: PredictorPipeline, df_fit: pd.DataFrame, df_target: pd.DataFrame):
    """Fit scaler on df_fit; return feature matrix for df_target (fixed MTF, no enc)."""
    X_fit, _ = pipe.build_features(df_fit, train_frac=1.0)   # fits scaler
    X_fit = add_extra_features(df_fit, X_fit, fix_lookahead=True)
    cols  = list(X_fit.columns)
    X_t, _ = pipe._fp.build(df_target, fit=False)
    X_t = add_extra_features(df_target, X_t, fix_lookahead=True)
    for c in cols:
        if c not in X_t.columns:
            X_t[c] = 0.0
    return X_fit, X_t[cols], cols


# ── Primary models (side decision) ──────────────────────────────────────────────

def _primary_rule(X: pd.DataFrame) -> pd.Series:
    """Deterministic 1H-EMA trend rule on the leak-free MTF features."""
    r = X["ema_1h_ratio"]; s = X["ema_1h_slope"]
    side = pd.Series(np.nan, index=X.index)
    side[(r > 0) & (s > 0)] =  1.0
    side[(r < 0) & (s < 0)] = -1.0
    return side


def _primary_xgb_side(X_train, y_dir_train, X_target, thr=0.40) -> pd.Series:
    """XGBoost 3-class direction model → side where confident, else NaN."""
    m = XGBoostModel(n_estimators=300, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample=0.8, calibration_cv=0)
    m.train(X_train, y_dir_train)
    proba   = m.predict_proba(X_target)
    classes = list(m._classes)
    cm = {c: i for i, c in enumerate(classes)}
    pb = proba[:, cm.get(1,  0)]
    ps = proba[:, cm.get(-1, 2)]
    side = pd.Series(np.nan, index=X_target.index)
    side[(pb >= thr) & (pb > ps)] =  1.0
    side[(ps >= thr) & (ps > pb)] = -1.0
    return side


# ── R-unit Sharpe / equity ───────────────────────────────────────────────────────

def _sharpe_R(trades):
    if len(trades) < 10:
        return float("nan")
    r = pd.Series([t["R"] for t in trades]) * RISK_PCT
    span = (trades[-1]["ts"] - trades[0]["ts"]).total_seconds() / 86400
    tpy  = len(trades) / span * 365.25 if span > 0 else len(trades)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(tpy)) if sd > 1e-12 else float("nan")


def _equity_stats(trades):
    bal = INITIAL_BAL; eq = [bal]
    for t in trades:
        bal += bal * RISK_PCT * t["R"]
        eq.append(bal)
    s = pd.Series(eq)
    dd = float(((s.cummax() - s) / s.cummax()).max() * 100)
    ret = (s.iloc[-1] / INITIAL_BAL - 1) * 100
    return dd, ret


def _trades_from(side_test, p_win, label_df, meta_threshold):
    """Build trade list from test-window signals filtered by meta P(win) threshold.
    label_df (from side_barrier_meta_label on the full fold) supplies matched-exit
    pips + sl_pips. Applies spread+commission, converts to R-units."""
    trades = []
    for ts in side_test.index:
        if ts not in label_df.index:      # primary didn't fire / no atr
            continue
        if p_win.get(ts, 0.0) < meta_threshold:
            continue
        row = label_df.loc[ts]
        net_pips = row["pips"] - SPREAD_PIPS - COMM_PIPS
        sl_pips  = row["sl_pips"]
        if sl_pips <= 0:
            continue
        trades.append({"ts": ts, "R": net_pips / sl_pips,
                       "win": net_pips > 0, "p_win": p_win.get(ts, 0.0)})
    return trades


# ── Per-symbol walk-forward ──────────────────────────────────────────────────────

def run_symbol(symbol, primary, thresholds, gate_a=False, max_folds=None, data_path=None):
    cfg_s    = SYMBOL_CFG[symbol]
    pip_size = cfg_s["pip_size"]
    df_raw   = _load_raw(data_path or cfg_s["data_path"])
    folds    = _get_expanding_folds(df_raw.index, MIN_TRAIN_DAYS, STEP_DAYS,
                                    TEST_DAYS, max_folds=max_folds)

    print(f"\n{'='*74}")
    print(f"  META-LABELING WF — {symbol}  |  primary={primary}"
          f"{'  [GATE A: no meta filter]' if gate_a else ''}")
    print(f"  {len(df_raw):,} bars  ({df_raw.index[0].date()} → {df_raw.index[-1].date()})")
    print(f"  expanding {MIN_TRAIN_DAYS}d/{STEP_DAYS}d  |  {len(folds)} folds  |  "
          f"barriers tp={TP_MULT}*ATR sl={SL_MULT}*ATR h={BARRIER_HORIZON}")
    print(f"  leak-free: encoder OFF, MTF lookahead FIXED, raw ATR, purged splits")
    print(f"{'='*74}\n")

    cfg = PipelineConfig(label_horizon=LABEL_HORIZON, label_threshold=LABEL_THRESHOLD,
                         encoder_enabled=False)
    bar_td = pd.Timedelta(minutes=15)

    # cache per-fold test predictions so threshold sweep is instant
    fold_cache = []
    t0 = time.time()

    for fi, tr_start, tr_end, te_end in folds:
        df_tr   = df_raw[(df_raw.index >= tr_start) & (df_raw.index < tr_end)].copy()
        df_full = df_raw[(df_raw.index >= tr_start) & (df_raw.index < te_end)].copy()
        df_te   = df_raw[(df_raw.index >= tr_end)   & (df_raw.index < te_end)].copy()
        if len(df_tr) < 1000 or len(df_te) < 50:
            continue

        # features (scaler fit on train window only), transform full fold
        try:
            X_fit, X_full, cols = _build_X(PredictorPipeline(cfg), df_tr, df_full)
        except Exception as e:
            print(f"  fold {fi}: feature build failed ({e})"); continue

        atr_full = raw_atr(df_full, 14)

        # primary side over the whole fold
        if primary == "rule":
            side_full = _primary_rule(X_full)
        else:
            # xgb primary trained on A (first 60% of train window) to predict B+test
            mid = tr_start + (tr_end - tr_start) * A_FRAC
            df_A = df_raw[(df_raw.index >= tr_start) & (df_raw.index < mid)].copy()
            X_A, _, _ = _build_X(PredictorPipeline(cfg), df_A, df_A)
            _, y_A = PredictorPipeline(cfg).build_features(df_A, train_frac=1.0)
            X_A = X_A.loc[X_A.index.intersection(y_A.index)]
            y_A = y_A.loc[X_A.index]
            side_full = _primary_xgb_side(X_A, y_A, X_full)

        # barrier labels over the whole fold (used for both meta-train and exits)
        # align side to the full price index (feature matrix drops indicator-warmup bars)
        side_aligned = side_full.reindex(df_full.index)
        label_full = side_barrier_meta_label(
            df_full["high"], df_full["low"], df_full["close"],
            side_aligned, atr_full, TP_MULT, SL_MULT, BARRIER_HORIZON, pip_size)

        # ── meta-training set = B window (last 40% of train), PURGED ──
        mid = tr_start + (tr_end - tr_start) * A_FRAC
        purge_cut = tr_end - BARRIER_HORIZON * bar_td
        B_mask = (label_full.index >= mid) & (label_full.index < purge_cut)
        lab_B  = label_full[B_mask]

        test_mask = (side_full.index >= tr_end) & (side_full.index < te_end)
        side_te   = side_full[test_mask].dropna()

        if gate_a:
            # Gate A: NO meta model — trade every primary signal (matched exit only)
            p_win = pd.Series(1.0, index=side_te.index)
            fold_cache.append((fi, tr_end, te_end, side_te, p_win,
                               label_full.reindex(side_te.index).dropna()))
            n_fire = len(side_te)
            print(f"  fold {fi:>2}  {str(tr_end.date())}→{str(te_end.date())}  "
                  f"meta_train(B)={len(lab_B)}  test_signals={n_fire}", flush=True)
            continue

        if len(lab_B) < 150 or lab_B["meta_y"].nunique() < 2:
            print(f"  fold {fi:>2}: too few meta-train rows ({len(lab_B)}) — skip")
            continue

        # meta features = engineered + side
        Xm_B = X_full.reindex(lab_B.index).copy()
        Xm_B["primary_side"] = lab_B["side"].values
        y_B  = lab_B["meta_y"].astype(int)

        meta = TemporalCalibratedXGBoost()
        meta.train(Xm_B, y_B)

        # predict P(win) on test signals
        Xm_te = X_full.reindex(side_te.index).copy()
        Xm_te["primary_side"] = side_te.values
        proba = meta.predict_proba(Xm_te)
        classes = list(meta._classes)
        win_col = classes.index(1) if 1 in classes else (len(classes) - 1)
        p_win = pd.Series(proba[:, win_col], index=side_te.index)

        lab_te = label_full.reindex(side_te.index).dropna()
        fold_cache.append((fi, tr_end, te_end, side_te, p_win, lab_te))
        print(f"  fold {fi:>2}  {str(tr_end.date())}→{str(te_end.date())}  "
              f"meta_train(B)={len(lab_B)} (win {y_B.mean():.0%})  "
              f"test_signals={len(side_te)}  p_win[{p_win.min():.2f},{p_win.max():.2f}]",
              flush=True)

    print(f"\n  Folds processed in {(time.time()-t0)/60:.1f} min.\n")

    # ── evaluate threshold(s) against cached predictions ──
    sweep = thresholds if not gate_a else [0.0]
    results = []
    for thr in sweep:
        all_trades, fold_sharpes = [], []
        for fi, tr_end, te_end, side_te, p_win, lab_te in fold_cache:
            tr = _trades_from(side_te, p_win, lab_te, thr)
            all_trades += tr
            if len(tr) >= 10:
                fold_sharpes.append(_sharpe_R(tr))
        if len(all_trades) < 10:
            print(f"  thr={thr:.2f}: {len(all_trades)} trades — insufficient")
            results.append(dict(symbol=symbol, primary=primary, thr=thr,
                                sharpe=float("nan"), n=len(all_trades),
                                win=float("nan"), dd=float("nan")))
            continue
        sh = _sharpe_R(all_trades)
        wins = sum(1 for t in all_trades if t["win"])
        wr = wins / len(all_trades)
        dd, ret = _equity_stats(all_trades)
        worst = min(fold_sharpes) if fold_sharpes else float("nan")
        med_n = int(np.median([sum(1 for t in all_trades
                    if t["ts"] >= te and t["ts"] < tn) for _, te, tn, *_ in fold_cache] or [0]))
        tag = "  [GATE A]" if gate_a else ""
        print(f"  thr={thr:.2f}{tag}  Sharpe={sh:+.3f}  win={wr:.1%}  "
              f"trades={len(all_trades)}  DD={dd:.1f}%  worst-fold={worst:+.2f}")
        results.append(dict(symbol=symbol, primary=primary, thr=thr, sharpe=sh,
                            n=len(all_trades), win=wr, dd=dd, ret=ret, worst=worst))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, choices=list(SYMBOL_CFG.keys()))
    ap.add_argument("--primary", default="rule", choices=["rule", "xgb"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--thresholds", type=float, nargs="+", default=None)
    ap.add_argument("--gate", default=None, choices=["A"])
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--tp-mult", type=float, default=None, help="override TP ATR multiple")
    ap.add_argument("--sl-mult", type=float, default=None, help="override SL ATR multiple")
    ap.add_argument("--horizon", type=int, default=None, help="override barrier horizon (bars)")
    ap.add_argument("--data", default=None, help="override data CSV path (e.g. deep history)")
    args = ap.parse_args()

    # allow barrier-geometry overrides (for asymmetric-barrier experiments)
    global TP_MULT, SL_MULT, BARRIER_HORIZON
    if args.tp_mult is not None:  TP_MULT = args.tp_mult
    if args.sl_mult is not None:  SL_MULT = args.sl_mult
    if args.horizon is not None:  BARRIER_HORIZON = args.horizon

    if args.sweep:
        thresholds = [0.50, 0.55, 0.60, 0.65]
    elif args.thresholds:
        thresholds = sorted(args.thresholds)
    else:
        thresholds = [META_THRESHOLD]

    symbols = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())
    print(f"\n{'#'*74}\n  PHASE 3 — CLEAN META-LABELING  (primary={args.primary})\n{'#'*74}")

    allr = []
    for sym in symbols:
        allr += run_symbol(sym, args.primary, thresholds,
                           gate_a=(args.gate == "A"), max_folds=args.folds,
                           data_path=args.data)

    print(f"\n{'='*74}\n  SUMMARY  (primary={args.primary})")
    print(f"  {'Symbol':>8} {'Thr':>5} {'Sharpe':>8} {'Win%':>6} {'Trades':>7} {'MaxDD':>7} {'Worst':>7}")
    print(f"  {'-'*56}")
    for r in allr:
        sh = f"{r['sharpe']:+.3f}" if not np.isnan(r['sharpe']) else "n/a"
        wr = f"{r['win']:.1%}" if not np.isnan(r['win']) else "n/a"
        dd = f"{r.get('dd', float('nan')):.1f}%" if not np.isnan(r.get('dd', float('nan'))) else "n/a"
        wf = f"{r.get('worst', float('nan')):+.2f}" if not np.isnan(r.get('worst', float('nan'))) else "n/a"
        print(f"  {r['symbol']:>8} {r['thr']:>5.2f} {sh:>8} {wr:>6} {r['n']:>7} {dd:>7} {wf:>7}")
    print(f"{'='*74}\n")
    print("Done.")


if __name__ == "__main__":
    main()
