"""
backtest_clean_wf_finetune.py — Leakage-free WF via pre-train + per-fold fine-tune.

THE ENCODER LEAKAGE PROBLEM
────────────────────────────
Standard WF trains the encoder on first 80% of data. Early fold OOS windows fall
inside that 80% → encoder already "knows" those bars → inflated Sharpe.

Fixing this naively (fresh encoder per fold) fails: 120d sliding = ~8k bars +
90% hold-class → encoder collapses (Sharpe −10 to −22).

THE SOLUTION: Transfer Learning (Pre-train + Fine-tune)
────────────────────────────────────────────────────────
Stage 1  Pre-train encoder on first ENC_FRAC (60%) of data.
         36k bars, 30 epochs → good foundation weights (57% accuracy).

Stage 2  WF on remaining (1-ENC_FRAC) holdout, SHORT test windows (15d):
         For each fold, load Stage-1 weights then fine-tune 10 epochs on that
         fold's 120d sliding window. The fine-tuned encoder never saw the test
         window. CatBoost trains on fine-tuned latent features.

Why this retains Sharpe vs naked per-fold training:
  - Foundation weights already encode directional market structure
  - 10 fine-tune epochs on 120d adapts to recent regime without catastrophic
    forgetting (gradient steps are small relative to good initial weights)
  - 15d test windows → 15+ folds on the 40% holdout (vs 3 folds with 30d)
    → statistically meaningful result

Usage:
    conda run -n envmt5 python scripts/backtest_clean_wf_finetune.py
    conda run -n envmt5 python scripts/backtest_clean_wf_finetune.py --symbol EURUSD
    conda run -n envmt5 python scripts/backtest_clean_wf_finetune.py --symbol EURUSD --folds 3
    conda run -n envmt5 python scripts/backtest_clean_wf_finetune.py --enc-frac 0.60 --finetune-epochs 10
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
from scripts.train_candle_model import _add_extra_features, SYMBOL_CFG

# ── Defaults ───────────────────────────────────────────────────────────────────
ENC_FRAC        = 0.60   # first 60% used for foundation encoder
TRAIN_DAYS      = 120    # sliding window for CatBoost + fine-tune
TEST_DAYS       = 15     # short → more folds (15 vs 3 with 30d)
FINETUNE_EPOCHS = 10     # fine-tune epochs per fold
PRETRAIN_EPOCHS = 30     # foundation encoder epochs (full convergence)
THRESHOLD       = 0.60
SL_PIPS         = 10.0
TP_PIPS         = 30.0
SPREAD_PIPS     = 1.0
COMM_PIPS       = 0.5
RISK_PCT        = 0.01
INITIAL_BAL     = 10_000.0
LABEL_HORIZON   = 1
LABEL_THRESHOLD = 0.0005


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def _get_sliding_folds(index, train_days: int, test_days: int, max_folds: int = None):
    start    = index[0]
    end      = index[-1]
    td       = pd.Timedelta(days=train_days)
    te       = pd.Timedelta(days=test_days)
    folds    = []
    fold_idx = 0
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
    pnl     = [t["pips"] for t in trades]
    returns = pd.Series(pnl) / SL_PIPS * RISK_PCT
    span_days = (trades[-1]["ts"] - trades[0]["ts"]).total_seconds() / 86400
    tpy = len(trades) / span_days * 365.25 if span_days > 0 else len(trades)
    mean_r = returns.mean()
    std_r  = returns.std(ddof=1)
    if std_r < 1e-9:
        return float("nan")
    return float(mean_r / std_r * np.sqrt(tpy))


def _equity_stats(trades):
    balance = INITIAL_BAL
    eq = [balance]
    for t in trades:
        balance += balance * RISK_PCT * (t["pips"] / SL_PIPS)
        eq.append(balance)
    eq_s = pd.Series(eq)
    dd   = float(((eq_s.cummax() - eq_s) / eq_s.cummax()).max() * 100)
    ret  = (eq_s.iloc[-1] / INITIAL_BAL - 1) * 100
    return dd, ret


# ── Main per-symbol run ────────────────────────────────────────────────────────

def run_symbol(
    symbol: str,
    enc_frac: float,
    finetune_epochs: int,
    max_folds: int = None,
) -> None:
    cfg_s    = SYMBOL_CFG[symbol]
    pip_size = cfg_s["pip_size"]
    df_raw   = _load_raw(cfg_s["data_path"])
    n_all    = len(df_raw)

    enc_cutoff_idx = int(n_all * enc_frac)
    df_enc_train   = df_raw.iloc[:enc_cutoff_idx].copy()
    df_holdout     = df_raw.iloc[enc_cutoff_idx:].copy()
    enc_cutoff_dt  = df_enc_train.index[-1]

    print(f"\n{'='*72}")
    print(f"  CLEAN WF — PRE-TRAIN + FINE-TUNE  [{symbol}]")
    print(f"  Stage 1: foundation encoder on first {enc_frac:.0%} "
          f"({len(df_enc_train):,} bars, up to {enc_cutoff_dt.date()})")
    print(f"  Stage 2: WF on {len(df_holdout):,} bars  "
          f"({df_holdout.index[0].date()} → {df_holdout.index[-1].date()})")
    print(f"  WF: sliding {TRAIN_DAYS}d/{TEST_DAYS}d | "
          f"finetune_epochs={finetune_epochs} | "
          f"threshold={THRESHOLD} | SL={SL_PIPS}p TP={TP_PIPS}p")
    print(f"{'='*72}\n")

    t0_total = time.time()

    # ── Stage 1: Pre-train foundation encoder ─────────────────────────────────
    print(f"  [Stage 1] Training foundation encoder on first {enc_frac:.0%}...", flush=True)
    t0 = time.time()

    pipe_foundation = PredictorPipeline(PipelineConfig(
        label_horizon    = LABEL_HORIZON,
        label_threshold  = LABEL_THRESHOLD,
        encoder_mode     = "supervised",
        encoder_latent_dim = 8,
        encoder_epochs   = PRETRAIN_EPOCHS,
    ))
    pipe_foundation.build_features(df_enc_train, train_frac=1.0)
    foundation_weights = pipe_foundation._enc.get_state_dict()

    print(f"  [Stage 1] Done in {(time.time()-t0)/60:.1f} min. "
          f"Foundation encoder ready.\n", flush=True)

    # ── Stage 2: Per-fold fine-tune WF on holdout ─────────────────────────────
    folds = _get_sliding_folds(df_holdout.index, TRAIN_DAYS, TEST_DAYS,
                                max_folds=max_folds)
    print(f"  [Stage 2] {len(folds)} WF folds on holdout "
          f"({TRAIN_DAYS}d train / {TEST_DAYS}d test)\n", flush=True)

    if not folds:
        print("  ERROR: not enough holdout data. Reduce --enc-frac or "
              "check data length.")
        return

    header = (f"  {'Fold':>4}  {'Train window':>25}  {'Test window':>21}  "
              f"{'Trd':>4}  {'Win%':>5}  {'Sharpe':>7}  {'Enc(s)':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_trades = []

    for fold_idx, train_start, train_end, test_end in folds:
        df_fold_train = df_holdout[
            (df_holdout.index >= train_start) & (df_holdout.index < train_end)
        ].copy()
        df_fold_test = df_holdout[
            (df_holdout.index >= train_end) & (df_holdout.index < test_end)
        ].copy()

        if len(df_fold_train) < 500 or len(df_fold_test) < 30:
            continue

        t_enc = time.time()

        # Fine-tune: load foundation weights, train finetune_epochs on fold window.
        # Use 5× lower lr to avoid disrupting the foundation weights.
        pipe_fold = PredictorPipeline(PipelineConfig(
            label_horizon    = LABEL_HORIZON,
            label_threshold  = LABEL_THRESHOLD,
            encoder_mode     = "supervised",
            encoder_latent_dim = 8,
            encoder_epochs   = finetune_epochs,
            encoder_lr       = 2e-4,
        ))
        X_train, y_train = pipe_fold.build_features(
            df_fold_train, train_frac=1.0,
            pretrained_state_dict=foundation_weights,
        )
        X_train = _add_extra_features(df_fold_train, X_train)
        feature_cols = list(X_train.columns)
        enc_secs = time.time() - t_enc

        if len(X_train) < 50:
            continue

        # Build test features using the fine-tuned encoder (no re-fitting)
        df_fold_full = df_holdout[
            (df_holdout.index >= train_start) & (df_holdout.index < test_end)
        ].copy()
        X_base_full, _ = pipe_fold._fp.build(df_fold_full, fit=False)
        lat_full        = pipe_fold._enc.transform(df_fold_full)
        shared          = X_base_full.index.intersection(lat_full.index)
        X_full          = pd.concat([X_base_full.loc[shared], lat_full.loc[shared]], axis=1)
        X_full          = _add_extra_features(df_fold_full, X_full)
        for c in feature_cols:
            if c not in X_full.columns:
                X_full[c] = 0.0
        X_full  = X_full[feature_cols]
        X_test  = X_full[(X_full.index >= train_end) & (X_full.index < test_end)]

        if len(X_test) < 10:
            continue

        # CatBoost train + predict
        model = CatBoostModel(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, subsample=0.8, calibration_cv=0,
        )
        model.train(X_train, y_train)
        proba       = model.predict_proba(X_test)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        classes     = list(model._classes)
        prices_test = df_raw.reindex(X_test.index)
        fold_trades = _simulate_trades(proba, classes, X_test.index,
                                       prices_test, pip_size)

        n_t   = len(fold_trades)
        win_r = sum(1 for t in fold_trades if t["pips"] > 0) / n_t if n_t else 0.0
        f_sh  = _annualized_sharpe(fold_trades) if n_t >= 5 else float("nan")
        sh_s  = f"{f_sh:+.2f}" if not np.isnan(f_sh) else "  n/a"

        print(f"  {fold_idx:>4}  "
              f"{str(train_start.date()):>12} → {str(train_end.date()):<12}  "
              f"{str(train_end.date()):>10} → {str(test_end.date()):<10}  "
              f"{n_t:>4}  {win_r:>4.0%}  {sh_s:>7}  {enc_secs:>5.1f}s",
              flush=True)

        all_trades.extend(fold_trades)

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = (time.time() - t0_total) / 60
    print(f"\n  Elapsed total: {elapsed:.1f} min\n")

    print(f"  {'─'*68}")
    print(f"  FINAL RESULT  [{symbol}]  Pre-train {enc_frac:.0%} + Fine-tune "
          f"{finetune_epochs}ep/{TRAIN_DAYS}d/{TEST_DAYS}d")
    print(f"  {'─'*68}")

    if not all_trades:
        print("  No trades generated. Check data length or threshold.")
        return

    n_total = len(all_trades)
    wins    = sum(1 for t in all_trades if t["pips"] > 0)
    wr      = wins / n_total
    dd, ret = _equity_stats(all_trades)
    sh      = _annualized_sharpe(all_trades)
    sh_s    = f"{sh:+.3f}" if not np.isnan(sh) else "n/a"

    print(f"  Sharpe (annualized) : {sh_s}")
    print(f"  Win rate            : {wr:.1%}  ({wins}W / {n_total-wins}L)")
    print(f"  Max drawdown        : {dd:.1f}%")
    print(f"  Net return          : {ret:+.1f}%")
    print(f"  Total trades        : {n_total}  (~{n_total/(elapsed/60*525600/525600 * (len(df_holdout)/n_all*2.4)):.0f}/yr)" if n_total > 0 else "")
    print(f"\n  Comparison:")
    print(f"    Original WF (leaky encoder)      : "
          f"EURUSD +7.118 / USDJPY +14.414")
    print(f"    Per-fold fresh (starved encoder) : "
          f"EURUSD −10.580 / USDJPY −15.479")
    print(f"    Option B fixed encoder (3 folds) : "
          f"EURUSD −4.398 / USDJPY −5.096")
    print(f"    Pre-train + fine-tune (this)     : {sh_s}")
    print(f"  {'─'*68}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Leakage-free WF: pre-train foundation encoder then fine-tune per fold"
    )
    parser.add_argument("--symbol",          default=None, choices=list(SYMBOL_CFG.keys()))
    parser.add_argument("--enc-frac",        type=float, default=ENC_FRAC,
                        help="Fraction of data for foundation encoder training (default 0.60)")
    parser.add_argument("--finetune-epochs", type=int,   default=FINETUNE_EPOCHS,
                        help="Epochs per fold fine-tune (default 10)")
    parser.add_argument("--folds",           type=int,   default=None,
                        help="Limit to first N folds for a quick smoke test")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())

    print(f"\n{'='*72}")
    print(f"  CLEAN WF BENCHMARK — PRE-TRAIN + PER-FOLD FINE-TUNE")
    print(f"  Encoder: MLP supervised, {args.enc_frac:.0%} pre-train / "
          f"{args.finetune_epochs}ep fine-tune")
    print(f"  WF: sliding {TRAIN_DAYS}d/{TEST_DAYS}d  CatBoost  "
          f"threshold={THRESHOLD}")
    if args.folds:
        print(f"  NOTE: limited to first {args.folds} folds (smoke test)")
    print(f"{'='*72}")

    for sym in symbols:
        run_symbol(sym, args.enc_frac, args.finetune_epochs, max_folds=args.folds)

    print("Done.")


if __name__ == "__main__":
    main()
