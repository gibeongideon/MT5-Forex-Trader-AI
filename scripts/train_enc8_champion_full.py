"""train_enc8_champion_full.py — reproduce the +3.14 LEAKY enc8 champion on ALL data.

Same config as the original champion (the live pipeline_<SYM> model the bots run):
  XGBoost + supervised enc8 encoder, 40 feat (31 base + fractal_corr + 8 latent),
  4-bar label (h=4, thr=0.0003), expanding 180d/30d WF, exit thr=0.40 SL=30p TP=60p.

LEAKY exactly like the original +3.14: the enc8 encoder + scaler are fit ONCE on the first
80% of ALL bars (train_frac=0.80) and the resulting latents are reused across every WF fold
(encoder leak). This is NOT leak-free — it reproduces the inflated number on purpose, with the
full 285k-bar dataset, for all four pairs. Then a final XGBoost is fit on all data (keeping the
80%-fit encoder) and the model is saved to a SEPARATE folder data/models/pipeline_full_<SYM>/.

Usage:
    python scripts/train_enc8_champion_full.py              # all 4 pairs
    python scripts/train_enc8_champion_full.py --symbol EURUSD --no-save
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.pipeline import PredictorPipeline, PipelineConfig
from src.models.xgboost_model import XGBoostModel
from scripts.train_candle_model import SYMBOL_CFG
from scripts.backtest_champion_baseline import (
    _get_expanding_folds, _load_raw, _simulate_trades, _annualized_sharpe,
    _equity_stats, THRESHOLD, SL_PIPS, TP_PIPS,
)

MIN_TRAIN_DAYS, STEP_DAYS, TEST_DAYS = 180, 30, 30
LABEL_HORIZON, LABEL_THRESHOLD = 4, 0.0003
ENC_FRAC = 0.80   # encoder fit on first 80% of ALL bars, reused across folds (the leak)


def _cfg() -> PipelineConfig:
    return PipelineConfig(
        label_horizon=LABEL_HORIZON, label_threshold=LABEL_THRESHOLD,
        encoder_enabled=True, encoder_mode="supervised", encoder_latent_dim=8,
        encoder_epochs=30, fractal_enabled=True,
    )


def run_symbol(symbol: str, save: bool, max_folds=None) -> dict:
    pip_size = SYMBOL_CFG[symbol]["pip_size"]
    data_path = f"data/{symbol}_M15_long.csv"
    out_dir   = f"data/models/pipeline_full_{symbol}"
    df_raw = _load_raw(data_path)

    print(f"\n{'='*74}\n  +3.14 LEAKY ENC8 CHAMPION (all data) — {symbol}")
    print(f"  {len(df_raw):,} bars  {df_raw.index[0].date()} → {df_raw.index[-1].date()}")
    print(f"  XGBoost + enc8 (fit on first {ENC_FRAC:.0%} of ALL bars, reused = LEAK)")
    print(f"  label h={LABEL_HORIZON} thr={LABEL_THRESHOLD} | exit thr={THRESHOLD} SL={SL_PIPS} TP={TP_PIPS}\n{'='*74}", flush=True)

    # ── leaky features: encoder+scaler fit on first 80% of ALL bars, transform all ──
    cfg = _cfg()
    pipe = PredictorPipeline(cfg)
    t0 = time.time()
    X, y = pipe.build_features(df_raw, train_frac=ENC_FRAC)
    cols = list(X.columns)
    print(f"  Feature matrix: {X.shape[0]:,} × {X.shape[1]} (enc8 latents leaked across folds)")
    print(f"  feature build + encoder: {(time.time()-t0)/60:.1f} min", flush=True)

    # ── expanding WF: XGBoost per fold on the leaky features ──
    folds = _get_expanding_folds(X.index, MIN_TRAIN_DAYS, STEP_DAYS, TEST_DAYS, max_folds=max_folds)
    all_trades, t1 = [], time.time()
    for fi, tr_start, tr_end, te_end in folds:
        Xtr = X[(X.index >= tr_start) & (X.index < tr_end)]
        ytr = y.reindex(Xtr.index)
        Xte = X[(X.index >= tr_end) & (X.index < te_end)]
        if len(Xtr) < 500 or len(Xte) < 20 or ytr.nunique() < 2:
            continue
        m = XGBoostModel(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample=0.8, calibration_cv=3)
        m.train(Xtr, ytr)
        proba = m.predict_proba(Xte)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        all_trades += _simulate_trades(proba, list(m._classes), Xte.index,
                                       df_raw.reindex(Xte.index), pip_size)
    print(f"  WF: {len(folds)} folds, {(time.time()-t1)/60:.1f} min", flush=True)

    if not all_trades:
        print("  No trades."); return {"symbol": symbol, "sharpe": float("nan"), "n": 0}
    n = len(all_trades); wins = sum(1 for t in all_trades if t["pips"] > 0)
    sh = _annualized_sharpe(all_trades); dd, ret = _equity_stats(all_trades)
    print(f"\n  ── LEAKY RESULT — {symbol} ──")
    print(f"  Sharpe (annualized): {sh:+.3f}   win={wins/n:.1%}  trades={n}  maxDD={dd:.1f}%  netPnL={ret:+.1f}%", flush=True)

    # ── save deploy model: XGBoost on ALL data, keeping the 80%-fit enc8 ──
    if save:
        pipe.cfg.artifacts_dir = out_dir
        pipe._feature_cols = cols
        pipe.fit_full(X, y)
        pipe.save()
        Path(out_dir, "pair_meta.json").write_text(json.dumps(dict(
            symbol=symbol, pip_size=pip_size, sl_pips=SL_PIPS, tp_pips=TP_PIPS,
            threshold=THRESHOLD, label_horizon=LABEL_HORIZON, label_threshold=LABEL_THRESHOLD,
            encoder="enc8", enc_train_frac=ENC_FRAC, model="xgboost",
            data=data_path, n_features=len(cols), leaky=True, wf_sharpe=round(sh, 3),
        ), indent=2))
        print(f"  saved → {out_dir}/  ({len(cols)} feat, enc8@{ENC_FRAC:.0%})", flush=True)
    return {"symbol": symbol, "sharpe": sh, "n": n, "win": wins / n, "dd": dd, "ret": ret}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, choices=list(SYMBOL_CFG.keys()))
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    syms = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())
    print(f"\n{'#'*74}\n  +3.14 LEAKY ENC8 CHAMPION — RETRAIN ON ALL DATA (4 pairs)\n{'#'*74}")
    res = [run_symbol(s, save=not args.no_save, max_folds=args.folds) for s in syms]
    print(f"\n{'='*74}\n  SUMMARY — leaky WF Sharpe (all data)\n{'='*74}")
    for r in res:
        sh = f"{r['sharpe']:+.3f}" if not np.isnan(r['sharpe']) else "n/a"
        print(f"    {r['symbol']:>8}  Sharpe={sh}  trades={r.get('n',0)}  "
              f"win={r.get('win',0):.0%}  maxDD={r.get('dd',0):.0f}%")
    print(f"{'='*74}\nDone.")


if __name__ == "__main__":
    main()
