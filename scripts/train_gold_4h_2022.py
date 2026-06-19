"""train_gold_4h_2022.py — train + save the 2022-regime XAUUSD 4H turning-point model.

REGIME-SPECIFIC BY DESIGN: trained only on 2022→present gold (the bull regime). On all data this
config is negative in 2016–21 (see data/GOLD_MTF_4H.md) — it is a forward-test candidate for demo,
NOT a validated all-weather edge. Config = the best 2022 cell: 4H-only features (no MTF), trend-scan
turning-point label, SL/TP entry (P(up)≥thr long / ≤1−thr short), ATR barriers SL=1×ATR, TP=3×ATR.

Saves a self-contained bundle (fitted feature pipeline + calibrated XGB + meta) to
data/models/gold_4h_2022/ for the live runner (scripts/gold_4h_live.py).

Usage: python scripts/train_gold_4h_2022.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import PredictorPipeline, PipelineConfig
from src.features.trend_labels import trend_scan_labels
from scripts.backtest_meta_labeling import _build_X
from scripts.backtest_champion_baseline import TemporalCalibratedXGBoost
from scripts.honest_champion_h4 import _resample

CFG = dict(symbol="XAUUSD", tf="H4", start="2022-01-01",
           thr=0.55, k_sl=1.0, k_tp=3.0, horizon=6,           # SL=1×ATR, TP=3×ATR, 6-bar (24h) force-close
           h_min=4, h_max=24, t_thresh=2.0, atr_n=14, pip=0.1)
OUT = ROOT / "data" / "models" / "gold_4h_2022"


def main():
    df = _resample(CFG["symbol"], CFG["tf"])
    df = df[df.index >= pd.Timestamp(CFG["start"])]
    print(f"Training XAUUSD 4H 2022-model on {len(df):,} bars "
          f"({df.index[0].date()}→{df.index[-1].date()})")

    # leak-free 4H features (encoder OFF), scaler fit on ALL the 2022→ data (final deploy model)
    pcfg = PipelineConfig(label_horizon=1, label_threshold=0.0, encoder_enabled=False)
    pipe = PredictorPipeline(pcfg)
    _, X, cols = _build_X(pipe, df, df)

    # trend-scan turning-point target → binary up/down on non-neutral bars
    lab = trend_scan_labels(df["close"], h_min=CFG["h_min"], h_max=CFG["h_max"], t_thresh=CFG["t_thresh"])
    y = lab.reindex(X.index)
    m = y.notna() & (y != 0)
    Xtr, ytr = X[m], (y[m] > 0).astype(int)
    print(f"  features={len(cols)}  train rows={len(Xtr)}  up={int(ytr.sum())}/{len(ytr)}")

    model = TemporalCalibratedXGBoost().train(Xtr, ytr)
    up_idx = list(model._classes).index(1) if 1 in model._classes else -1

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipe": pipe, "model": model, "cols": cols, "up_idx": up_idx, "cfg": CFG},
                OUT / "bundle.joblib")
    (OUT / "meta.json").write_text(json.dumps(
        {**CFG, "n_features": len(cols), "trained": str(df.index[-1].date()),
         "n_bars": len(df), "regime": "2022+ gold bull — regime-specific, demo forward-test only"},
        indent=2))
    print(f"  saved → {OUT}/bundle.joblib  (+ meta.json)")

    # sanity: reload + predict on the last 3 bars
    b = joblib.load(OUT / "bundle.joblib")
    p = b["model"].predict_proba(X.tail(3))[:, b["up_idx"]]
    print(f"  reload OK — last 3 P(up): {np.round(p, 3)}")


if __name__ == "__main__":
    main()
