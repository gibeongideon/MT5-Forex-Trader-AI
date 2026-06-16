"""candle_color_backtest.py — predict next candle GREEN/RED (H1/H4/D1), tradeable test.

Leak-free calibrated classifier of next-bar direction (close[t+1] > close[t]):
  • features = validated engineered set + fixed MTF (reuse meta-labeling `_build_X`)
  • model = TemporalCalibratedXGBoost (binary, temporal-holdout isotonic calibration)
  • trade = 1-bar hold: enter at bar t close in predicted color if P(color) >= threshold,
            exit at bar t+1 close; net of REAL per-bar spread + commission
  • validation = expanding WF + discover/confirm + block-bootstrap CI + threshold sweep
  • GO = confirm net Sharpe >= +0.5 with CI lower bound > 0 AND positive both discover halves

Resamples the deep Dukascopy M15 (real spread) to the requested TF IN-MEMORY (no file writes,
so it never clobbers the CTA Yahoo D1 files). Cardinal rule: Sharpe >> 1 / hit ~100% => audit.

Usage:
    python scripts/candle_color_backtest.py --symbol EURUSD --tf H1 --sweep
    python scripts/candle_color_backtest.py --symbol XAUUSD --tf D1 --sweep
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import PredictorPipeline, PipelineConfig
from src.cta.bootstrap import block_bootstrap_sharpe
from scripts.backtest_meta_labeling import _build_X
from scripts.backtest_champion_baseline import TemporalCalibratedXGBoost, _load_raw

PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2, "XAUUSD": 1e-1}
RULE = {"H1": "1h", "H4": "4h", "D1": "1D"}
COMM_PIPS = 0.5
SPLIT = pd.Timestamp("2022-01-01")
# expanding WF params per TF (train days / step / test days)
WF = {"H1": (365, 60, 60), "H4": (540, 90, 90), "D1": (1095, 180, 180)}


def _resample(sym, tf):
    d = _load_raw(ROOT / "data" / f"{sym}_M15_long.csv")
    o = d.resample(RULE[tf], label="left", closed="left")
    df = pd.DataFrame({
        "open": o["open"].first(), "high": o["high"].max(), "low": o["low"].min(),
        "close": o["close"].last(), "tick_volume": o["tick_volume"].sum(),
        "spread": o["spread"].mean(), "real_volume": 0,
    }).dropna(subset=["open"])
    if tf == "D1":                       # weekday grid
        df = df[df.index.dayofweek < 5]
    return df


def _folds(index, train_d, step_d, test_d):
    start, end = index[0], index[-1]
    out, te = [], start + pd.Timedelta(days=train_d)
    while te + pd.Timedelta(days=test_d) <= end + pd.Timedelta(days=1):
        out.append((start, te, min(te + pd.Timedelta(days=test_d), end)))
        te += pd.Timedelta(days=step_d)
    return out


def _bars_per_year(idx):
    span = (idx[-1] - idx[0]).days / 365.25
    return len(idx) / span if span > 0 else len(idx)


def run(sym, tf, thresholds):
    pip = PIP[sym]
    df = _resample(sym, tf)
    print(f"\n{'='*72}\n  CANDLE COLOR — {sym} {tf}  ({len(df):,} bars {df.index[0].date()}→{df.index[-1].date()})")
    print(f"  next-bar up/down, calibrated XGB, 1-bar hold, net real spread\n{'='*72}", flush=True)
    cfg = PipelineConfig(label_horizon=1, label_threshold=0.0, encoder_enabled=False)
    folds = _folds(df.index, *WF[tf])
    bpy = _bars_per_year(df.index)

    cache = []   # (test_index, P_green, prices, spread)
    t0 = time.time()
    for tr_start, tr_end, te_end in folds:
        df_tr = df[(df.index >= tr_start) & (df.index < tr_end)].copy()
        df_full = df[(df.index >= tr_start) & (df.index < te_end)].copy()
        if len(df_tr) < 200:
            continue
        try:
            _, Xf, cols = _build_X(PredictorPipeline(cfg), df_tr, df_full)
        except Exception as e:
            print(f"  fold {tr_end.date()}: feature build failed ({e})"); continue
        # target on the SAME index as features: next-bar up
        up = (df_full["close"].shift(-1) > df_full["close"]).astype(int).reindex(Xf.index)
        tr_mask = Xf.index < tr_end
        Xtr, ytr = Xf[tr_mask], up[tr_mask]
        good = ytr.notna()
        Xtr, ytr = Xtr[good], ytr[good].astype(int)
        if len(Xtr) < 150 or ytr.nunique() < 2:
            continue
        m = TemporalCalibratedXGBoost(); m.train(Xtr, ytr)
        te_mask = (Xf.index >= tr_end) & (Xf.index < te_end)
        Xte = Xf[te_mask]
        if len(Xte) < 20:
            continue
        proba = m.predict_proba(Xte)
        cls = list(m._classes)
        gi = cls.index(1) if 1 in cls else len(cls) - 1
        pg = pd.Series(proba[:, gi], index=Xte.index)
        cache.append((pg, df.reindex(Xte.index.union(Xte.index + pd.Timedelta(1, "s")))))
        cache[-1] = (pg, df)   # keep full df for next-bar lookup
    print(f"  folds done in {(time.time()-t0)/60:.1f} min\n", flush=True)

    # evaluate thresholds against cached probabilities
    results = []
    for thr in thresholds:
        rets = {}
        for pg, fdf in cache:
            idx = list(fdf.index)
            pos = {ts: i for i, ts in enumerate(idx)}
            for ts, p in pg.items():
                if p >= thr:       direction = +1
                elif (1 - p) >= thr: direction = -1
                else:              continue
                i = pos.get(ts)
                if i is None or i + 1 >= len(idx):
                    continue
                nxt = idx[i + 1]
                entry, exit_ = fdf["close"].iloc[i], fdf["close"].iloc[i + 1]
                spr = (fdf["spread"].iloc[i] + fdf["spread"].iloc[i + 1] + 2 * COMM_PIPS) * pip / entry
                rets[ts] = direction * (exit_ / entry - 1.0) - spr
        r = pd.Series(rets).sort_index()
        if len(r) < 40:
            print(f"  thr={thr:.2f}: {len(r)} trades — insufficient"); continue
        def stat(x):
            x = x.dropna()
            if len(x) < 30: return None
            sd = x.std(ddof=1)
            return float(x.mean()/sd*np.sqrt(bpy)) if sd > 0 else float("nan")
        rc = r[r.index >= SPLIT]
        sh = stat(r); shc = stat(rc)
        lo, hi = block_bootstrap_sharpe(r.values, block=10, ppy=int(bpy))
        loc, hic = block_bootstrap_sharpe(rc.values, block=10, ppy=int(bpy)) if len(rc) >= 30 else (float("nan"), float("nan"))
        f = lambda v: f"{v:+.2f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else " n/a"
        hit = (r > 0).mean() * 100
        print(f"  thr={thr:.2f}  FULL Sh={f(sh)}[{f(lo)},{f(hi)}]  "
              f"CONFIRM Sh={f(shc)}[{f(loc)},{f(hic)}]  hit={hit:.1f}%  trades={len(r)} (conf {len(rc)})", flush=True)
        results.append((thr, sh, shc, loc, hit, len(r)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD", choices=list(PIP))
    ap.add_argument("--tf", default="H1", choices=list(RULE))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--thresholds", type=float, nargs="+", default=None)
    args = ap.parse_args()
    thr = [0.50, 0.55, 0.60] if args.sweep else (args.thresholds or [0.55])
    run(args.symbol, args.tf, thr)
    print("Done.")


if __name__ == "__main__":
    main()
