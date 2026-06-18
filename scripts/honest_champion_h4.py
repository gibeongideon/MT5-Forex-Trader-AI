"""honest_champion_h4.py — the enc8/champion APPROACH, leak-free, on H4 with all data.

Reproduces the spirit of the old +3.14 champion (XGBoost direction + SL/TP barrier exit +
probability threshold) but with EVERY leak removed and 10+ years of data:
  • data    = deep Dukascopy M15 (2015–2026) resampled to H4 in-memory (real spread)
  • features= validated engineered set, ENCODER OFF, NO leaky MTF-EMA (the two original leaks)
  • model   = TemporalCalibratedXGBoost (temporal-holdout isotonic calibration)
  • label   = next-bar direction (up/down); trade via ATR triple-barrier (1.5×ATR SL / 3×ATR TP,
              6-bar force-close) so barriers self-scale per instrument — no arbitrary M15 pips
  • valid   = expanding WF + threshold sweep + discover(2015-21)/confirm(2022-26) + bootstrap CI
  • GO       = confirm net Sharpe ≥ +0.5 with CI lower bound > 0 AND discover > 0

Cardinal rule: Sharpe ≫1 or hit ~100% ⇒ audit (the original 3.14 was leakage).

Usage:
    python scripts/honest_champion_h4.py              # all 4 instruments
    python scripts/honest_champion_h4.py --symbol XAUUSD --sweep
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
COMM_PIPS = 0.5
SPLIT = pd.Timestamp("2022-01-01")
RULE = {"M15": None, "H4": "4h", "D1": "1D"}                 # None = native, no resample
# expanding WF (train days / step / test) per timeframe
WF = {"M15": (365, 90, 90), "H4": (540, 90, 90), "D1": (1095, 180, 180)}
HORIZON = {"M15": 8, "H4": 6, "D1": 5}                       # force-close after N bars
K_SL, K_TP = 1.5, 3.0   # ATR multiples (1:2 R:R)
ATR_N = 14


def _resample(sym, tf):
    d = _load_raw(ROOT / "data" / f"{sym}_M15_long.csv")
    if RULE[tf] is None:                                      # M15 native
        df = d[["open", "high", "low", "close", "tick_volume", "spread"]].copy()
        df["real_volume"] = 0
    else:
        o = d.resample(RULE[tf], label="left", closed="left")
        df = pd.DataFrame({
            "open": o["open"].first(), "high": o["high"].max(), "low": o["low"].min(),
            "close": o["close"].last(), "tick_volume": o["tick_volume"].sum(),
            "spread": o["spread"].mean(), "real_volume": 0,
        }).dropna(subset=["open"])
        if tf == "D1":
            df = df[df.index.dayofweek < 5]
    # ATR(14) — uses only bars ≤ t (lookahead-free)
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_N).mean()
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
    horizon = HORIZON[tf]
    df = _resample(sym, tf)
    print(f"\n{'='*78}\n  HONEST CHAMPION (leak-free) — {sym} {tf}  "
          f"({len(df):,} bars {df.index[0].date()}→{df.index[-1].date()})")
    print(f"  XGB direction + {K_SL}×ATR/{K_TP}×ATR barrier, {horizon}-bar exit, real spread"
          f"\n{'='*78}", flush=True)
    cfg = PipelineConfig(label_horizon=1, label_threshold=0.0, encoder_enabled=False)
    folds = _folds(df.index, *WF[tf])

    cache = []   # (P_up Series, full df)
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
        if len(Xte) < 10:
            continue
        proba = m.predict_proba(Xte)
        cls = list(m._classes)
        gi = cls.index(1) if 1 in cls else len(cls) - 1
        cache.append((pd.Series(proba[:, gi], index=Xte.index), df))
    print(f"  folds done in {(time.time()-t0)/60:.1f} min ({len(cache)} test windows)\n", flush=True)

    def stat(x):
        x = x.dropna()
        if len(x) < 30:
            return None
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(_trades_per_year(x))) if sd > 0 else float("nan")

    def _trades_per_year(x):
        span = (x.index[-1] - x.index[0]).days / 365.25
        return len(x) / span if span > 0 else len(x)

    def _barrier_R(fdf, idx_pos, i, entry, d, sl_p, tp_p):
        """Realized gross R-multiple for direction d via the ATR triple-barrier (direction-
        specific: flipping d is a genuine re-sim, not a sign flip, because SL≠TP)."""
        for j in range(i + 1, i + 1 + horizon):
            hi, lo = fdf["high"].iloc[j], fdf["low"].iloc[j]
            if d == +1:
                if lo <= entry - sl_p: return -1.0
                if hi >= entry + tp_p: return K_TP / K_SL
            else:
                if hi >= entry + sl_p: return -1.0
                if lo <= entry - tp_p: return K_TP / K_SL
        c = fdf["close"].iloc[i + horizon]
        return (d * (c - entry)) / sl_p

    results = []
    for thr in thresholds:
        # per trade: gross R for the model direction, gross R for the ANTI direction, and cost
        gN, gA, C, ts_list = [], [], [], []
        for pup, fdf in cache:
            idx = list(fdf.index)
            pos = {ts: i for i, ts in enumerate(idx)}
            for ts, p in pup.items():
                if p >= thr:        d = +1
                elif (1 - p) >= thr: d = -1
                else:               continue
                i = pos.get(ts)
                atr = fdf["atr"].iloc[i] if i is not None else np.nan
                if i is None or np.isnan(atr) or atr <= 0 or i + horizon >= len(idx):
                    continue
                entry = fdf["close"].iloc[i]
                sl_p, tp_p = K_SL * atr, K_TP * atr
                cost_R = ((fdf["spread"].iloc[i] + 2 * COMM_PIPS) * pip) / sl_p
                gN.append(_barrier_R(fdf, pos, i, entry, d, sl_p, tp_p))
                gA.append(_barrier_R(fdf, pos, i, entry, -d, sl_p, tp_p))
                C.append(cost_R); ts_list.append(ts)
        if len(ts_list) < 40:
            print(f"  thr={thr:.2f}: {len(ts_list)} trades — insufficient"); continue
        gN = pd.Series(gN, index=ts_list).sort_index()
        gA = pd.Series(gA, index=ts_list).sort_index()
        C  = pd.Series(C,  index=ts_list).sort_index()
        f = lambda v: f"{v:+.2f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else " n/a"

        def line(tag, series):
            rc = series[series.index >= SPLIT]
            sh, shd, shc = stat(series), stat(series[series.index < SPLIT]), stat(rc)
            ppy = int(_trades_per_year(series))
            loc, hic = block_bootstrap_sharpe(rc.values, block=10, ppy=ppy) if len(rc) >= 30 else (np.nan, np.nan)
            win = (series > 0).mean() * 100
            go = (shc is not None and shc >= 0.5 and not np.isnan(loc) and loc > 0 and shd is not None and shd > 0)
            print(f"    {tag:16} FULL={f(sh)}  disc={f(shd)}  conf={f(shc)}[{f(loc)},{f(hic)}]  "
                  f"win={win:.0f}%{'  ✅GO' if go else ''}")

        print(f"  thr={thr:.2f}  ({len(ts_list)} trades, avg cost={C.mean():.3f}R)", flush=True)
        line("NORMAL net",  gN - C)
        line("ANTI   net",  gA - C)
        line("NORMAL gross", gN)
        line("ANTI   gross", gA)
        results.append((thr, stat(gN - C), stat(gA - C)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None, choices=list(PIP))
    ap.add_argument("--tf", default="H4", choices=list(RULE))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--thresholds", type=float, nargs="+", default=None)
    args = ap.parse_args()
    thr = [0.50, 0.55, 0.60] if args.sweep else (args.thresholds or [0.50, 0.55, 0.60])
    for s in ([args.symbol] if args.symbol else list(PIP)):
        run(s, args.tf, thr)
    print("\nDone.")


if __name__ == "__main__":
    main()
