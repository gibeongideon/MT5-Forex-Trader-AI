"""gold_intraday_turning.py — intraday turning-point predictor with ATR SL/TP, multi-timeframe.

Trade the highest-probability turning points intraday: classify trend direction (turning-point
labels), enter LONG on P(up)≥thr / SHORT on P(up)≤1−thr, exit via ATR triple-barrier (SL=1×ATR,
TP swept 1.5/2/3×ATR, force-close after the TF horizon). Net of REAL spread+commission, R-units.
Sweep timeframes 4H/2H/1H/30M/15M × thresholds × R:R and compare which supports intraday turning-
point trading net of cost. Leak-free: encoder OFF, features past-only, labels forward (target only).

Honest prior: intraday is cost-dominated (M15 directional was −29 to spread); expect 4H/2H best,
15M/30M cost-negative. GO = confirm Sharpe ≥+0.5, CI lower>0, positive both discover halves, +ve
expectancy. Cardinal rule: Sharpe ≫1 ⇒ audit.

Usage:
    python scripts/gold_intraday_turning.py --sweep
    python scripts/gold_intraday_turning.py --label zigzag --sweep --symbol XAUUSD
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
from src.features.trend_labels import trend_scan_labels, zigzag_labels
from scripts.backtest_meta_labeling import _build_X
from scripts.backtest_champion_baseline import TemporalCalibratedXGBoost, _load_raw

PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2, "XAUUSD": 1e-1}
COMM_PIPS = 0.5
SPLIT = pd.Timestamp("2022-01-01")
MID = pd.Timestamp("2019-01-01")          # discover sub-half split
ATR_N = 14
RISK_PCT = 0.01                            # 1% of equity risked per trade (for DD/equity only)
K_SL = 1.0                                 # SL = 1×ATR (1R)
RR = [1.5, 2.0, 3.0]                       # TP multiples of ATR (reward:risk)
# tf -> (resample rule | None, WF train/step/test days, force-close horizon bars)
TFCFG = {
    "4H":  ("4h",    (540, 90, 90), 6),
    "2H":  ("2h",    (540, 90, 90), 8),
    "1H":  ("1h",    (540, 90, 90), 10),
    "30M": ("30min", (365, 90, 90), 12),
    "15M": (None,    (365, 90, 90), 16),
}
THRESHOLDS = [0.55, 0.60, 0.65, 0.70]


def _resample(sym, rule):
    d = _load_raw(ROOT / "data" / f"{sym}_M15_long.csv")
    if rule is None:
        df = d[["open", "high", "low", "close", "tick_volume", "spread"]].copy()
    else:
        o = d.resample(rule, label="left", closed="left")
        df = pd.DataFrame({
            "open": o["open"].first(), "high": o["high"].max(), "low": o["low"].min(),
            "close": o["close"].last(), "tick_volume": o["tick_volume"].sum(),
            "spread": o["spread"].mean(),
        }).dropna(subset=["open"])
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


def _bpy(idx):
    span = (idx[-1] - idx[0]).days / 365.25
    return len(idx) / span if span > 0 else len(idx)


def _wf_pup(df, label_kind):
    """Expanding WF → concatenated OOS P(up-trend) series (leak-free)."""
    cfg = PipelineConfig(label_horizon=1, label_threshold=0.0, encoder_enabled=False)
    parts = []
    for tr_start, tr_end, te_end in _folds(df.index, *_WF):
        df_tr = df[(df.index >= tr_start) & (df.index < tr_end)].copy()
        df_full = df[(df.index >= tr_start) & (df.index < te_end)].copy()
        if len(df_tr) < 300:
            continue
        if label_kind == "zigzag":
            lab = zigzag_labels(df_tr["high"], df_tr["low"], df_tr["close"], df_tr["atr"], k=3.0)
        else:
            lab = trend_scan_labels(df_tr["close"], h_min=4, h_max=24, t_thresh=2.0)
        try:
            _, Xf, cols = _build_X(PredictorPipeline(cfg), df_tr, df_full)
        except Exception as e:
            print(f"    fold {tr_end.date()}: features failed ({e})"); continue
        y = lab.reindex(Xf.index)
        m = (Xf.index < tr_end) & y.notna() & (y != 0)
        Xtr, ytr = Xf[m], (y[m] > 0).astype(int)
        if len(Xtr) < 150 or ytr.nunique() < 2:
            continue
        mod = TemporalCalibratedXGBoost(); mod.train(Xtr, ytr)
        te_m = (Xf.index >= tr_end) & (Xf.index < te_end)
        Xte = Xf[te_m]
        if len(Xte) < 10:
            continue
        proba = mod.predict_proba(Xte)
        gi = list(mod._classes).index(1) if 1 in mod._classes else proba.shape[1] - 1
        parts.append(pd.Series(proba[:, gi], index=Xte.index))
    if not parts:
        return pd.Series(dtype=float)
    pup = pd.concat(parts).sort_index()
    return pup[~pup.index.duplicated()]


def _simulate(pup, df, pip, thr, horizon, k_tp):
    """Enter at high-prob turning point, ATR triple-barrier exit. Returns R-multiple per trade."""
    idx = list(df.index)
    pos = {ts: i for i, ts in enumerate(idx)}
    R = {}
    for ts, p in pup.items():
        d = 1 if p >= thr else (-1 if p <= 1 - thr else 0)
        if d == 0:
            continue
        i = pos.get(ts)
        atr = df["atr"].iloc[i] if i is not None else np.nan
        if i is None or not np.isfinite(atr) or atr <= 0 or i + horizon >= len(idx):
            continue
        entry = df["close"].iloc[i]
        sl_p, tp_p = K_SL * atr, k_tp * atr
        cost_R = ((df["spread"].iloc[i] + 2 * COMM_PIPS) * pip) / sl_p
        r = None
        for j in range(i + 1, i + 1 + horizon):
            hi, lo = df["high"].iloc[j], df["low"].iloc[j]
            if d == 1:
                if lo <= entry - sl_p: r = -1.0; break
                if hi >= entry + tp_p: r = k_tp / K_SL; break
            else:
                if hi >= entry + sl_p: r = -1.0; break
                if lo <= entry - tp_p: r = k_tp / K_SL; break
        if r is None:
            r = (d * (df["close"].iloc[i + horizon] - entry)) / sl_p
        R[ts] = r - cost_R
    return pd.Series(R).sort_index()


def _stat(r, bpy):
    if len(r) < 30 or r.std(ddof=1) == 0:
        return None
    # trades-per-year for annualization (trade-frequency, not bar-frequency)
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    tpy = len(r) / yrs if yrs > 0 else len(r)
    return float(r.mean() / r.std(ddof=1) * np.sqrt(tpy))


def run(symbol, label_kind, thresholds):
    global _WF
    pip = PIP[symbol]
    print(f"\n{'='*100}\n  INTRADAY TURNING-POINT — {symbol}  label={label_kind}  "
          f"SL=1×ATR  TP∈{{{','.join(str(x) for x in RR)}}}×ATR  (enter at high-prob turning point)\n{'='*100}")
    print(f"  {'TF':4} {'thr':>4} {'R:R':>4} {'trades':>7} {'win%':>5} {'avgR':>6} "
          f"{'FULL':>6} {'disc':>6} {'conf':>6} {'CI':>16} {'DD%':>5}  GO", flush=True)
    fmt = lambda v: f"{v:+.2f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "  n/a"

    for tf, (rule, wf, horizon) in TFCFG.items():
        _WF = wf
        df = _resample(symbol, rule)
        t0 = time.time()
        pup = _wf_pup(df, label_kind)
        if pup.empty:
            print(f"  {tf:4}  no folds"); continue
        for thr in thresholds:
            for k_tp in RR:
                r = _simulate(pup, df, pip, thr, horizon, k_tp)
                if len(r) < 30:
                    continue
                rd, rc = r[r.index < SPLIT], r[r.index >= SPLIT]
                d1, d2 = rd[rd.index < MID], rd[rd.index >= MID]
                bpy = _bpy(df.index)
                sh, shd, shc = _stat(r, bpy), _stat(rd, bpy), _stat(rc, bpy)
                shd1, shd2 = _stat(d1, bpy), _stat(d2, bpy)
                tpy = len(rc) / max((rc.index[-1] - rc.index[0]).days / 365.25, 1e-9) if len(rc) else 0
                lo, hi = block_bootstrap_sharpe(rc.values, block=10, ppy=int(max(tpy, 2))) if len(rc) >= 30 else (np.nan, np.nan)
                eq = (1 + RISK_PCT * r).cumprod()           # 1% risk/trade → realistic equity
                dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
                avgR = r.mean(); win = (r > 0).mean() * 100
                go = (shc is not None and shc >= 0.5 and not np.isnan(lo) and lo > 0
                      and avgR > 0 and (shd1 or -9) > 0 and (shd2 or -9) > 0)
                trd_yr = len(r) / ((r.index[-1] - r.index[0]).days / 365.25)
                print(f"  {tf:4} {thr:>4.2f} 1:{k_tp:<3g} {int(trd_yr):>5}/y {win:>4.0f}% {avgR:>+6.3f} "
                      f"{fmt(sh)} {fmt(shd)} {fmt(shc)} [{fmt(lo)},{fmt(hi)}] {dd:>4.0f}%"
                      f"{'  ✅' if go else ''}", flush=True)
        print(f"    ({tf} WF {(time.time()-t0)/60:.1f} min)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD", choices=list(PIP))
    ap.add_argument("--label", default="trend_scan", choices=["trend_scan", "zigzag"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--thresholds", type=float, nargs="+", default=None)
    args = ap.parse_args()
    thr = THRESHOLDS if args.sweep else (args.thresholds or THRESHOLDS)
    run(args.symbol, args.label, thr)
    print("\nDone.")


if __name__ == "__main__":
    main()
