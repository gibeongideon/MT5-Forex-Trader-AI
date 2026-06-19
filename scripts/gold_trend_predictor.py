"""gold_trend_predictor.py — GOLD (XAUUSD) trend-direction / turning-point predictor (H4).

Classify the current trend direction (up/down), enter at the turning point ("the tip"), ride it,
flip when the opposite direction is predicted past a threshold. Honest, leak-free, benchmarked
against mechanical EWMAC trend AND vol-targeted buy-and-hold gold — the model must BEAT both to
earn its place (single-asset gold trend ≈ buy-and-hold beta, so beating B&H is the real bar).

  label   : trend_scan (López de Prado, default) | zigzag (ATR pivots)        [target only, forward]
  features: leak-free engineered set (encoder OFF) via backtest_meta_labeling._build_X
  model   : TemporalCalibratedXGBoost → P(up-trend), expanding WF (540d/90d/90d)
  modes   : ls (long↔short, always-in) | lfs (long/flat/short) | ls_atr (ls + m×ATR stop)
  eval    : threshold(band) sweep + discover(<2022)/confirm(≥2022) + block-bootstrap CI
  GO      : confirm Sharpe ≥ +0.5 AND CI lower > 0 AND both discover halves > 0 AND > EWMAC AND > B&H

Usage:
    python scripts/gold_trend_predictor.py --sweep
    python scripts/gold_trend_predictor.py --label zigzag --sweep
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
from src.cta.signals import ewmac
from src.features.trend_labels import trend_scan_labels, zigzag_labels
from scripts.backtest_meta_labeling import _build_X
from scripts.backtest_champion_baseline import TemporalCalibratedXGBoost
from scripts.honest_champion_h4 import _resample, _folds, _bars_per_year

PIP = 1e-1                     # gold: 1 pip = 0.1 price
COMM_PIPS = 0.5
SPLIT = pd.Timestamp("2022-01-01")
WF = (540, 90, 90)            # H4 expanding WF (train/step/test days)
MODES = ["ls", "lfs", "ls_atr"]
ATR_STOP_MULT = 3.0


def _positions(pup: pd.Series, close: pd.Series, atr: pd.Series, thr: float,
               mode: str) -> pd.Series:
    """State machine over the OOS bars. thr defines the band [1-thr, thr]."""
    pos = np.zeros(len(pup))
    cur = 0.0
    entry = np.nan
    cl = close.reindex(pup.index).values
    av = atr.reindex(pup.index).values
    pv = pup.values
    for i in range(len(pv)):
        p = pv[i]
        want = 1.0 if p >= thr else (-1.0 if p <= 1 - thr else None)
        if mode == "lfs":
            cur = want if want is not None else 0.0
        elif mode == "ls":
            if want is not None:
                cur = want
        elif mode == "ls_atr":
            if want is not None and want != cur:
                cur = want; entry = cl[i]
            elif cur != 0.0 and np.isfinite(entry) and np.isfinite(av[i]):
                adverse = (entry - cl[i]) if cur > 0 else (cl[i] - entry)
                if adverse >= ATR_STOP_MULT * av[i]:
                    cur = 0.0; entry = np.nan
        if cur != 0.0 and not np.isfinite(entry):
            entry = cl[i]
        pos[i] = cur
    return pd.Series(pos, index=pup.index)


def _pnl(pos: pd.Series, ret: pd.Series, close: pd.Series, spread: pd.Series) -> pd.Series:
    held = pos.shift(1).fillna(0.0)
    cost_rate = (spread.reindex(pos.index) * PIP + COMM_PIPS * PIP) / close.reindex(pos.index)
    turn = (pos - pos.shift(1)).abs().fillna(0.0)
    return (held * ret.reindex(pos.index) - turn * cost_rate).dropna()


def _metrics(net: pd.Series, pos: pd.Series, bpy: float) -> dict:
    if len(net) < 60:
        return {}
    ann = np.sqrt(bpy)
    f = lambda x: float(x.mean() / x.std(ddof=1) * ann) if len(x) > 30 and x.std(ddof=1) > 0 else float("nan")
    rd, rc = net[net.index < SPLIT], net[net.index >= SPLIT]
    d1 = rd[rd.index < pd.Timestamp("2019-01-01")]; d2 = rd[rd.index >= pd.Timestamp("2019-01-01")]
    lo, hi = block_bootstrap_sharpe(rc.values, block=10, ppy=int(bpy)) if len(rc) >= 30 else (np.nan, np.nan)
    eq = (1 + net).cumprod(); dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
    turn = float((pos - pos.shift(1)).abs().sum() / ((net.index[-1] - net.index[0]).days / 365.25))
    return dict(full=f(net), disc=f(rd), d1=f(d1), d2=f(d2), conf=f(rc),
                lo=lo, hi=hi, dd=dd, win=(net > 0).mean() * 100, turn=turn)


def run(label_kind: str, thresholds):
    sym = "XAUUSD"
    df = _resample(sym, "H4")
    ret = df["close"].pct_change()
    bpy = _bars_per_year(df.index)
    print(f"\n{'='*90}\n  GOLD TREND PREDICTOR — XAUUSD H4  ({len(df):,} bars "
          f"{df.index[0].date()}→{df.index[-1].date()})  label={label_kind}")
    print(f"  enter at the tip, ride trend, flip on opposite signal ≥ thr | vs EWMAC & buy-and-hold\n{'='*90}", flush=True)

    cfg = PipelineConfig(label_horizon=1, label_threshold=0.0, encoder_enabled=False)
    folds = _folds(df.index, *WF)
    pup_parts, t0 = [], time.time()
    for tr_start, tr_end, te_end in folds:
        df_tr = df[(df.index >= tr_start) & (df.index < tr_end)].copy()
        df_full = df[(df.index >= tr_start) & (df.index < te_end)].copy()
        if len(df_tr) < 300:
            continue
        # leak-free target (computed on TRAIN window only → forward window stays in-sample)
        if label_kind == "zigzag":
            lab = zigzag_labels(df_tr["high"], df_tr["low"], df_tr["close"], df_tr["atr"], k=3.0)
        else:
            lab = trend_scan_labels(df_tr["close"], h_min=4, h_max=24, t_thresh=2.0)
        try:
            _, Xf, cols = _build_X(PredictorPipeline(cfg), df_tr, df_full)
        except Exception as e:
            print(f"  fold {tr_end.date()}: feature build failed ({e})"); continue
        y = lab.reindex(Xf.index)
        tr_mask = (Xf.index < tr_end) & y.notna() & (y != 0)
        Xtr, ytr = Xf[tr_mask], (y[tr_mask] > 0).astype(int)
        if len(Xtr) < 150 or ytr.nunique() < 2:
            continue
        m = TemporalCalibratedXGBoost(); m.train(Xtr, ytr)
        te_mask = (Xf.index >= tr_end) & (Xf.index < te_end)
        Xte = Xf[te_mask]
        if len(Xte) < 10:
            continue
        proba = m.predict_proba(Xte)
        gi = list(m._classes).index(1) if 1 in m._classes else proba.shape[1] - 1
        pup_parts.append(pd.Series(proba[:, gi], index=Xte.index))
    if not pup_parts:
        print("  no folds"); return
    pup = pd.concat(pup_parts).sort_index()
    pup = pup[~pup.index.duplicated()]
    oos = pup.index
    print(f"  WF done {(time.time()-t0)/60:.1f} min — OOS bars={len(oos)} "
          f"({oos[0].date()}→{oos[-1].date()})\n", flush=True)

    fmt = lambda v: f"{v:+.2f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else " n/a"

    # ── benchmarks on the same OOS window/cost ──
    ew = (ewmac(df[["close"]].rename(columns={"close": sym}))[sym] / 10.0).clip(-1, 1).reindex(oos)
    bench = {}
    bench["EWMAC"] = _metrics(_pnl(ew, ret, df["close"], df["spread"]), ew, bpy)
    bh = pd.Series(1.0, index=oos)
    bench["BUY&HOLD"] = _metrics(_pnl(bh, ret, df["close"], df["spread"]), bh, bpy)
    for nm, m in bench.items():
        print(f"  [BENCH {nm:9}] FULL={fmt(m['full'])} disc={fmt(m['disc'])} conf={fmt(m['conf'])}"
              f"[{fmt(m['lo'])},{fmt(m['hi'])}] win={m['win']:.0f}% DD={m['dd']:.0f}% turn={m['turn']:.0f}/yr")
    ew_conf, bh_conf = bench["EWMAC"]["conf"], bench["BUY&HOLD"]["conf"]
    print()

    # ── $10,000 account simulation (best active model vs buy-and-hold) ──
    def equity_report(net, label, e0=10000.0):
        net = net.dropna()
        if len(net) < 60:
            print(f"  {label}: too few bars"); return
        yrs = (net.index[-1] - net.index[0]).days / 365.25
        eq = e0 * (1 + net).cumprod()
        cagr = (eq.iloc[-1] / e0) ** (1 / yrs) - 1
        dd = float(((eq.cummax() - eq) / eq.cummax()).max())
        cn = net[net.index >= SPLIT]
        cyrs = (cn.index[-1] - cn.index[0]).days / 365.25
        ccagr = (e0 * (1 + cn).cumprod()).iloc[-1] / e0
        ccagr = ccagr ** (1 / cyrs) - 1
        sh = float(net.mean() / net.std(ddof=1) * np.sqrt(bpy))
        csh = float(cn.mean() / cn.std(ddof=1) * np.sqrt(bpy))
        print(f"  {label:20} Sharpe full={sh:+.2f} confirm={csh:+.2f}  "
              f"CAGR={cagr*100:+.1f}%/yr → ${eq.iloc[-1]:,.0f} on $10k after {yrs:.1f}y  "
              f"maxDD=-${e0*dd:,.0f} ({dd*100:.0f}%)  | confirm CAGR={ccagr*100:+.1f}%/yr")
        yr = (1 + net).groupby(net.index.year).apply(lambda x: x.prod() - 1) * 100
        print("     per-year %: " + "  ".join(f"{y}:{v:+.0f}" for y, v in yr.items()))

    print(f"\n  ── $10,000 ACCOUNT SIMULATION (gold, full notional ±1) ──")
    pos_m = _positions(pup, df["close"], df["atr"], 0.55, "ls_atr")
    equity_report(_pnl(pos_m, ret, df["close"], df["spread"]), "MODEL ls_atr@0.55")
    equity_report(_pnl(bh, ret, df["close"], df["spread"]), "BUY&HOLD gold")
    print()

    # ── model: modes × thresholds ──
    for mode in MODES:
        for thr in thresholds:
            pos = _positions(pup, df["close"], df["atr"], thr, mode)
            m = _metrics(_pnl(pos, ret, df["close"], df["spread"]), pos, bpy)
            if not m:
                continue
            go = (m["conf"] is not None and m["conf"] >= 0.5 and not np.isnan(m["lo"]) and m["lo"] > 0
                  and (m["d1"] or -9) > 0 and (m["d2"] or -9) > 0
                  and m["conf"] > (ew_conf or -9) and m["conf"] > (bh_conf or -9))
            beat = "  ".join([f"{'>EWMAC' if (m['conf'] or -9) > (ew_conf or -9) else '<ewmac'}",
                              f"{'>B&H' if (m['conf'] or -9) > (bh_conf or -9) else '<b&h'}"])
            print(f"  {mode:6} thr={thr:.2f}  FULL={fmt(m['full'])} disc={fmt(m['disc'])} "
                  f"conf={fmt(m['conf'])}[{fmt(m['lo'])},{fmt(m['hi'])}]  win={m['win']:.0f}% "
                  f"DD={m['dd']:.0f}% turn={m['turn']:.0f}/yr  {beat}{'  ✅GO' if go else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="trend_scan", choices=["trend_scan", "zigzag"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--thresholds", type=float, nargs="+", default=None)
    args = ap.parse_args()
    thr = [0.55, 0.60, 0.65] if args.sweep else (args.thresholds or [0.55, 0.60, 0.65])
    run(args.label, thr)
    print("\nDone.")


if __name__ == "__main__":
    main()
