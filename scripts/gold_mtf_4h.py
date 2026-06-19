"""gold_mtf_4h.py — does adding 15M/30M/1H/2H features improve the 4H gold trend prediction?

Target = 4H turning-point/trend direction (trend_scan label). Two models per fold, side-by-side:
  4H-only   : leak-free 4H engineered features (encoder OFF)
  4H+MTF    : same + lower-TF features (src/features/mtf_features, point-in-time aggregated)
Reports BOTH prediction quality (OOS ROC-AUC / accuracy of P(up) vs realized trend) AND trading
(flip ls/lfs/ls_atr vs EWMAC + buy-and-hold + $10k), so the MTF contribution is isolated.
Run --period all (2015–26, discover<2022/confirm≥2022) and --period 2022 (2022–26 alone, split 2024).

Leak guard: MTF features are causality-tested (tests/test_mtf_features.py). A large 4H+MTF jump
over 4H-only ⇒ audit before believing (this is where the +3.14 enc8 champion leaked).

Usage: python scripts/gold_mtf_4h.py --period all   |   --period 2022
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import PredictorPipeline, PipelineConfig
from src.cta.bootstrap import block_bootstrap_sharpe
from src.cta.signals import ewmac
from src.features.trend_labels import trend_scan_labels
from src.features.mtf_features import mtf_features
from scripts.backtest_meta_labeling import _build_X
from scripts.backtest_champion_baseline import TemporalCalibratedXGBoost
from scripts.honest_champion_h4 import _resample, _folds, _bars_per_year
from scripts.gold_trend_predictor import _positions, _pnl, PIP, COMM_PIPS
from scripts.gold_intraday_turning import _simulate as _barrier_sim   # ATR triple-barrier (SL=1×ATR)

SYM = "XAUUSD"
THRESH = [0.55, 0.60, 0.65]
RR_SLTP = [1.5, 2.0, 3.0]      # TP multiples of ATR for the SL/TP variant
H_4H = 6                       # 4H force-close horizon (bars), matches intraday harness
PERIODS = {  # start, WF(train/step/test days), confirm-split, discover-mid
    "all":  (pd.Timestamp("2015-01-01"), (540, 90, 90), pd.Timestamp("2022-01-01"), pd.Timestamp("2019-01-01")),
    "2022": (pd.Timestamp("2022-01-01"), (365, 60, 60), pd.Timestamp("2024-01-01"), pd.Timestamp("2023-01-01")),
}
fmt = lambda v: f"{v:+.2f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else " n/a"


def _metrics(net, bpy, split, mid):
    if len(net) < 40:
        return {}
    ann = np.sqrt(bpy)
    s = lambda x: float(x.mean() / x.std(ddof=1) * ann) if len(x) > 20 and x.std(ddof=1) > 0 else None
    rd, rc = net[net.index < split], net[net.index >= split]
    d1, d2 = rd[rd.index < mid], rd[rd.index >= mid]
    lo, hi = block_bootstrap_sharpe(rc.values, block=10, ppy=int(bpy)) if len(rc) >= 30 else (np.nan, np.nan)
    eq = (1 + 0.0 + net).cumprod(); dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
    return dict(full=s(net), disc=s(rd), d1=s(d1), d2=s(d2), conf=s(rc), lo=lo, hi=hi, dd=dd,
                win=(net > 0).mean() * 100)


def _rstat(r, split, mid):
    """R-unit per-trade metrics (trade-frequency annualization) for the SL/TP variant."""
    if len(r) < 30:
        return {}
    def sh(x):
        if len(x) < 20 or x.std(ddof=1) == 0:
            return None
        yrs = (x.index[-1] - x.index[0]).days / 365.25
        tpy = len(x) / yrs if yrs > 0 else len(x)
        return float(x.mean() / x.std(ddof=1) * np.sqrt(tpy))
    rd, rc = r[r.index < split], r[r.index >= split]
    d1, d2 = rd[rd.index < mid], rd[rd.index >= mid]
    tpy = len(rc) / max((rc.index[-1] - rc.index[0]).days / 365.25, 1e-9) if len(rc) else 0
    lo, hi = block_bootstrap_sharpe(rc.values, block=10, ppy=int(max(tpy, 2))) if len(rc) >= 30 else (np.nan, np.nan)
    eq = (1 + 0.01 * r).cumprod(); dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
    return dict(full=sh(r), disc=sh(rd), d1=sh(d1), d2=sh(d2), conf=sh(rc), lo=lo, hi=hi,
                dd=dd, avgR=r.mean(), win=(r > 0).mean() * 100, n=len(r))


def _wf(df, mtf_df, period):
    """One WF pass → P(up) for BOTH 4H-only and 4H+MTF, plus the realized trend label on OOS."""
    cfg = PipelineConfig(label_horizon=1, label_threshold=0.0, encoder_enabled=False)
    _, wf, _, _ = PERIODS[period]
    base_parts, mtf_parts = [], []
    for tr_start, tr_end, te_end in _folds(df.index, *wf):
        df_tr = df[(df.index >= tr_start) & (df.index < tr_end)].copy()
        df_full = df[(df.index >= tr_start) & (df.index < te_end)].copy()
        if len(df_tr) < 250:
            continue
        lab = trend_scan_labels(df_tr["close"], h_min=4, h_max=24, t_thresh=2.0)
        try:
            _, Xf, cols = _build_X(PredictorPipeline(cfg), df_tr, df_full)
        except Exception as e:
            print(f"    fold {tr_end.date()}: features failed ({e})"); continue
        Xm = Xf.join(mtf_df.reindex(Xf.index)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = lab.reindex(Xf.index)
        m = (Xf.index < tr_end) & y.notna() & (y != 0)
        ytr = (y[m] > 0).astype(int)
        if m.sum() < 150 or ytr.nunique() < 2:
            continue
        te = (Xf.index >= tr_end) & (Xf.index < te_end)
        if te.sum() < 10:
            continue
        for Xsrc, store in ((Xf, base_parts), (Xm, mtf_parts)):
            mod = TemporalCalibratedXGBoost(); mod.train(Xsrc[m], ytr)
            pr = mod.predict_proba(Xsrc[te])
            gi = list(mod._classes).index(1) if 1 in mod._classes else pr.shape[1] - 1
            store.append(pd.Series(pr[:, gi], index=Xsrc.index[te]))
    pb = pd.concat(base_parts).sort_index() if base_parts else pd.Series(dtype=float)
    pm = pd.concat(mtf_parts).sort_index() if mtf_parts else pd.Series(dtype=float)
    return pb[~pb.index.duplicated()], pm[~pm.index.duplicated()]


def _pred_quality(pup, df, split):
    """OOS ROC-AUC / accuracy of P(up) vs realized trend label, full and confirm."""
    lab = trend_scan_labels(df["close"], h_min=4, h_max=24, t_thresh=2.0).reindex(pup.index)
    m = lab.notna() & (lab != 0)
    y, p = (lab[m] > 0).astype(int), pup[m]
    def qa(yy, pp):
        if len(yy) < 30 or yy.nunique() < 2:
            return (float("nan"), float("nan"))
        return roc_auc_score(yy, pp), ((pp > 0.5).astype(int) == yy).mean()
    cf = y.index >= split
    return qa(y, p), qa(y[cf], p[cf])


def run(period, exit_mode="both"):
    start, wf, split, mid = PERIODS[period]
    df = _resample(SYM, "H4")
    df = df[df.index >= start]
    ret = df["close"].pct_change()
    bpy = _bars_per_year(df.index)
    mtf_df = mtf_features(SYM)            # leak-free lower-TF features on the 4H grid
    print(f"\n{'='*96}\n  GOLD 4H MTF TREND — period={period}  ({len(df):,} 4H bars "
          f"{df.index[0].date()}→{df.index[-1].date()})  confirm≥{split.date()}\n{'='*96}", flush=True)

    t0 = time.time()
    pb, pm = _wf(df, mtf_df, period)
    if pb.empty:
        print("  no folds"); return
    print(f"  WF {(time.time()-t0)/60:.1f} min — OOS {pb.index[0].date()}→{pb.index[-1].date()} ({len(pb)} bars)\n")

    # ── prediction quality ──
    (ab_f, acc_b_f), (ab_c, acc_b_c) = _pred_quality(pb, df, split)
    (am_f, acc_m_f), (am_c, acc_m_c) = _pred_quality(pm, df, split)
    print("  PREDICTION (ROC-AUC / accuracy of P(up) vs realized 4H trend)")
    print(f"    4H-only : AUC full={ab_f:.3f} conf={ab_c:.3f}   acc full={acc_b_f:.3f} conf={acc_b_c:.3f}")
    print(f"    4H+MTF  : AUC full={am_f:.3f} conf={am_c:.3f}   acc full={acc_m_f:.3f} conf={acc_m_c:.3f}")
    print(f"    Δ MTF   : AUC {am_f-ab_f:+.3f} (full) / {am_c-ab_c:+.3f} (conf)\n", flush=True)

    # ── benchmarks ──
    ew = (ewmac(df[["close"]].rename(columns={"close": SYM}))[SYM] / 10.0).clip(-1, 1).reindex(pb.index)
    bh = pd.Series(1.0, index=pb.index)
    mb = _metrics(_pnl(ew, ret, df["close"], df["spread"]), bpy, split, mid)
    mh = _metrics(_pnl(bh, ret, df["close"], df["spread"]), bpy, split, mid)
    print("  TRADING (net Sharpe full / confirm[CI])")
    print(f"    {'EWMAC':16} full={fmt(mb['full'])} conf={fmt(mb['conf'])}[{fmt(mb['lo'])},{fmt(mb['hi'])}]")
    print(f"    {'BUY&HOLD':16} full={fmt(mh['full'])} conf={fmt(mh['conf'])}[{fmt(mh['lo'])},{fmt(mh['hi'])}]")
    bh_conf = mh["conf"]
    if exit_mode in ("flip", "both"):
        for tag, pup in (("4H-only", pb), ("4H+MTF", pm)):
            for mode in ("ls", "ls_atr"):
                for thr in THRESH:
                    pos = _positions(pup, df["close"], df["atr"], thr, mode)
                    m = _metrics(_pnl(pos, ret, df["close"], df["spread"]), bpy, split, mid)
                    if not m:
                        continue
                    go = (m["conf"] and m["conf"] >= 0.5 and not np.isnan(m["lo"]) and m["lo"] > 0
                          and (m["d1"] or -9) > 0 and (m["d2"] or -9) > 0 and m["conf"] > (bh_conf or -9))
                    print(f"    {tag:8} {mode:6} thr={thr:.2f}  full={fmt(m['full'])} "
                          f"disc={fmt(m['disc'])} conf={fmt(m['conf'])}[{fmt(m['lo'])},{fmt(m['hi'])}] "
                          f"DD={m['dd']:.0f}%{'  ✅>B&H' if go else ''}", flush=True)

    # ── SL/TP ATR triple-barrier (4H alone): SL=1×ATR, TP swept, 6-bar force-close ──
    if exit_mode in ("sltp", "both"):
        print(f"\n  SL/TP ATR triple-barrier — 4H (SL=1×ATR, TP∈{{{','.join(str(x) for x in RR_SLTP)}}}×ATR, "
              f"{H_4H}-bar exit; R-unit net Sharpe)")
        print(f"    {'model':8} {'thr':>4} {'R:R':>5} {'trd/y':>5} {'win%':>5} {'avgR':>6} "
              f"{'full':>6} {'disc':>6} {'conf':>6} {'CI':>16} {'DD%':>5}  GO")
        for tag, pup in (("4H-only", pb), ("4H+MTF", pm)):
            for thr in THRESH:
                for k_tp in RR_SLTP:
                    r = _barrier_sim(pup, df, PIP, thr, H_4H, k_tp)
                    m = _rstat(r, split, mid)
                    if not m:
                        continue
                    go = (m["conf"] and m["conf"] >= 0.5 and not np.isnan(m["lo"]) and m["lo"] > 0
                          and (m["d1"] or -9) > 0 and (m["d2"] or -9) > 0 and m["avgR"] > 0)
                    trd_yr = m["n"] / max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
                    print(f"    {tag:8} {thr:>4.2f} 1:{k_tp:<3g} {int(trd_yr):>4}/y {m['win']:>4.0f}% "
                          f"{m['avgR']:>+6.3f} {fmt(m['full'])} {fmt(m['disc'])} {fmt(m['conf'])}"
                          f"[{fmt(m['lo'])},{fmt(m['hi'])}] {m['dd']:>4.0f}%{'  ✅' if go else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="all", choices=list(PERIODS))
    ap.add_argument("--exit", dest="exit_mode", default="both", choices=["flip", "sltp", "both"])
    args = ap.parse_args()
    run(args.period, args.exit_mode)
    print("\nDone.")


if __name__ == "__main__":
    main()
