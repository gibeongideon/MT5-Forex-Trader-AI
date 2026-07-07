"""XAUUSD H4 pattern exploration (V5 Track 0, informative/non-promotable).

Characterizes the structure the sizing program will exploit, so feature/label
choices are evidence-led rather than assumed. Read-only: no trading decisions
change here. Writes data/v5_runs/xau-eda/{report.json,report.md}.

Metrics:
  1. Trend run-length distribution (consecutive same-sign forecast bars).
  2. Forecast autocorrelation & half-life.
  3. Volatility clustering: ACF of |log returns|.
  4. Conditional trend-reversal base rates by (|forecast| bucket x vol regime
     x session) over horizon K.
  5. Conditional trade win-rate map over the engine's own trades.

    conda run -n envmt5 python scripts/v5_xau_eda.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v5.xau_trend import confidence_bucket, run_trades, wilder_atr, xau_signal

DATA = ROOT / "data" / "XAUUSD_H4_long.csv"
OUT = ROOT / "data" / "v5_runs" / "xau-eda"
CONF_RISK = {"low": 0.5, "med": 1.0, "high": 1.5}


def _session(hour: int) -> str:
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 13:
        return "london"
    if 13 <= hour < 17:
        return "london_ny"
    return "ny"


def run_lengths(sign: pd.Series) -> dict:
    s = sign[sign != 0].values
    if len(s) == 0:
        return {}
    runs, cur = [], 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    runs = np.array(runs)
    return {"n_runs": int(len(runs)), "mean_bars": round(float(runs.mean()), 2),
            "median_bars": int(np.median(runs)), "p90_bars": int(np.percentile(runs, 90)),
            "max_bars": int(runs.max())}


def autocorr_halflife(series: pd.Series, max_lag: int = 60) -> dict:
    x = series.dropna().values
    if len(x) < max_lag + 10:
        return {}
    x = x - x.mean()
    denom = np.dot(x, x)
    acf = [float(np.dot(x[:-k], x[k:]) / denom) for k in range(1, max_lag + 1)]
    hl = next((k + 1 for k, a in enumerate(acf) if a < 0.5), None)
    return {"acf_lag1": round(acf[0], 3), "acf_lag6": round(acf[5], 3),
            "acf_lag24": round(acf[23], 3),
            "half_life_bars": hl, "max_lag": max_lag}


def reversal_table(df: pd.DataFrame, sig: pd.Series, atr: pd.Series,
                   horizon: int) -> list[dict]:
    sign = np.sign(sig)
    fut_sign = sign.shift(-horizon)
    reversal = ((sign != 0) & (fut_sign != 0) & (sign != fut_sign)).astype(float)
    reversal[fut_sign.isna() | (sign == 0)] = np.nan
    conf = sig.abs().map(lambda v: confidence_bucket(v) if np.isfinite(v) else None)
    vol_pct = (atr / df["close"]).rolling(1560, min_periods=390).rank(pct=True)
    vol_bucket = pd.cut(vol_pct, [0, 1 / 3, 2 / 3, 1.0],
                        labels=["lo", "mid", "hi"], include_lowest=True)
    session = pd.Series(df.index.hour, index=df.index).map(_session)
    tab = pd.DataFrame({"reversal": reversal, "conf": conf,
                        "vol": vol_bucket.astype(object), "session": session}).dropna()
    rows = []
    for (c, v, s), g in tab.groupby(["conf", "vol", "session"], observed=True):
        if len(g) >= 30:
            rows.append({"conf": c, "vol": v, "session": s,
                         "n": int(len(g)),
                         "reversal_rate": round(float(g["reversal"].mean()), 3)})
    return sorted(rows, key=lambda r: r["reversal_rate"], reverse=True)


def win_map(trades: pd.DataFrame) -> list[dict]:
    t = trades.dropna(subset=["r_multiple"]).copy()
    t["win"] = (t["r_multiple"] > 0).astype(int)
    t["hour"] = pd.to_datetime(t["open_time"]).dt.hour.map(_session)
    rows = []
    for (c, s), g in t.groupby(["confidence", "hour"], observed=True):
        if len(g) >= 20:
            rows.append({"confidence": c, "session": s, "n": int(len(g)),
                         "win_rate": round(float(g["win"].mean()), 3),
                         "mean_r": round(float(g["r_multiple"].mean()), 3)})
    return sorted(rows, key=lambda r: r["win_rate"], reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--horizon", type=int, default=6, help="H4 bars for reversal label")
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    sig = xau_signal(df["close"])
    atr = wilder_atr(df, 14)
    rets = np.log(df["close"]).diff()

    trades = run_trades(df, exit_mode="trail", flip_mode="confidence",
                        params={"conf_risk_scale": CONF_RISK})["trades"]

    report = {
        "data": Path(args.data).name,
        "bars": int(len(df)),
        "date_range": [str(df.index[0]), str(df.index[-1])],
        "engine_trades": int(len(trades)),
        "reversal_horizon_bars": args.horizon,
        "trend_run_length": run_lengths(np.sign(sig)),
        "forecast_autocorr": autocorr_halflife(sig),
        "abs_return_autocorr": autocorr_halflife(rets.abs()),
        "reversal_base_rate_overall": None,
        "reversal_by_state": reversal_table(df, sig, atr, args.horizon),
        "trade_win_map": win_map(trades),
    }
    # overall reversal base rate
    sign = np.sign(sig)
    fut = sign.shift(-args.horizon)
    mask = (sign != 0) & fut.notna() & (fut != 0)
    report["reversal_base_rate_overall"] = round(
        float((sign[mask] != fut[mask]).mean()), 3)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [f"# XAUUSD H4 Pattern EDA", "",
             f"- bars: {report['bars']}  range: {report['date_range'][0]} → {report['date_range'][1]}",
             f"- engine trades: {report['engine_trades']}",
             f"- overall {args.horizon}-bar trend-reversal base rate: "
             f"{report['reversal_base_rate_overall']}",
             f"- trend run-length: {report['trend_run_length']}",
             f"- forecast autocorr: {report['forecast_autocorr']}",
             f"- |return| autocorr (vol clustering): {report['abs_return_autocorr']}",
             "", "## Highest reversal-rate states (top 10)", ""]
    for r in report["reversal_by_state"][:10]:
        lines.append(f"- conf={r['conf']} vol={r['vol']} session={r['session']} "
                     f"n={r['n']} reversal={r['reversal_rate']}")
    lines += ["", "## Trade win-rate map (top 10 by win rate)", ""]
    for r in report["trade_win_map"][:10]:
        lines.append(f"- conf={r['confidence']} session={r['session']} n={r['n']} "
                     f"win={r['win_rate']} meanR={r['mean_r']}")
    (OUT / "report.md").write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nwrote {OUT/'report.json'} and {OUT/'report.md'}")


if __name__ == "__main__":
    main()
