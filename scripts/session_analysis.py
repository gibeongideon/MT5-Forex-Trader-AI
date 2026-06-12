"""
session_analysis.py — Study candle model predictions by forex trading session.

All MT5 data is naive UTC. Display times are shown in both UTC and EAT (UTC+3,
Nairobi / East Africa Time).

Sessions (from user's EAT schedule, converted to UTC):
  Sydney        UTC 22:00–07:00  (EAT 01:00–10:00)  — spans midnight
  Tokyo         UTC 00:00–09:00  (EAT 03:00–12:00)
  London        UTC 08:00–17:00  (EAT 11:00–20:00)
  New York      UTC 13:00–22:00  (EAT 16:00–01:00)
  Tokyo+London  UTC 08:00–09:00  (EAT 11:00–12:00)  overlap
  London+NY     UTC 13:00–17:00  (EAT 16:00–20:00)  ★ best

Usage:
    conda run -n envmt5 python scripts/session_analysis.py
    conda run -n envmt5 python scripts/session_analysis.py --symbol EURUSD
    conda run -n envmt5 python scripts/session_analysis.py --plot
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.pipeline import PredictorPipeline

# ── Constants ──────────────────────────────────────────────────────────────────

EAT_OFFSET = 3   # EAT = UTC + 3

SYMBOL_CFG = {
    "EURUSD": dict(
        data_path      = "data/EURUSD_M15.csv",
        candle_model   = "data/models/candle_EURUSD",
        pip_size       = 0.0001,
    ),
    "USDJPY": dict(
        data_path      = "data/USDJPY_M15.csv",
        candle_model   = "data/models/candle_USDJPY",
        pip_size       = 0.01,
    ),
}

# Each entry: (name, utc_open, utc_close, note)
# utc_close=None means "wraps midnight" handled separately via lambda
SESSION_DEFS = [
    ("Sydney",           22, 7,  True,  "01:00–10:00"),   # midnight-crossing
    ("Tokyo",             0, 9,  False, "03:00–12:00"),
    ("London",            8, 17, False, "11:00–20:00"),
    ("New York",         13, 22, False, "16:00–01:00"),
    ("Tokyo+London ▲",   8,  9, False, "11:00–12:00"),
    ("London+NY ★",     13, 17, False, "16:00–20:00"),
    ("Nairobi Business",  6, 15, False, "09:00–18:00"),
]


def session_mask(hour: pd.Series, utc_open: int, utc_close: int,
                 midnight_crossing: bool) -> pd.Series:
    if midnight_crossing:
        return (hour >= utc_open) | (hour < utc_close)
    return (hour >= utc_open) & (hour < utc_close)


def eat_range(utc_open: int, utc_close: int, midnight_crossing: bool) -> str:
    o = (utc_open  + EAT_OFFSET) % 24
    c = (utc_close + EAT_OFFSET) % 24
    return f"{o:02d}:00–{c:02d}:00"


# ── Extra features (must exactly match train_candle_model.py) ─────────────────

def _add_extra_features(df_raw: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    idx = X.index
    hour = idx.hour
    extra = pd.DataFrame(index=idx)

    # Session flags
    extra["session_sydney"]  = ((hour >= 22) | (hour < 7)).astype(float)
    extra["session_tokyo"]   = ((hour >= 0)  & (hour < 9)).astype(float)
    extra["session_london"]  = ((hour >= 8)  & (hour < 17)).astype(float)
    extra["session_ny"]      = ((hour >= 13) & (hour < 22)).astype(float)
    extra["session_tok_lon"] = ((hour >= 8)  & (hour < 9)).astype(float)
    extra["session_lon_ny"]  = ((hour >= 13) & (hour < 17)).astype(float)
    extra["hour_sin"]        = np.sin(2 * np.pi * hour / 24)
    extra["hour_cos"]        = np.cos(2 * np.pi * hour / 24)

    close_1h   = df_raw["close"].resample("1h").last().ffill()
    ema_1h     = close_1h.ewm(span=20, adjust=False).mean()
    ema_1h_m15 = ema_1h.reindex(df_raw.index, method="ffill")
    extra["ema_1h_ratio"] = ((df_raw["close"] - ema_1h_m15) / df_raw["close"]).reindex(idx).fillna(0)
    extra["ema_1h_slope"] = (ema_1h_m15.diff(4) / df_raw["close"]).reindex(idx).fillna(0)

    close_4h   = df_raw["close"].resample("4h").last().ffill()
    ema_4h     = close_4h.ewm(span=50, adjust=False).mean()
    ema_4h_m15 = ema_4h.reindex(df_raw.index, method="ffill")
    extra["ema_4h_ratio"] = ((df_raw["close"] - ema_4h_m15) / df_raw["close"]).reindex(idx).fillna(0)
    extra["ema_4h_slope"] = (ema_4h_m15.diff(16) / df_raw["close"]).reindex(idx).fillna(0)

    return pd.concat([X, extra.reindex(idx).fillna(0)], axis=1)


# ── Core analysis ─────────────────────────────────────────────────────────────

def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def _directional_accuracy(preds: pd.DataFrame, df_raw: pd.DataFrame) -> pd.Series:
    """
    For each predicted bar, was the model right?
    A BUY is correct if close[t+1] > close[t].
    A SELL is correct if close[t+1] < close[t].
    HOLD predictions are excluded.
    Returns a boolean Series indexed like preds.
    """
    close     = df_raw["close"].reindex(preds.index)
    close_nxt = close.shift(-1)
    actual_up = (close_nxt > close)

    correct = pd.Series(index=preds.index, dtype=float)
    buy_mask  = preds["signal"] == "buy"
    sell_mask = preds["signal"] == "sell"
    correct[buy_mask]  = actual_up[buy_mask].astype(float)
    correct[sell_mask] = (~actual_up[sell_mask]).astype(float)
    return correct


def _ascii_bar(val: float, width: int = 20) -> str:
    n = int(round(val * width))
    return "█" * n + "░" * (width - n)


def run_symbol(symbol: str, do_plot: bool) -> None:
    cfg      = SYMBOL_CFG[symbol]
    model_dir = cfg["candle_model"]

    if not Path(model_dir).exists():
        print(f"  No candle model found at {model_dir} — run train_candle_model.py first")
        return

    # Load model
    print(f"  Loading model from {model_dir}...")
    pipe = PredictorPipeline.from_config()
    pipe.load(model_dir)

    # Load data
    df_raw = _load_raw(cfg["data_path"])
    print(f"  Data: {len(df_raw):,} bars  {df_raw.index[0].date()} → {df_raw.index[-1].date()}")

    # Build features
    print("  Building features...", end=" ", flush=True)
    X_base, _ = pipe._fp.build(df_raw, fit=False)
    if pipe._enc is not None:
        latent = pipe._enc.transform(df_raw)
        shared = X_base.index.intersection(latent.index)
        X = pd.concat([X_base.loc[shared], latent.loc[shared]], axis=1)
    else:
        X = X_base
    X = _add_extra_features(df_raw, X)
    for c in pipe._feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[pipe._feature_cols]
    print(f"done  ({X.shape[0]:,} rows × {X.shape[1]} features)")

    # Predict
    print("  Running predictions...", end=" ", flush=True)
    preds = pipe.predict_batch(X)
    print(f"done  buy={( preds['signal']=='buy').sum():,}  "
          f"sell={(preds['signal']=='sell').sum():,}  "
          f"hold={(preds['signal']=='hold').sum():,}")

    # Directional accuracy
    correct = _directional_accuracy(preds, df_raw)
    hour    = preds.index.hour

    # ── Session table ──────────────────────────────────────────────────────────
    W = 100
    print(f"\n{'═'*W}")
    print(f"  SESSION ANALYSIS — {symbol}   "
          f"(data: naive UTC  |  display: UTC and EAT = UTC+{EAT_OFFSET})")
    print(f"  {len(df_raw):,} bars  {df_raw.index[0].date()} → {df_raw.index[-1].date()}")
    print(f"{'═'*W}")
    hdr = (f"{'Session':<22}  {'UTC Range':>12}  {'EAT Range':>12}  "
           f"{'Bars':>7}  {'Sigs':>6}  {'Buy%':>6}  {'Sell%':>6}  "
           f"{'AvgConf':>8}  {'Accuracy':>9}")
    print(hdr)
    print(f"{'─'*W}")

    for name, u_open, u_close, midnight, eat_str in SESSION_DEFS:
        mask   = session_mask(hour, u_open, u_close, midnight)
        p_sess = preds[mask]
        n_bars = mask.sum()
        if len(p_sess) == 0:
            continue

        sigs      = p_sess[p_sess["signal"] != "hold"]
        n_sigs    = len(sigs)
        n_buy     = (sigs["signal"] == "buy").sum()
        n_sell    = (sigs["signal"] == "sell").sum()
        buy_pct   = n_buy  / n_sigs * 100 if n_sigs else 0
        sell_pct  = n_sell / n_sigs * 100 if n_sigs else 0
        avg_conf  = sigs["confidence"].mean() * 100 if n_sigs else 0

        cor_sess  = correct[mask].dropna()
        acc       = cor_sess.mean() * 100 if len(cor_sess) > 0 else float("nan")

        utc_str = (f"{u_open:02d}:00–{u_close:02d}:00" if not midnight
                   else f"{u_open:02d}:00–{u_close:02d}:00")
        acc_str = f"{acc:.1f}%" if not np.isnan(acc) else "  n/a"

        print(f"  {name:<20}  {utc_str:>12}  {eat_str:>12}  "
              f"{n_bars:>7,}  {n_sigs:>6,}  {buy_pct:>5.1f}%  {sell_pct:>5.1f}%  "
              f"{avg_conf:>7.1f}%  {acc_str:>9}")

    # All-session totals
    all_sigs  = preds[preds["signal"] != "hold"]
    n_all     = len(preds)
    n_sig_all = len(all_sigs)
    buy_all   = (all_sigs["signal"] == "buy").sum()
    sell_all  = (all_sigs["signal"] == "sell").sum()
    conf_all  = all_sigs["confidence"].mean() * 100 if n_sig_all else 0
    acc_all   = correct.dropna().mean() * 100 if len(correct.dropna()) > 0 else float("nan")

    print(f"{'─'*W}")
    print(f"  {'All sessions (24/5)':<20}  {'00:00–24:00':>12}  {'03:00–03:00':>12}  "
          f"{n_all:>7,}  {n_sig_all:>6,}  "
          f"{buy_all/n_sig_all*100:>5.1f}%  {sell_all/n_sig_all*100:>5.1f}%  "
          f"{conf_all:>7.1f}%  {acc_all:>8.1f}%")
    print(f"{'═'*W}")

    # ── Hourly breakdown ───────────────────────────────────────────────────────
    print(f"\n  ── HOURLY BREAKDOWN (UTC → EAT) — {symbol} ──")
    print(f"  {'UTC':>5}  {'EAT':>5}  {'Sigs':>6}  {'Buy%':>6}  {'Sell%':>6}  "
          f"{'AvgConf':>8}  {'Accuracy':>9}  Chart")
    print(f"  {'─'*75}")

    hour_stats = []
    for h in range(24):
        m      = (hour == h)
        p_h    = preds[m]
        sigs_h = p_h[p_h["signal"] != "hold"]
        n_s    = len(sigs_h)
        if n_s == 0:
            hour_stats.append((h, 0, 0, 0, 0, float("nan")))
            continue
        bp  = (sigs_h["signal"] == "buy").sum()  / n_s * 100
        sp  = (sigs_h["signal"] == "sell").sum() / n_s * 100
        cf  = sigs_h["confidence"].mean() * 100
        ac  = correct[m].dropna().mean() * 100 if len(correct[m].dropna()) > 0 else float("nan")
        hour_stats.append((h, n_s, bp, sp, cf, ac))

    max_sigs = max(s[1] for s in hour_stats) or 1
    for h, n_s, bp, sp, cf, ac in hour_stats:
        eat_h  = (h + EAT_OFFSET) % 24
        ac_str = f"{ac:.1f}%" if not np.isnan(ac) else "   n/a"
        bar    = _ascii_bar(n_s / max_sigs, 18) if n_s > 0 else ""
        # Mark active sessions
        tags = []
        if (h >= 22) or (h < 7):  tags.append("SYD")
        if h < 9:                  tags.append("TOK")
        if 8 <= h < 17:            tags.append("LON")
        if 13 <= h < 22:           tags.append("NY")
        tag_str = ",".join(tags)
        print(f"  {h:02d}:00  {eat_h:02d}:00  {n_s:>6,}  {bp:>5.1f}%  {sp:>5.1f}%  "
              f"{cf:>7.1f}%  {ac_str:>9}  {bar}  {tag_str}")

    # ── Best hours recommendation ──────────────────────────────────────────────
    ranked = sorted(
        [(h, ac) for h, _, _, _, _, ac in hour_stats
         if not np.isnan(ac) and hour_stats[h][1] >= 10],
        key=lambda x: x[1], reverse=True
    )
    if ranked:
        print(f"\n  ── TOP 5 HOURS BY ACCURACY — {symbol} ──")
        for rank, (h, ac) in enumerate(ranked[:5], 1):
            eat_h = (h + EAT_OFFSET) % 24
            n_s   = hour_stats[h][1]
            print(f"  #{rank}  UTC {h:02d}:00  →  EAT {eat_h:02d}:00  "
                  f"accuracy={ac:.1f}%  signals={n_s:,}")

    if do_plot:
        _ascii_hourly_plot(hour_stats, symbol)

    print()


def _ascii_hourly_plot(hour_stats: list, symbol: str) -> None:
    """Simple ASCII bar chart of signal count by UTC hour."""
    print(f"\n  ── SIGNAL VOLUME BY HOUR (UTC) — {symbol} ──")
    max_sigs = max(s[1] for s in hour_stats) or 1
    for h, n_s, _, _, _, ac in hour_stats:
        eat_h  = (h + EAT_OFFSET) % 24
        bar    = "█" * int(n_s / max_sigs * 40)
        ac_str = f"{ac:.0f}%" if not np.isnan(ac) else "n/a"
        print(f"  {h:02d}(EAT{eat_h:02d}) |{bar:<40}| {n_s:4d}  acc={ac_str}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Candle model session performance analysis")
    p.add_argument("--symbol", default=None, choices=list(SYMBOL_CFG.keys()))
    p.add_argument("--plot", action="store_true", help="Show ASCII signal volume chart")
    args = p.parse_args()

    symbols = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())
    for sym in symbols:
        print(f"\n{'='*70}")
        print(f"  Analysing {sym}...")
        print(f"{'='*70}")
        run_symbol(sym, args.plot)


if __name__ == "__main__":
    main()
