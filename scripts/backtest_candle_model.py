"""
backtest_candle_model.py — Backtest the 1-bar candle predictor model.

Trade lifecycle (candle_predictor mode):
  Signal fires at close of bar i
    → open trade at close[i] ± spread
    → SL/TP = intra-bar protective stops (15p / 20p)

  Bar i+1:
    → Check bar i+1 high/low: SL or TP touched?
        YES → exit at SL/TP price (flash crash / spike protection)
        NO  → force-close at close[i+1]  ← primary exit (bar end)

Every trade lasts exactly 1 bar unless a protective stop fires first.

Usage:
    conda run -n envmt5 python scripts/backtest_candle_model.py
    conda run -n envmt5 python scripts/backtest_candle_model.py --symbol EURUSD
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.pipeline import PredictorPipeline
from src.evaluation.metrics import sharpe_ratio


# ── Extra features (must match train_candle_model.py) ─────────────────────────

def _add_extra_features(df_raw: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    idx = X.index
    hour = idx.hour
    extra = pd.DataFrame(index=idx)
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

# ── Constants ──────────────────────────────────────────────────────────────────

SYMBOL_CFG = {
    "EURUSD": dict(
        model_dir = "data/models/candle_EURUSD",
        data_path = "data/EURUSD_M15.csv",
        pip_size  = 0.0001,
        sl_pips   = 10.0,   # v2: 1:3 R:R
        tp_pips   = 30.0,
    ),
    "USDJPY": dict(
        model_dir = "data/models/candle_USDJPY",
        data_path = "data/USDJPY_M15.csv",
        pip_size  = 0.01,
        sl_pips   = 10.0,
        tp_pips   = 30.0,
    ),
}

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01
SPREAD_PIPS     = 1.0
COMMISSION_PIPS = 0.5
THRESHOLD       = 0.60   # v2: matches training threshold


# ── Trade dataclass ────────────────────────────────────────────────────────────

@dataclass
class CandleTrade:
    ticket:        int
    direction:     str    # "buy" | "sell"
    entry_bar:     int
    entry_price:   float
    entry_balance: float
    sl:            float
    tp:            float
    sl_pips:       float
    tp_pips:       float
    exit_bar:      Optional[int]   = None
    exit_price:    Optional[float] = None
    exit_reason:   str   = ""     # "bar_end" | "sl" | "tp" | "end"
    pnl_pips:      float = 0.0
    pnl_dollars:   float = 0.0


def _close(t: CandleTrade, bar: int, exit_p: float, reason: str,
           pip_size: float, cost_pips: float) -> float:
    t.exit_bar    = bar
    t.exit_price  = exit_p
    t.exit_reason = reason
    raw = (
        (exit_p - t.entry_price) / pip_size if t.direction == "buy"
        else (t.entry_price - exit_p) / pip_size
    )
    t.pnl_pips    = raw - cost_pips
    dpp           = (t.entry_balance * RISK_PCT) / t.sl_pips
    t.pnl_dollars = t.pnl_pips * dpp
    return t.pnl_dollars


# ── Core simulation ────────────────────────────────────────────────────────────

def simulate_candle(
    signals:  pd.DataFrame,
    prices:   pd.DataFrame,
    sl_pips:  float,
    tp_pips:  float,
    pip_size: float,
) -> dict:
    """
    Bar-by-bar simulation for candle_predictor mode.

    Each trade opens at the current bar's close and closes at the NEXT bar's
    close (force-close), unless a protective SL or TP is hit intra-bar first.
    """
    cost_pips  = SPREAD_PIPS + COMMISSION_PIPS
    spread_pts = SPREAD_PIPS * pip_size
    sl_pts     = sl_pips  * pip_size
    tp_pts     = tp_pips  * pip_size

    balance    = INITIAL_BALANCE
    peak_bal   = INITIAL_BALANCE
    max_dd_pct = 0.0
    equity_pts : List[float] = []
    all_trades : List[CandleTrade] = []
    open_trade : Optional[CandleTrade] = None
    ticket     = 0

    p_buys  = signals["P_buy"].values
    p_sells = signals["P_sell"].values
    highs   = prices["high"].reindex(signals.index).values
    lows_   = prices["low"].reindex(signals.index).values
    closes  = prices["close"].reindex(signals.index).values
    n       = len(signals)

    for i in range(n):
        high  = float(highs[i])
        low   = float(lows_[i])
        close = float(closes[i])

        # ── 1. Close trade from previous bar (this is bar i = entry_bar + 1) ──
        if open_trade is not None:
            # Check SL/TP first (intra-bar protective stops)
            hit = False
            if open_trade.direction == "buy":
                if low <= open_trade.sl:
                    balance += _close(open_trade, i, open_trade.sl, "sl", pip_size, cost_pips)
                    hit = True
                elif high >= open_trade.tp:
                    balance += _close(open_trade, i, open_trade.tp, "tp", pip_size, cost_pips)
                    hit = True
            else:
                if high >= open_trade.sl:
                    balance += _close(open_trade, i, open_trade.sl, "sl", pip_size, cost_pips)
                    hit = True
                elif low <= open_trade.tp:
                    balance += _close(open_trade, i, open_trade.tp, "tp", pip_size, cost_pips)
                    hit = True

            if not hit:
                # Primary candle exit: force-close at this bar's close
                balance += _close(open_trade, i, close, "bar_end", pip_size, cost_pips)

            all_trades.append(open_trade)
            open_trade = None

        # ── 2. Equity tracking ─────────────────────────────────────────────────
        peak_bal   = max(peak_bal, balance)
        max_dd_pct = max(max_dd_pct, (peak_bal - balance) / peak_bal * 100)
        equity_pts.append(balance)

        # ── 3. Signal — open new 1-bar trade at this bar's close ──────────────
        p_buy  = float(p_buys[i])
        p_sell = float(p_sells[i])
        if   p_buy  >= THRESHOLD and p_buy  > p_sell: direction = "buy"
        elif p_sell >= THRESHOLD and p_sell > p_buy:  direction = "sell"
        else: continue

        # ── 4. Open trade (fills at close ± spread) ───────────────────────────
        fill = close + spread_pts if direction == "buy" else close - spread_pts
        sl_  = fill - sl_pts     if direction == "buy" else fill + sl_pts
        tp_  = fill + tp_pts     if direction == "buy" else fill - tp_pts
        ticket += 1
        open_trade = CandleTrade(
            ticket        = ticket,
            direction     = direction,
            entry_bar     = i,
            entry_price   = fill,
            entry_balance = balance,
            sl            = sl_,
            tp            = tp_,
            sl_pips       = sl_pips,
            tp_pips       = tp_pips,
        )

    # ── Force-close last open trade at end of data ────────────────────────────
    if open_trade is not None:
        balance += _close(open_trade, n - 1, float(closes[-1]), "end", pip_size, cost_pips)
        all_trades.append(open_trade)

    # ── Metrics ────────────────────────────────────────────────────────────────
    eq       = pd.Series(equity_pts, index=signals.index[:len(equity_pts)])
    wins     = sum(1 for t in all_trades if t.pnl_pips > 0)
    n_trades = len(all_trades)

    # Exit reason breakdown
    by_reason: dict[str, int] = {}
    for t in all_trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

    return {
        "n_trades":    n_trades,
        "win_rate":    wins / n_trades if n_trades else 0.0,
        "net_pnl_pct": (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100,
        "max_dd_pct":  max_dd_pct,
        "balance":     balance,
        "equity":      eq,
        "trades":      all_trades,
        "by_reason":   by_reason,
    }


# ── Per-symbol runner ──────────────────────────────────────────────────────────

def run_symbol(symbol: str) -> None:
    cfg = SYMBOL_CFG[symbol]

    model_dir = Path(cfg["model_dir"])
    if not model_dir.exists():
        print(f"\n  [{symbol}] Model not found at {model_dir}")
        print(f"  Train first:  conda run -n envmt5 python scripts/train_candle_model.py --symbol {symbol}")
        return

    # Load candle model
    pipe = PredictorPipeline.from_config()
    pipe.load(str(model_dir))

    # Override SL/TP/threshold from saved pair_meta.json if present
    meta_path = model_dir / "pair_meta.json"
    if meta_path.exists():
        saved_meta = json.loads(meta_path.read_text())
        cfg = dict(cfg)  # local copy
        cfg["sl_pips"] = float(saved_meta.get("sl_pips", cfg["sl_pips"]))
        cfg["tp_pips"] = float(saved_meta.get("tp_pips", cfg["tp_pips"]))

    # Load raw data
    df_raw = pd.read_csv(cfg["data_path"], index_col=0, parse_dates=True)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    df_raw = df_raw.sort_index()

    span_yrs      = (df_raw.index[-1] - df_raw.index[0]).days / 365.25
    bars_per_year = len(df_raw) / span_yrs

    print(f"\n{'='*68}")
    print(f"  CANDLE PREDICTOR BACKTEST — {symbol}")
    print(f"  {len(df_raw):,} bars  {df_raw.index[0].date()} → {df_raw.index[-1].date()}")
    print(f"  SL={cfg['sl_pips']:.0f}p  TP={cfg['tp_pips']:.0f}p  "
          f"threshold={THRESHOLD:.0%}  risk={RISK_PCT:.0%}")
    print(f"{'='*68}")

    # Build features (fit=False — uses loaded scaler+encoder)
    print("  Building features...", end=" ", flush=True)
    try:
        X_base, _ = pipe._fp.build(df_raw, fit=False)
        if pipe._enc is not None:
            latent = pipe._enc.transform(df_raw)
            shared = X_base.index.intersection(latent.index)
            X = pd.concat([X_base.loc[shared], latent.loc[shared]], axis=1)
        else:
            X = X_base
        # Add session + MTF extra features (v2 models expect these)
        X = _add_extra_features(df_raw, X)
        for c in pipe._feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[pipe._feature_cols]
    except Exception as e:
        print(f"FAILED: {e}")
        return
    print(f"done  ({X.shape[0]:,} rows × {X.shape[1]} features)")

    # Generate signals
    print("  Generating signals...", end=" ", flush=True)
    signals = pipe.predict_batch(X)
    prices  = df_raw.reindex(signals.index)
    n_buy   = (signals["signal"] == "buy").sum()
    n_sell  = (signals["signal"] == "sell").sum()
    n_hold  = (signals["signal"] == "hold").sum()
    print(f"done  buy={n_buy:,}  sell={n_sell:,}  hold={n_hold:,}")

    # Simulate
    print("  Simulating...", end=" ", flush=True)
    r = simulate_candle(
        signals,
        prices,
        sl_pips  = cfg["sl_pips"],
        tp_pips  = cfg["tp_pips"],
        pip_size = cfg["pip_size"],
    )
    print("done")

    eq = r["equity"]
    s  = float(eq.pct_change().dropna().mean() / eq.pct_change().dropna().std() * np.sqrt(bars_per_year)) \
         if eq.pct_change().dropna().std() > 0 else float("nan")

    print(f"\n  ── RESULTS ─────────────────────────────────────────────────────")
    print(f"  Trades     : {r['n_trades']:,}")
    print(f"  Win rate   : {r['win_rate']:.1%}")
    print(f"  Sharpe     : {s:+.3f}  (annualized)")
    print(f"  Max DD     : {r['max_dd_pct']:.1f}%")
    print(f"  Net PnL    : {r['net_pnl_pct']:+.1f}%")
    print(f"  Final bal  : ${r['balance']:,.2f}")

    by_r = r["by_reason"]
    n    = r["n_trades"]
    print(f"\n  ── EXIT BREAKDOWN ──────────────────────────────────────────────")
    for reason in ("bar_end", "sl", "tp", "end"):
        cnt = by_r.get(reason, 0)
        pct = cnt / n * 100 if n else 0
        label = {
            "bar_end": "bar_end (normal force-close)",
            "sl":      "sl      (flash crash / spike)",
            "tp":      "tp      (TP hit intra-bar)",
            "end":     "end     (end of data)",
        }.get(reason, reason)
        print(f"    {label:<38}: {cnt:>6,}  ({pct:.1f}%)")

    # Per-year Sharpe
    years = sorted(eq.index.year.unique())
    if len(years) > 1:
        print(f"\n  ── PER-YEAR SHARPE ─────────────────────────────────────────────")
        hdr = f"  {'Year':<8}" + "".join(f"{y:>10}" for y in years) + f"{'Overall':>10}"
        print(hdr)
        row = f"  {'Sharpe':<8}"
        for yr in years:
            yr_eq = eq[eq.index.year == yr]
            if len(yr_eq) < 10 or yr_eq.pct_change().dropna().std() == 0:
                row += f"{'n/a':>10}"
            else:
                s_yr = float(yr_eq.pct_change().dropna().mean() /
                             yr_eq.pct_change().dropna().std() * np.sqrt(bars_per_year))
                row += f"{s_yr:>10.3f}"
        row += f"{s:>10.3f}"
        print(row)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Backtest candle predictor model")
    p.add_argument("--symbol", default=None, choices=list(SYMBOL_CFG.keys()))
    args = p.parse_args()
    symbols = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())
    for sym in symbols:
        run_symbol(sym)
    print()


if __name__ == "__main__":
    main()
