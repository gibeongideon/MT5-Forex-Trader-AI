"""
backtest_candle_trail.py — Backtest the candle_trail mode vs candle_predictor.

Trade lifecycle (candle_trail mode):
  Signal fires at close of bar i with confidence C:
    → max_bars = 1 (C<0.70) | 2 (0.70≤C<0.80) | 4 (C≥0.80)
    → open trade at close[i] ± spread
    → SL/TP = protective stops (same as candle_predictor)

  Each subsequent bar:
    → Update peak profit (bar_high for BUY, bar_low for SELL)
    → If peak >= trail_activation_pips: advance trailing SL
    → Check SL/TP hit using bar high/low
    → If bars_held == max_bars and still open: force-close at next bar open

Outputs a side-by-side comparison table vs candle_predictor (base).

Usage:
    conda run -n envmt5 python scripts/backtest_candle_trail.py
    conda run -n envmt5 python scripts/backtest_candle_trail.py --symbol EURUSD
    conda run -n envmt5 python scripts/backtest_candle_trail.py --symbol USDJPY \\
        --trail-activation-pips 12 --trail-pips-behind 8 \\
        --max-bars-low 1 --max-bars-med 3 --max-bars-high 5
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.pipeline import PredictorPipeline


# ── Extra features (must match train_candle_model.py v3) ─────────────────────

def _add_extra_features(df_raw: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    idx  = X.index
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


# ── Constants ─────────────────────────────────────────────────────────────────

SYMBOL_CFG = {
    "EURUSD": dict(
        model_dir = "data/models/candle_EURUSD",
        data_path = "data/EURUSD_M15.csv",
        pip_size  = 0.0001,
        sl_pips   = 10.0,
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
THRESHOLD       = 0.60


# ── Trade dataclass ───────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticket:        int
    direction:     str
    entry_bar:     int
    entry_price:   float
    entry_balance: float
    sl:            float
    tp:            float
    sl_pips:       float
    max_bars:      int
    confidence:    float
    exit_bar:      Optional[int]   = None
    exit_price:    Optional[float] = None
    exit_reason:   str             = ""
    pnl_pips:      float           = 0.0
    pnl_dollars:   float           = 0.0
    peak_pips:     float           = 0.0
    bars_held:     int             = 0


def _close_trade(t: Trade, bar: int, exit_p: float, reason: str,
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


# ── Base simulation (candle_predictor — 1 bar, force-close) ──────────────────

def simulate_base(signals: pd.DataFrame, prices: pd.DataFrame,
                  sl_pips: float, tp_pips: float, pip_size: float) -> dict:
    cost_pips  = SPREAD_PIPS + COMMISSION_PIPS
    spread_pts = SPREAD_PIPS * pip_size
    sl_pts     = sl_pips * pip_size
    tp_pts     = tp_pips * pip_size

    balance    = INITIAL_BALANCE
    peak_bal   = INITIAL_BALANCE
    max_dd_pct = 0.0
    equity_pts: List[float] = []
    all_trades: List[Trade] = []
    open_trade: Optional[Trade] = None
    ticket = 0

    p_buys  = signals["P_buy"].values
    p_sells = signals["P_sell"].values
    highs   = prices["high"].reindex(signals.index).values
    lows_   = prices["low"].reindex(signals.index).values
    closes  = prices["close"].reindex(signals.index).values
    confs   = signals[["P_buy", "P_sell"]].max(axis=1).values
    n       = len(signals)

    for i in range(n):
        high  = float(highs[i])
        low   = float(lows_[i])
        close = float(closes[i])

        if open_trade is not None:
            hit = False
            if open_trade.direction == "buy":
                if low <= open_trade.sl:
                    balance += _close_trade(open_trade, i, open_trade.sl, "sl", pip_size, cost_pips)
                    hit = True
                elif high >= open_trade.tp:
                    balance += _close_trade(open_trade, i, open_trade.tp, "tp", pip_size, cost_pips)
                    hit = True
            else:
                if high >= open_trade.sl:
                    balance += _close_trade(open_trade, i, open_trade.sl, "sl", pip_size, cost_pips)
                    hit = True
                elif low <= open_trade.tp:
                    balance += _close_trade(open_trade, i, open_trade.tp, "tp", pip_size, cost_pips)
                    hit = True
            if not hit:
                balance += _close_trade(open_trade, i, close, "bar_end", pip_size, cost_pips)
            all_trades.append(open_trade)
            open_trade = None

        peak_bal   = max(peak_bal, balance)
        max_dd_pct = max(max_dd_pct, (peak_bal - balance) / peak_bal * 100)
        equity_pts.append(balance)

        p_buy  = float(p_buys[i])
        p_sell = float(p_sells[i])
        if   p_buy  >= THRESHOLD and p_buy  > p_sell: direction = "buy"
        elif p_sell >= THRESHOLD and p_sell > p_buy:  direction = "sell"
        else: continue

        fill = close + spread_pts if direction == "buy" else close - spread_pts
        sl_  = fill - sl_pts     if direction == "buy" else fill + sl_pts
        tp_  = fill + tp_pts     if direction == "buy" else fill - tp_pts
        ticket += 1
        open_trade = Trade(
            ticket=ticket, direction=direction, entry_bar=i,
            entry_price=fill, entry_balance=balance,
            sl=sl_, tp=tp_, sl_pips=sl_pips,
            max_bars=1, confidence=float(confs[i]),
        )

    if open_trade is not None:
        balance += _close_trade(open_trade, n - 1, float(closes[-1]), "end", pip_size, cost_pips)
        all_trades.append(open_trade)

    return _stats(all_trades, equity_pts, balance, max_dd_pct, signals.index)


# ── Trail simulation (candle_trail) ──────────────────────────────────────────

def simulate_trail(
    signals:              pd.DataFrame,
    prices:               pd.DataFrame,
    sl_pips:              float,
    tp_pips:              float,
    pip_size:             float,
    trail_activation_pips: float = 15.0,
    trail_pips_behind:    float = 10.0,
    max_bars_low:         int   = 1,
    max_bars_med:         int   = 2,
    max_bars_high:        int   = 4,
) -> dict:
    """
    Bar-by-bar simulation for candle_trail mode.

    Trade opens at bar i close. For each subsequent bar the trade is held:
      1. Update peak pips from bar_high (BUY) or bar_low (SELL)
      2. Advance trailing SL from peak if peak >= trail_activation_pips
      3. Check SL/TP hit using bar high/low; exit if triggered
      4. If bars_held == max_bars → force-close at this bar's close
    """
    cost_pips  = SPREAD_PIPS + COMMISSION_PIPS
    spread_pts = SPREAD_PIPS * pip_size
    sl_pts     = sl_pips * pip_size
    tp_pts     = tp_pips * pip_size

    balance    = INITIAL_BALANCE
    peak_bal   = INITIAL_BALANCE
    max_dd_pct = 0.0
    equity_pts: List[float] = []
    all_trades: List[Trade] = []
    open_trade: Optional[Trade] = None
    ticket = 0

    p_buys  = signals["P_buy"].values
    p_sells = signals["P_sell"].values
    highs   = prices["high"].reindex(signals.index).values
    lows_   = prices["low"].reindex(signals.index).values
    closes  = prices["close"].reindex(signals.index).values
    opens_  = prices["open"].reindex(signals.index).values
    confs   = signals[["P_buy", "P_sell"]].max(axis=1).values
    n       = len(signals)

    for i in range(n):
        high  = float(highs[i])
        low   = float(lows_[i])
        close = float(closes[i])
        open_ = float(opens_[i])

        # ── Manage open trade ─────────────────────────────────────────────
        if open_trade is not None:
            open_trade.bars_held += 1

            # Update peak pips this bar (intra-bar worst case for peak)
            if open_trade.direction == "buy":
                bar_peak_pips = (high - open_trade.entry_price) / pip_size
            else:
                bar_peak_pips = (open_trade.entry_price - low) / pip_size
            if bar_peak_pips > open_trade.peak_pips:
                open_trade.peak_pips = bar_peak_pips

            # Advance trailing SL if threshold reached
            if open_trade.peak_pips >= trail_activation_pips:
                trail_offset = open_trade.peak_pips - trail_pips_behind
                if trail_offset > 0:
                    if open_trade.direction == "buy":
                        candidate_sl = open_trade.entry_price + trail_offset * pip_size
                        if candidate_sl > open_trade.sl:
                            open_trade.sl = candidate_sl
                    else:
                        candidate_sl = open_trade.entry_price - trail_offset * pip_size
                        if candidate_sl < open_trade.sl:
                            open_trade.sl = candidate_sl

            # Check SL/TP hit this bar
            hit = False
            if open_trade.direction == "buy":
                if low <= open_trade.sl:
                    balance += _close_trade(open_trade, i, open_trade.sl, "trail_sl", pip_size, cost_pips)
                    hit = True
                elif high >= open_trade.tp:
                    balance += _close_trade(open_trade, i, open_trade.tp, "tp", pip_size, cost_pips)
                    hit = True
            else:
                if high >= open_trade.sl:
                    balance += _close_trade(open_trade, i, open_trade.sl, "trail_sl", pip_size, cost_pips)
                    hit = True
                elif low <= open_trade.tp:
                    balance += _close_trade(open_trade, i, open_trade.tp, "tp", pip_size, cost_pips)
                    hit = True

            if not hit:
                if open_trade.bars_held >= open_trade.max_bars:
                    # Force-close at this bar's close (bar i = entry + max_bars)
                    balance += _close_trade(open_trade, i, close, "bar_end", pip_size, cost_pips)
                    hit = True

            if hit:
                all_trades.append(open_trade)
                open_trade = None

        # ── Equity snapshot ───────────────────────────────────────────────
        peak_bal   = max(peak_bal, balance)
        max_dd_pct = max(max_dd_pct, (peak_bal - balance) / peak_bal * 100)
        equity_pts.append(balance)

        # ── Open new trade (only if none currently open) ──────────────────
        if open_trade is not None:
            continue

        p_buy  = float(p_buys[i])
        p_sell = float(p_sells[i])
        if   p_buy  >= THRESHOLD and p_buy  > p_sell: direction = "buy"
        elif p_sell >= THRESHOLD and p_sell > p_buy:  direction = "sell"
        else: continue

        conf = float(confs[i])
        if conf < 0.70:
            mb = max_bars_low
        elif conf < 0.80:
            mb = max_bars_med
        else:
            mb = max_bars_high

        fill = close + spread_pts if direction == "buy" else close - spread_pts
        sl_  = fill - sl_pts     if direction == "buy" else fill + sl_pts
        tp_  = fill + tp_pts     if direction == "buy" else fill - tp_pts
        ticket += 1
        open_trade = Trade(
            ticket=ticket, direction=direction, entry_bar=i,
            entry_price=fill, entry_balance=balance,
            sl=sl_, tp=tp_, sl_pips=sl_pips,
            max_bars=mb, confidence=conf,
        )

    # Force-close at end of data
    if open_trade is not None:
        balance += _close_trade(open_trade, n - 1, float(closes[-1]), "end", pip_size, cost_pips)
        all_trades.append(open_trade)

    return _stats(all_trades, equity_pts, balance, max_dd_pct, signals.index)


# ── Shared metrics helper ─────────────────────────────────────────────────────

def _stats(trades: List[Trade], equity_pts: list, balance: float,
           max_dd_pct: float, index: pd.Index) -> dict:
    n_trades = len(trades)
    wins     = sum(1 for t in trades if t.pnl_pips > 0)
    by_reason: dict[str, int] = {}
    for t in trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

    avg_win_pips  = (sum(t.pnl_pips for t in trades if t.pnl_pips > 0) / max(wins, 1))
    avg_loss_pips = (sum(t.pnl_pips for t in trades if t.pnl_pips <= 0)
                     / max(n_trades - wins, 1))

    # Distribution of max_bars held
    bars_dist: dict[int, int] = {}
    for t in trades:
        bars_dist[t.bars_held] = bars_dist.get(t.bars_held, 0) + 1

    eq = pd.Series(equity_pts, index=index[:len(equity_pts)])
    return {
        "n_trades":     n_trades,
        "win_rate":     wins / n_trades if n_trades else 0.0,
        "net_pnl_pct":  (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100,
        "max_dd_pct":   max_dd_pct,
        "balance":      balance,
        "equity":       eq,
        "trades":       trades,
        "by_reason":    by_reason,
        "avg_win_pips": avg_win_pips,
        "avg_los_pips": avg_loss_pips,
        "bars_dist":    bars_dist,
    }


# ── Sharpe helper ─────────────────────────────────────────────────────────────

def _sharpe(eq: pd.Series, bars_per_year: float) -> float:
    ret = eq.pct_change().dropna()
    if ret.std() == 0 or len(ret) < 2:
        return float("nan")
    return float(ret.mean() / ret.std() * np.sqrt(bars_per_year))


# ── Per-symbol runner ─────────────────────────────────────────────────────────

def run_symbol(symbol: str, args: argparse.Namespace) -> None:
    cfg = dict(SYMBOL_CFG[symbol])

    model_dir = Path(cfg["model_dir"])
    if not model_dir.exists():
        print(f"\n  [{symbol}] Model not found at {model_dir}")
        print(f"  Train first: conda run -n envmt5 python scripts/train_candle_model.py --symbol {symbol}")
        return

    pipe = PredictorPipeline.from_config()
    pipe.load(str(model_dir))

    meta_path = model_dir / "pair_meta.json"
    if meta_path.exists():
        saved = json.loads(meta_path.read_text())
        cfg["sl_pips"] = float(saved.get("sl_pips", cfg["sl_pips"]))
        cfg["tp_pips"] = float(saved.get("tp_pips", cfg["tp_pips"]))

    df_raw = pd.read_csv(cfg["data_path"], index_col=0, parse_dates=True)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    df_raw = df_raw.sort_index()

    span_yrs      = (df_raw.index[-1] - df_raw.index[0]).days / 365.25
    bars_per_year = len(df_raw) / span_yrs

    print(f"\n{'═'*72}")
    print(f"  CANDLE TRAIL BACKTEST — {symbol}")
    print(f"  {len(df_raw):,} bars  {df_raw.index[0].date()} → {df_raw.index[-1].date()}")
    print(f"  SL={cfg['sl_pips']:.0f}p  TP={cfg['tp_pips']:.0f}p  threshold={THRESHOLD:.0%}  risk={RISK_PCT:.0%}")
    print(f"  trail_activation={args.trail_activation_pips:.0f}p  "
          f"trail_behind={args.trail_pips_behind:.0f}p  "
          f"max_bars=[{args.max_bars_low}/{args.max_bars_med}/{args.max_bars_high}]"
          f"  (low/med/high conf)")
    print(f"{'═'*72}")

    print("  Building features...", end=" ", flush=True)
    try:
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
    except Exception as e:
        print(f"FAILED: {e}")
        return
    print(f"done  ({X.shape[0]:,} rows × {X.shape[1]} features)")

    print("  Generating signals...", end=" ", flush=True)
    signals = pipe.predict_batch(X)
    prices  = df_raw.reindex(signals.index)
    n_buy   = (signals["signal"] == "buy").sum()
    n_sell  = (signals["signal"] == "sell").sum()
    print(f"done  buy={n_buy:,}  sell={n_sell:,}")

    print("  Simulating base (candle_predictor)...", end=" ", flush=True)
    r_base = simulate_base(signals, prices, cfg["sl_pips"], cfg["tp_pips"], cfg["pip_size"])
    print("done")

    print("  Simulating trail (candle_trail)...", end=" ", flush=True)
    r_trail = simulate_trail(
        signals, prices, cfg["sl_pips"], cfg["tp_pips"], cfg["pip_size"],
        trail_activation_pips = args.trail_activation_pips,
        trail_pips_behind     = args.trail_pips_behind,
        max_bars_low          = args.max_bars_low,
        max_bars_med          = args.max_bars_med,
        max_bars_high         = args.max_bars_high,
    )
    print("done")

    s_base  = _sharpe(r_base["equity"],  bars_per_year)
    s_trail = _sharpe(r_trail["equity"], bars_per_year)

    W = 58
    print(f"\n  {'─'*W}")
    print(f"  {'Metric':<28}  {'candle_predictor':>13}  {'candle_trail':>13}")
    print(f"  {'─'*W}")

    def row(label, base_val, trail_val, fmt="{:.1f}"):
        b = fmt.format(base_val)
        t = fmt.format(trail_val)
        diff = trail_val - base_val
        arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "─")
        print(f"  {label:<28}  {b:>13}  {t:>13}  {arrow}")

    print(f"  {'Sharpe (annualized)':<28}  {s_base:>+13.3f}  {s_trail:>+13.3f}  "
          f"{'▲' if s_trail > s_base else '▼'}")
    row("Win rate (%)",       r_base["win_rate"]*100,     r_trail["win_rate"]*100)
    row("Max drawdown (%)",   r_base["max_dd_pct"],       r_trail["max_dd_pct"])
    row("Net PnL (%)",        r_base["net_pnl_pct"],      r_trail["net_pnl_pct"],    fmt="{:+.1f}")
    row("Final balance ($)",  r_base["balance"],          r_trail["balance"],         fmt="${:,.0f}")
    row("Trades",             r_base["n_trades"],         r_trail["n_trades"],        fmt="{:.0f}")
    row("Avg win (pips)",     r_base["avg_win_pips"],     r_trail["avg_win_pips"])
    row("Avg loss (pips)",    r_base["avg_los_pips"],     r_trail["avg_los_pips"])
    print(f"  {'─'*W}")

    # Exit reason breakdown
    print(f"\n  ── EXIT BREAKDOWN ───────────────────────────────────────────────")
    all_reasons = sorted(set(r_base["by_reason"]) | set(r_trail["by_reason"]))
    nb, nt = r_base["n_trades"], r_trail["n_trades"]
    print(f"  {'Reason':<18}  {'base cnt':>9}  {'base %':>7}  {'trail cnt':>10}  {'trail %':>8}")
    for reason in all_reasons:
        bc = r_base["by_reason"].get(reason, 0)
        tc = r_trail["by_reason"].get(reason, 0)
        bp = bc / nb * 100 if nb else 0
        tp = tc / nt * 100 if nt else 0
        print(f"  {reason:<18}  {bc:>9,}  {bp:>6.1f}%  {tc:>10,}  {tp:>7.1f}%")

    # Bars held distribution (trail only)
    print(f"\n  ── TRAIL: BARS HELD DISTRIBUTION ────────────────────────────────")
    total = r_trail["n_trades"]
    for nb_val in sorted(r_trail["bars_dist"]):
        cnt = r_trail["bars_dist"][nb_val]
        pct = cnt / total * 100 if total else 0
        print(f"    {nb_val} bar{'s' if nb_val != 1 else ' '}: {cnt:>6,}  ({pct:.1f}%)")

    # Per-year Sharpe
    for label, r, s in [("base", r_base, s_base), ("trail", r_trail, s_trail)]:
        eq    = r["equity"]
        years = sorted(eq.index.year.unique())
        if len(years) > 1:
            print(f"\n  ── PER-YEAR SHARPE [{label}] {'─'*30}")
            hdr = f"  {'Year':<8}" + "".join(f"{y:>10}" for y in years) + f"{'Overall':>10}"
            print(hdr)
            row_s = f"  {'Sharpe':<8}"
            for yr in years:
                yr_eq = eq[eq.index.year == yr]
                if len(yr_eq) < 10 or yr_eq.pct_change().dropna().std() == 0:
                    row_s += f"{'n/a':>10}"
                else:
                    s_yr = float(yr_eq.pct_change().dropna().mean() /
                                 yr_eq.pct_change().dropna().std() * np.sqrt(bars_per_year))
                    row_s += f"{s_yr:>10.3f}"
            row_s += f"{s:>10.3f}"
            print(row_s)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Backtest candle_trail vs candle_predictor")
    p.add_argument("--symbol", default=None, choices=list(SYMBOL_CFG.keys()))
    p.add_argument("--trail-activation-pips", type=float, default=15.0,
                   help="Pips in profit before trailing SL activates (default 15)")
    p.add_argument("--trail-pips-behind", type=float, default=10.0,
                   help="Trailing SL distance behind peak (default 10)")
    p.add_argument("--max-bars-low",  type=int, default=1,
                   help="Max bars for conf<0.70 (default 1)")
    p.add_argument("--max-bars-med",  type=int, default=2,
                   help="Max bars for conf 0.70-0.80 (default 2)")
    p.add_argument("--max-bars-high", type=int, default=4,
                   help="Max bars for conf>=0.80 (default 4)")
    args = p.parse_args()
    symbols = [args.symbol] if args.symbol else list(SYMBOL_CFG.keys())
    for sym in symbols:
        run_symbol(sym, args)
    print()


if __name__ == "__main__":
    main()
