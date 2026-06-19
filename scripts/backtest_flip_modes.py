"""
backtest_flip_modes.py — Compare all 8 hedging modes on live champion models

Modes:
  always        — close opposite unconditionally, flip
  hedge_loss    — keep losing opposite open (natural SL/TP), add hedge
  hedge_exit    — same as hedge_loss but close loser at first profit tick
  trailing_hedge— same as hedge_loss but close loser via N-pip trailing stop
  lock          — equal-lot hedge; both close when combined P&L >= 0
  ratio_hedge   — 2× hedge lot; both close when combined P&L >= 0
  partial_close — close 50% of loser immediately, remaining 50% runs to SL/TP
  zone_recovery — layered hedge: layers 1×→2×→4×→8×; all close when combined >= 0
                  layers 3-4 triggered by price (latest layer losing by zone_pips)

Usage:
    conda run -n envmt5 python scripts/backtest_flip_modes.py
    conda run -n envmt5 python scripts/backtest_flip_modes.py --symbol EURUSD
    conda run -n envmt5 python scripts/backtest_flip_modes.py --trail-pips 15
    conda run -n envmt5 python scripts/backtest_flip_modes.py --hedge-ratio 2.5
    conda run -n envmt5 python scripts/backtest_flip_modes.py --zone-pips 20
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import PredictorPipeline
from src.evaluation.metrics import sharpe_ratio, max_drawdown, calmar_ratio

# ── Constants ──────────────────────────────────────────────────────────────────

MODES = [
    "always",
    "hedge_loss",
    "hedge_exit",
    "trailing_hedge",
    "lock",
    "ratio_hedge",
    "partial_close",
    "zone_recovery",
]

TRAIL_PIPS      = 10.0   # trailing_hedge: pips behind peak
HEDGE_RATIO     = 2.0    # ratio_hedge / zone layer-2: hedge lot multiplier
ZONE_PIPS       = 30.0   # zone_recovery: pip gap before new layer opens
MAX_ZONE_LAYERS = 4      # zone_recovery blowup protection

SYMBOL_PARAMS = {
    "EURUSD": dict(
        model_dir = "data/models/pipeline_EURUSD",
        data_path = "data/EURUSD_M15.csv",
        pip_size  = 0.0001,
        sl_pips   = 30.0,
        tp_pips   = 60.0,
    ),
    "USDJPY": dict(
        model_dir = "data/models/pipeline_USDJPY",
        data_path = "data/USDJPY_M15.csv",
        pip_size  = 0.01,
        sl_pips   = 30.0,
        tp_pips   = 60.0,
    ),
}

INITIAL_BALANCE = 10_000.0
RISK_PCT        = 0.01
SPREAD_PIPS     = 1.0
COMMISSION_PIPS = 0.5
THRESHOLD       = 0.40

# ── Position dataclass ─────────────────────────────────────────────────────────

@dataclass
class Pos:
    ticket:         int
    direction:      str    # "buy" | "sell"
    entry_bar:      int
    entry_price:    float
    entry_balance:  float
    sl:             float
    tp:             float
    sl_pips:        float
    tp_pips:        float
    lot_multiplier: float = 1.0   # >1 for ratio/zone hedge legs; 0.5 for partial_close remainder
    exit_bar:       Optional[int]   = None
    exit_price:     Optional[float] = None
    exit_reason:    str   = ""
    pnl_pips:       float = 0.0
    pnl_dollars:    float = 0.0


def _close(pos: Pos, bar: int, exit_p: float, reason: str,
           pip_size: float, cost_pips: float) -> float:
    """Fill exit fields and return pnl_dollars (lot_multiplier applied to dpp)."""
    pos.exit_bar    = bar
    pos.exit_price  = exit_p
    pos.exit_reason = reason
    raw = (
        (exit_p - pos.entry_price) / pip_size
        if pos.direction == "buy"
        else (pos.entry_price - exit_p) / pip_size
    )
    pos.pnl_pips    = raw - cost_pips
    dpp             = (pos.entry_balance * RISK_PCT) / pos.sl_pips * pos.lot_multiplier
    pos.pnl_dollars = pos.pnl_pips * dpp
    return pos.pnl_dollars


def _unreal(pos: Pos, price: float, pip_size: float) -> float:
    """Unrealized P&L in dollars at given price (no spread/commission deducted)."""
    raw_pips = (
        (price - pos.entry_price) / pip_size if pos.direction == "buy"
        else (pos.entry_price - price) / pip_size
    )
    dpp = (pos.entry_balance * RISK_PCT) / pos.sl_pips * pos.lot_multiplier
    return raw_pips * dpp


# ── Simulation ─────────────────────────────────────────────────────────────────

def simulate_mode(
    signals:     pd.DataFrame,
    prices:      pd.DataFrame,
    mode:        str,
    sl_pips:     float,
    tp_pips:     float,
    pip_size:    float,
    trail_pips:  float = TRAIL_PIPS,
    hedge_ratio: float = HEDGE_RATIO,
    zone_pips:   float = ZONE_PIPS,
    anti_mart:     float = 1.0,
    anti_mart_cap: float = 4.0,
) -> dict:
    """
    Bar-by-bar simulation of one flip mode.
    Signal fires at bar close; fill = close ± spread.
    SL/TP checked via bar high/low.

    anti_mart > 1.0 enables ANTI-MARTINGALE sizing: the directional bet's lot is multiplied by
    anti_mart**(consecutive wins), reset to 1× after any losing close, capped at anti_mart_cap.
    (Press winners; never add into losers.) anti_mart=1.0 → disabled (sizing unchanged).
    """
    cost_pips  = SPREAD_PIPS + COMMISSION_PIPS
    spread_pts = SPREAD_PIPS * pip_size
    sl_pts     = sl_pips  * pip_size
    tp_pts     = tp_pips  * pip_size

    balance     = INITIAL_BALANCE
    peak_bal    = INITIAL_BALANCE
    max_dd_pct  = 0.0
    equity_pts  : List[float] = []
    all_trades  : List[Pos]   = []
    open_pos    : List[Pos]   = []
    ticket      = 0

    # hedge_exit
    hedged_set  : Set[int]           = set()
    # trailing_hedge
    hedged_trail: Dict[int, float]   = {}
    # lock / ratio_hedge: orig_ticket → hedge_ticket (and reverse)
    pair_map    : Dict[int, int]     = {}
    pair_rev    : Dict[int, int]     = {}
    # partial_close: tickets where half-close already booked
    partial_done: Set[int]           = set()
    # zone_recovery: ordered list of tickets + lot multiplier per ticket
    zone_tickets: List[int]          = []
    zone_mults  : Dict[int, float]   = {}
    # anti-martingale: consecutive-win streak + how many closed trades already counted
    win_streak  = 0
    processed   = 0

    def _open_new(direction: str, lot_mult: float = 1.0) -> Pos:
        nonlocal ticket, balance
        fill = close + spread_pts if direction == "buy" else close - spread_pts
        sl_  = fill - sl_pts     if direction == "buy" else fill + sl_pts
        tp_  = fill + tp_pts     if direction == "buy" else fill - tp_pts
        ticket += 1
        pos = Pos(
            ticket        = ticket,
            direction     = direction,
            entry_bar     = i,
            entry_price   = fill,
            entry_balance = balance,
            sl            = sl_,
            tp            = tp_,
            sl_pips       = sl_pips,
            tp_pips       = tp_pips,
            lot_multiplier= lot_mult,
        )
        open_pos.append(pos)
        return pos

    def _purge(pos: Pos) -> None:
        """Remove a closed position from all tracking dicts."""
        hedged_set.discard(pos.ticket)
        hedged_trail.pop(pos.ticket, None)
        partial_done.discard(pos.ticket)
        if pos.ticket in pair_map:
            pair_rev.pop(pair_map.pop(pos.ticket), None)
        elif pos.ticket in pair_rev:
            pair_map.pop(pair_rev.pop(pos.ticket), None)
        if pos.ticket in zone_mults:
            if pos.ticket in zone_tickets:
                zone_tickets.remove(pos.ticket)
            zone_mults.pop(pos.ticket)

    p_buys  = signals["P_buy"].values
    p_sells = signals["P_sell"].values
    opens_  = prices["open"].reindex(signals.index).values
    highs   = prices["high"].reindex(signals.index).values
    lows_   = prices["low"].reindex(signals.index).values
    closes  = prices["close"].reindex(signals.index).values
    n       = len(signals)

    for i in range(n):
        high  = float(highs[i])
        low   = float(lows_[i])
        close = float(closes[i])

        # ── 1. SL / TP exits ──────────────────────────────────────────────────
        alive: List[Pos] = []
        for pos in open_pos:
            hit = False
            if pos.direction == "buy":
                if low <= pos.sl:
                    balance += _close(pos, i, pos.sl, "sl", pip_size, cost_pips); hit = True
                elif high >= pos.tp:
                    balance += _close(pos, i, pos.tp, "tp", pip_size, cost_pips); hit = True
            else:
                if high >= pos.sl:
                    balance += _close(pos, i, pos.sl, "sl", pip_size, cost_pips); hit = True
                elif low <= pos.tp:
                    balance += _close(pos, i, pos.tp, "tp", pip_size, cost_pips); hit = True
            if hit:
                all_trades.append(pos); _purge(pos)
            else:
                alive.append(pos)
        open_pos = alive

        # ── 2a. hedge_exit: close tracked loser at first profit ───────────────
        if mode == "hedge_exit" and hedged_set:
            alive = []
            for pos in open_pos:
                if pos.ticket not in hedged_set:
                    alive.append(pos); continue
                turned = False
                if pos.direction == "buy" and high > pos.entry_price:
                    ep = max(float(opens_[i]), pos.entry_price)
                    balance += _close(pos, i, ep, "hedge_exit", pip_size, cost_pips)
                    hedged_set.discard(pos.ticket); all_trades.append(pos); turned = True
                elif pos.direction == "sell" and low < pos.entry_price:
                    ep = min(float(opens_[i]), pos.entry_price)
                    balance += _close(pos, i, ep, "hedge_exit", pip_size, cost_pips)
                    hedged_set.discard(pos.ticket); all_trades.append(pos); turned = True
                if not turned:
                    alive.append(pos)
            open_pos = alive

        # ── 2b. trailing_hedge: update peak, close on pullback ────────────────
        if mode == "trailing_hedge" and hedged_trail:
            alive = []
            for pos in open_pos:
                if pos.ticket not in hedged_trail:
                    alive.append(pos); continue
                bar_peak = (
                    (high - pos.entry_price) / pip_size if pos.direction == "buy"
                    else (pos.entry_price - low) / pip_size
                )
                cur_pips = (
                    (close - pos.entry_price) / pip_size if pos.direction == "buy"
                    else (pos.entry_price - close) / pip_size
                )
                pk = max(hedged_trail[pos.ticket], bar_peak)
                hedged_trail[pos.ticket] = pk
                if pk > 0 and cur_pips <= pk - trail_pips:
                    balance += _close(pos, i, close, "trail_exit", pip_size, cost_pips)
                    hedged_trail.pop(pos.ticket); all_trades.append(pos)
                else:
                    alive.append(pos)
            open_pos = alive

        # ── 2c. lock / ratio_hedge: close pair when combined P&L >= 0 ─────────
        if mode in ("lock", "ratio_hedge") and pair_map:
            to_close: Set[int] = set()
            for orig_t, hedge_t in list(pair_map.items()):
                op = next((p for p in open_pos if p.ticket == orig_t), None)
                hp = next((p for p in open_pos if p.ticket == hedge_t), None)
                if op is None or hp is None:
                    pair_map.pop(orig_t, None); pair_rev.pop(hedge_t, None); continue
                if _unreal(op, close, pip_size) + _unreal(hp, close, pip_size) >= 0:
                    balance += _close(op, i, close, "lock_exit", pip_size, cost_pips)
                    balance += _close(hp, i, close, "lock_exit", pip_size, cost_pips)
                    all_trades.extend([op, hp])
                    to_close.update([orig_t, hedge_t])
                    pair_map.pop(orig_t); pair_rev.pop(hedge_t, None)
            if to_close:
                open_pos = [p for p in open_pos if p.ticket not in to_close]

        # ── 2d. zone_recovery: price-triggered new layers ─────────────────────
        if mode == "zone_recovery" and zone_tickets:
            # check if latest layer is losing enough to spawn next layer
            latest_t = zone_tickets[-1]
            latest_p = next((p for p in open_pos if p.ticket == latest_t), None)
            if latest_p and len(zone_tickets) < MAX_ZONE_LAYERS:
                latest_pips = (
                    (close - latest_p.entry_price) / pip_size if latest_p.direction == "buy"
                    else (latest_p.entry_price - close) / pip_size
                )
                if latest_pips <= -zone_pips:
                    new_dir  = "sell" if latest_p.direction == "buy" else "buy"
                    new_mult = 2.0 ** len(zone_tickets)   # 1→2→4→8
                    layer    = _open_new(new_dir, new_mult)
                    zone_tickets.append(layer.ticket)
                    zone_mults[layer.ticket] = new_mult

        # ── 2e. zone_recovery: close all when combined P&L >= 0 ───────────────
        if mode == "zone_recovery" and len(zone_tickets) > 1:
            open_zone = [p for p in open_pos if p.ticket in set(zone_tickets)]
            if open_zone and sum(_unreal(p, close, pip_size) for p in open_zone) >= 0:
                to_close = {p.ticket for p in open_zone}
                for pos in open_zone:
                    balance += _close(pos, i, close, "zone_exit", pip_size, cost_pips)
                    all_trades.append(pos)
                open_pos = [p for p in open_pos if p.ticket not in to_close]
                zone_tickets.clear(); zone_mults.clear()

        # ── 3. Equity tracking ─────────────────────────────────────────────────
        peak_bal   = max(peak_bal, balance)
        max_dd_pct = max(max_dd_pct, (peak_bal - balance) / peak_bal * 100)
        equity_pts.append(balance)

        # ── 3b. Anti-martingale: fold any trades that closed (in append order) into the
        #        consecutive-win streak. Win → +1; loss/scratch → reset to 0. ─────────
        while processed < len(all_trades):
            win_streak = win_streak + 1 if all_trades[processed].pnl_pips > 0 else 0
            processed += 1

        # ── 4. Signal ──────────────────────────────────────────────────────────
        p_buy  = float(p_buys[i])
        p_sell = float(p_sells[i])
        if   p_buy  >= THRESHOLD and p_buy  > p_sell: direction = "buy"
        elif p_sell >= THRESHOLD and p_sell > p_buy:  direction = "sell"
        else: continue

        # ── 5. Same-direction guard ────────────────────────────────────────────
        if any(p.direction == direction for p in open_pos):
            continue

        # ── 6. Handle opposite positions ──────────────────────────────────────
        opposite = [p for p in open_pos if p.direction != direction]
        next_lot_mult     = 1.0
        pending_pair_orig : Optional[int] = None
        open_zone_layer   = False

        for pos in opposite:
            profit_pips = (
                (close - pos.entry_price) / pip_size if pos.direction == "buy"
                else (pos.entry_price - close) / pip_size
            )
            in_profit = profit_pips > 0

            if mode == "always" or in_profit:
                balance += _close(pos, i, close, "flip", pip_size, cost_pips)
                open_pos.remove(pos); all_trades.append(pos); _purge(pos)

            elif mode in ("hedge_loss", "hedge_exit", "trailing_hedge"):
                if mode == "hedge_exit":
                    hedged_set.add(pos.ticket)
                elif mode == "trailing_hedge":
                    hedged_trail.setdefault(pos.ticket, float("-inf"))

            elif mode in ("lock", "ratio_hedge"):
                if pos.ticket not in pair_map and pos.ticket not in pair_rev:
                    pending_pair_orig = pos.ticket
                    next_lot_mult = hedge_ratio if mode == "ratio_hedge" else 1.0

            elif mode == "partial_close":
                if pos.ticket not in partial_done:
                    dpp = (pos.entry_balance * RISK_PCT / pos.sl_pips)
                    balance += profit_pips * dpp * 0.5   # book 50% of loss now
                    partial_done.add(pos.ticket)
                    pos.lot_multiplier = 0.5              # remaining 50% runs to SL/TP

            elif mode == "zone_recovery":
                if len(zone_tickets) < MAX_ZONE_LAYERS:
                    if pos.ticket not in zone_mults:
                        zone_tickets.append(pos.ticket)
                        zone_mults[pos.ticket] = 1.0
                    next_lot_mult  = 2.0   # signal-triggered layer is always 2×
                    open_zone_layer = True
                else:
                    # max layers hit — close everything and restart clean
                    all_zone = [p for p in open_pos if p.ticket in set(zone_tickets)]
                    for zp in all_zone:
                        balance += _close(zp, i, close, "zone_max", pip_size, cost_pips)
                        all_trades.append(zp); _purge(zp)
                    balance += _close(pos, i, close, "zone_max", pip_size, cost_pips)
                    all_trades.append(pos); _purge(pos)
                    zone_tickets.clear(); zone_mults.clear()
                    open_pos = [p for p in open_pos
                                if p.ticket not in {zp.ticket for zp in all_zone}
                                and p.ticket != pos.ticket]
                    next_lot_mult = 1.0; open_zone_layer = False

        # ── 7. Open new position ───────────────────────────────────────────────
        if any(p.direction == direction for p in open_pos):
            continue

        am_mult = min(anti_mart ** win_streak, anti_mart_cap) if anti_mart > 1.0 else 1.0
        new_pos = _open_new(direction, next_lot_mult * am_mult)

        if mode in ("lock", "ratio_hedge") and pending_pair_orig is not None:
            pair_map[pending_pair_orig] = new_pos.ticket
            pair_rev[new_pos.ticket]    = pending_pair_orig

        if mode == "zone_recovery" and open_zone_layer:
            zone_tickets.append(new_pos.ticket)
            zone_mults[new_pos.ticket] = next_lot_mult

    # ── Force-close remaining at end of data ──────────────────────────────────
    last_close = float(closes[-1])
    for pos in open_pos:
        balance += _close(pos, n - 1, last_close, "end", pip_size, cost_pips)
        all_trades.append(pos)

    # ── Metrics ────────────────────────────────────────────────────────────────
    eq       = pd.Series(equity_pts, index=signals.index[:len(equity_pts)])
    wins     = sum(1 for t in all_trades if t.pnl_pips > 0)
    n_trades = len(all_trades)
    MANAGED  = {"hedge_exit", "trail_exit", "lock_exit", "zone_exit"}
    n_managed = sum(1 for t in all_trades if t.exit_reason in MANAGED)

    return {
        "mode":        mode,
        "n_trades":    n_trades,
        "win_rate":    wins / n_trades if n_trades else 0.0,
        "net_pnl_pct": (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100,
        "max_dd_pct":  max_dd_pct,
        "sharpe":      sharpe_ratio(eq),
        "calmar":      calmar_ratio(eq),
        "n_managed":   n_managed,
        "balance":     balance,
        "equity":      eq,          # pd.Series indexed by bar timestamps
        "trades":      all_trades,  # list[Pos] for per-year breakdown
    }


# ── Per-symbol runner ──────────────────────────────────────────────────────────

def run_symbol(
    symbol:      str,
    trail_pips:  float = TRAIL_PIPS,
    hedge_ratio: float = HEDGE_RATIO,
    zone_pips:   float = ZONE_PIPS,
    anti_mart:     float = 1.0,
    anti_mart_cap: float = 4.0,
) -> None:
    params   = SYMBOL_PARAMS[symbol]
    pip_size = params["pip_size"]
    sl_pips  = params["sl_pips"]
    tp_pips  = params["tp_pips"]

    print(f"\n{'=' * 64}")
    print(f"  {symbol}  —  {params['model_dir']}")
    print(f"{'=' * 64}")

    pipe = PredictorPipeline.from_config()
    pipe.load(params["model_dir"])

    df_raw = pd.read_csv(params["data_path"], index_col=0, parse_dates=True)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    print(f"  Data: {len(df_raw):,} bars  "
          f"{df_raw.index[0].date()} → {df_raw.index[-1].date()}")

    print("  Building features (fit=False — uses loaded artifacts)...", flush=True)
    X_base, _ = pipe._fp.build(df_raw, fit=False)

    if pipe._enc is not None:
        latent = pipe._enc.transform(df_raw)
        shared = X_base.index.intersection(latent.index)
        X = pd.concat([X_base.loc[shared], latent.loc[shared]], axis=1)
    else:
        X = X_base

    for c in pipe._feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[pipe._feature_cols]
    print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    signals = pipe.predict_batch(X)
    prices  = df_raw.reindex(signals.index)

    n_buy  = (signals["signal"] == "buy").sum()
    n_sell = (signals["signal"] == "sell").sum()
    n_hold = (signals["signal"] == "hold").sum()
    print(f"  Signals: buy={n_buy:,}  sell={n_sell:,}  hold={n_hold:,}")

    results = []
    for mode in MODES:
        if mode == "trailing_hedge":
            label = f"trailing_hedge(t={trail_pips:.0f}p)"
        elif mode == "ratio_hedge":
            label = f"ratio_hedge(r={hedge_ratio:.1f}x)"
        elif mode == "zone_recovery":
            label = f"zone_recovery(z={zone_pips:.0f}p)"
        else:
            label = mode
        print(f"  Simulating [{label:<26}]...", end=" ", flush=True)
        r = simulate_mode(signals, prices, mode, sl_pips, tp_pips, pip_size,
                          trail_pips=trail_pips, hedge_ratio=hedge_ratio,
                          zone_pips=zone_pips, anti_mart=anti_mart, anti_mart_cap=anti_mart_cap)
        r["label"] = label
        results.append(r)
        extra = f"  mgd={r['n_managed']}" if r["n_managed"] else ""
        print(f"done → {r['n_trades']:,} trades  PnL={r['net_pnl_pct']:+.1f}%{extra}")

    print()
    title = (
        f"{symbol}  SL={sl_pips:.0f}p  TP={tp_pips:.0f}p  "
        f"threshold={THRESHOLD:.0%}  risk={RISK_PCT:.0%}"
    )
    print(title)
    hdr = (
        f"{'Mode':<30} {'Trades':>7} {'Win%':>7} "
        f"{'Net PnL':>9} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'MgdExits':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['label']:<30} "
            f"{r['n_trades']:>7,d} "
            f"{r['win_rate']:>6.1%} "
            f"{r['net_pnl_pct']:>+8.1f}% "
            f"{r['max_dd_pct']:>7.1f}% "
            f"{r['sharpe']:>8.2f} "
            f"{r['calmar']:>8.2f} "
            f"{r['n_managed']:>9,d}"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Compare all flip/hedge modes on champion models")
    p.add_argument("--symbol", default=None, choices=list(SYMBOL_PARAMS.keys()),
                   help="Single symbol (default: all)")
    p.add_argument("--trail-pips", type=float, default=TRAIL_PIPS,
                   help=f"trailing_hedge: pips behind peak (default {TRAIL_PIPS})")
    p.add_argument("--hedge-ratio", type=float, default=HEDGE_RATIO,
                   help=f"ratio_hedge: hedge lot multiplier (default {HEDGE_RATIO})")
    p.add_argument("--zone-pips", type=float, default=ZONE_PIPS,
                   help=f"zone_recovery: pip gap before new layer (default {ZONE_PIPS})")
    p.add_argument("--anti-mart", type=float, default=1.0,
                   help="anti-martingale factor: lot ×= factor^(consecutive wins). 1.0=off (default)")
    p.add_argument("--anti-mart-cap", type=float, default=4.0,
                   help="cap on the anti-martingale multiplier (default 4.0×)")
    args = p.parse_args()

    symbols = [args.symbol] if args.symbol else list(SYMBOL_PARAMS.keys())
    for sym in symbols:
        run_symbol(sym, trail_pips=args.trail_pips,
                   hedge_ratio=args.hedge_ratio, zone_pips=args.zone_pips,
                   anti_mart=args.anti_mart, anti_mart_cap=args.anti_mart_cap)
    print()


if __name__ == "__main__":
    main()
