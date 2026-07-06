"""M15-execution research variant of the validated XAUUSD H4 trend engine.

The H4 EWMAC signal, ATR, thresholds, confidence flips, trail geometry, and
sizing are UNCHANGED (imported from `src/v5/xau_trend.py`, which stays
untouched — it backs the live demo). Only execution moves to M15 bars:
entries (market or pullback limit), stop monitoring, and optionally the
trail ratchet.

Leakage guards (the audit's #1 historical failure class):
- H4 bars are resampled from M15 with MT5 open-labeling, but every lookup
  keys off the bar's COMPLETION time (open + 4h). An M15 bar opening at
  time T may only see H4 bars whose completion <= T; the forming H4 bar is
  structurally invisible (searchsorted mapping, no ffill anywhere).
- Stops/trails set from bar j's data take effect from bar j+1 (M15 trail)
  or use only strictly-past completed-H4 extremes (H4 trail).
- tests/test_v5_xau_m15_exec.py mutates future M15 bars and requires
  bit-identical earlier decisions.

Pre-registered cells (V5_PLAN.MD "M15 Execution Experiment"):
  E1  market entry at first M15 open after the H4 signal bar completes
  E2a limit at P0 - dir*0.25*ATR_H4, TTL 24 M15 bars then market
  E2b same with 0.50*ATR_H4
  E3  E1 + trail ratchet from M15 extremes (same 2xATR_H4 distance)
  E4  E1 + no NEW entries on M15 bars opening 00:00-07:59 server time
      (deferred to >=08:00; exits/trail always active)

Locked rules: limits apply to flip re-entries (the closing leg is always
market, immediately); a working limit is cancelled if a newer completed H4
bar invalidates the signal (direction change or |f| < enter threshold);
limit fills on touch at the limit price + half spread (no slippage); the
1-pip penetration fill is a stress, not a cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.v5.xau_trend import (PARAMS, SPECS, _round_lot, confidence_bucket,
                              wilder_atr, xau_signal)

PIP = SPECS["XAUUSD"]["pip"]
CONTRACT = SPECS["XAUUSD"]["contract"]
LIMIT_TTL = 24          # M15 bars a pullback limit stays working
SESSION_BLOCK = (0, 8)  # E4: no new entries when fill bar opens in [0,8)


def resample_h4(m15: pd.DataFrame) -> pd.DataFrame:
    """MT5 open-labeled H4 bars from M15 (gate-checked vs the H4 CSV)."""
    h4 = m15.resample("4h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last")).dropna()
    return h4


def h4_features(m15: pd.DataFrame):
    """Completed-bar H4 features + the leakage-safe M15->H4 mapping."""
    h4 = resample_h4(m15)
    close_time = (h4.index + pd.Timedelta(hours=4)).asi8
    sig = xau_signal(h4["close"]).values
    atr = wilder_atr(h4, PARAMS["atr_period"]).values
    hi, lo = h4["high"].values, h4["low"].values
    m15_open_ns = m15.index.asi8
    # H4 bar i is visible to M15 bar j iff close_time[i] <= m15 open time
    last_done = np.searchsorted(close_time, m15_open_ns, side="right") - 1
    return dict(sig=sig, atr=atr, high=hi, low=lo, last_done=last_done)


def ltf_trend(m15: pd.DataFrame, rule: str, fast: int = 8, slow: int = 24):
    """Completed-bar LTF trend sign (+1/-1) and its leakage-safe M15 mapping."""
    bars = m15["close"].resample(rule, label="left", closed="left").last().dropna()
    step = pd.tseries.frequencies.to_offset(rule)
    close_time = (bars.index + step).asi8
    trend = np.sign(bars.ewm(span=fast, min_periods=fast).mean()
                    - bars.ewm(span=slow, min_periods=slow).mean()).values
    last_done = np.searchsorted(close_time, m15.index.asi8, side="right") - 1
    return trend, last_done


def run_trades_m15(m15: pd.DataFrame, *, equity0: float = 3000.0,
                   limit_k: float | None = None,
                   trail_source: str = "h4",
                   session_block: tuple | None = None,
                   confirm_rule: str | None = None,
                   params: dict | None = None,
                   entry_delay_h4: int = 0,
                   limit_penetration_pips: float = 0.0) -> dict:
    """M15 state-machine replay of the H4 strategy. Returns trades/equity."""
    p = {**PARAMS, **(params or {})}
    f = h4_features(m15)
    sig_h4, atr_h4 = f["sig"], f["atr"]
    h4_hi, h4_lo, last_done = f["high"], f["low"], f["last_done"]

    o = m15["open"].values
    h = m15["high"].values
    l = m15["low"].values
    c = m15["close"].values
    spread_px = np.maximum(m15["spread"].values,
                           np.nanmedian(m15["spread"].values)) * PIP \
        * p["spread_cost_mult"]
    half_cost = spread_px / 2.0 + p["slippage_pips"] * PIP
    hours = m15.index.hour.values
    idx = m15.index
    n = len(m15)
    pen = limit_penetration_pips * PIP
    if confirm_rule is not None:
        ltf_sign, ltf_done = ltf_trend(m15, confirm_rule)

    eq = equity0
    equity = np.full(n, np.nan)
    trades: list[dict] = []
    pos = None
    order = None   # dict(kind='market'|'limit', dir, strength, i_h4, limit, ttl, delay)
    seen_h4 = last_done[0]

    def blocked(j):
        return (session_block is not None and
                session_block[0] <= hours[j] < session_block[1])

    def confirmed(j, d):
        """LTF trend agreement on COMPLETED bars only (E5 cells)."""
        if confirm_rule is None:
            return True
        k = ltf_done[j]
        return k >= 0 and np.isfinite(ltf_sign[k]) and ltf_sign[k] == d

    def open_pos(j, d, entry, strength, i_h4):
        nonlocal pos
        sl_dist = p["sl_atr"] * atr_h4[i_h4]
        first_trail_i = last_done[j] + 1   # the H4 window containing the fill
        risk = p["risk_frac"]
        if p.get("conf_risk_scale"):
            risk *= p["conf_risk_scale"][confidence_bucket(strength)]
        lots = _round_lot((risk * eq) / (sl_dist * CONTRACT))
        if lots <= 0:
            return
        pos = dict(dir=d, lots=lots, entry=entry,
                   sl=entry - d * sl_dist, peak=entry, trail_on=False,
                   first_trail_i=first_trail_i,
                   atr_at_entry=atr_h4[i_h4],
                   risk_usd=sl_dist * CONTRACT * lots,
                   opened_t=idx[j], conf=confidence_bucket(strength))

    def close_pos(j, price, reason):
        nonlocal eq, pos
        pnl = (price - pos["entry"]) * pos["dir"] * pos["lots"] * CONTRACT
        eq += pnl
        trades.append(dict(
            open_time=pos["opened_t"], close_time=idx[j],
            direction="buy" if pos["dir"] > 0 else "sell",
            lots=pos["lots"], entry=round(pos["entry"], 2),
            exit=round(price, 2), pnl=round(pnl, 2),
            r_multiple=round(pnl / pos["risk_usd"], 2),
            confidence=pos["conf"], exit_reason=reason))
        pos = None

    for j in range(n):
        i = last_done[j]
        new_h4 = i > seen_h4
        if new_h4:
            seen_h4 = i

        # ── 1) H4-boundary decision (uses ONLY completed H4 bars <= now)
        if new_h4 and i >= 0 and np.isfinite(sig_h4[i]) and np.isfinite(atr_h4[i]):
            s = sig_h4[i]
            d = 1 if s > 0 else -1
            strong = abs(s) >= p["enter_thresh"]
            # cancel a working limit the new bar invalidates
            if order is not None and order["kind"] == "limit" and \
                    (not strong or d != order["dir"]):
                order = None
            if order is not None and order.get("delay", 0) > 0:
                order["delay"] -= 1
            elif pos is None and order is None and strong:
                order = dict(kind="market" if limit_k is None else "arm",
                             dir=d, strength=s, i_h4=i, delay=entry_delay_h4)
            elif pos is not None and d != pos["dir"] and abs(s) >= p["flip_thresh"]:
                order = dict(kind="flip", dir=d, strength=s, i_h4=i,
                             delay=entry_delay_h4)

        # ── 2) order handling at this bar's open
        if order is not None and order.get("delay", 0) == 0:
            k = order["kind"]
            if k == "flip" and pos is not None:
                close_pos(j, o[j] - pos["dir"] * half_cost[j], "flip")
                order["kind"] = "market" if limit_k is None else "arm"
                k = order["kind"]
            if k == "market":
                wait = order.setdefault("confirm_ttl", LIMIT_TTL)
                ok = confirmed(j, order["dir"]) or wait <= 0
                if not blocked(j) and ok:
                    open_pos(j, order["dir"],
                             o[j] + order["dir"] * half_cost[j],
                             order["strength"], order["i_h4"])
                    order = None
                else:
                    order["confirm_ttl"] = wait - 1
            elif k == "arm":
                if not blocked(j):   # E4 defers limit placement too
                    order.update(kind="limit",
                                 limit=o[j] - order["dir"] * limit_k
                                 * atr_h4[order["i_h4"]],
                                 ttl=LIMIT_TTL)
            if order is not None and order["kind"] == "limit":
                d = order["dir"]
                lim = order["limit"]
                need = lim - d * pen    # stress: require 1-pip trade-through
                touched = (l[j] <= need) if d > 0 else (h[j] >= need)
                if touched and not blocked(j):
                    entry = lim + d * spread_px[j] / 2.0  # limit fill, no slip
                    open_pos(j, d, entry, order["strength"], order["i_h4"])
                    order = None
                else:
                    order["ttl"] -= 1
                    if order["ttl"] <= 0:
                        order["kind"] = "market"  # fills next bar's open

        # ── 3) stop monitoring on this M15 bar
        if pos is not None:
            hit = (l[j] <= pos["sl"]) if pos["dir"] > 0 else (h[j] >= pos["sl"])
            if hit:
                close_pos(j, pos["sl"] - pos["dir"] * half_cost[j],
                          "trail_stop" if pos["trail_on"] else "stop_loss")

        # ── 4) trail ratchet
        if pos is not None:
            if trail_source == "h4":
                # only completed windows during which the position existed —
                # never seed the peak from pre-entry extremes (E1 sanity bug)
                if new_h4 and i >= pos["first_trail_i"]:
                    ext = h4_hi[i] if pos["dir"] > 0 else h4_lo[i]
                    pos["peak"] = (max(pos["peak"], ext) if pos["dir"] > 0
                                   else min(pos["peak"], ext))
            else:                       # m15: bar j's extreme, effective j+1
                ext = h[j] if pos["dir"] > 0 else l[j]
                pos["peak"] = (max(pos["peak"], ext) if pos["dir"] > 0
                               else min(pos["peak"], ext))
            gain = (pos["peak"] - pos["entry"]) * pos["dir"]
            if gain >= p["trail_activation_atr"] * pos["atr_at_entry"]:
                pos["trail_on"] = True
                new_sl = pos["peak"] - pos["dir"] * p["trail_atr"] \
                    * pos["atr_at_entry"]
                if (new_sl - pos["sl"]) * pos["dir"] > 0:
                    pos["sl"] = new_sl

        floating = ((c[j] - pos["entry"]) * pos["dir"] * pos["lots"] * CONTRACT
                    if pos is not None else 0.0)
        equity[j] = eq + floating

    if pos is not None:
        close_pos(n - 1, c[-1], "eod_mark")
        equity[-1] = eq
    return dict(trades=pd.DataFrame(trades),
                equity=pd.Series(equity, index=idx, name="equity"))
