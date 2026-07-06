"""V5 standalone XAUUSD H4 trend trade engine.

Converts the validated continuous EWMAC trend forecast (the signal family
behind `h4-cta-v5`, where XAUUSD carries leg Sharpe ~0.77) into discrete
BUY/SELL trades with confidence-based flips, ATR stops/trails, and
percent-risk sizing — the execution style requested for live gold trading.

Pre-registered parameters (declared BEFORE the first backtest; no sweeps):

  signal        : EWMAC H4 speeds (48,192),(96,384),(192,768),(384,1536)
                  (daily Carver set x 6 — same as h4-cta-v5)
  enter         : |forecast| >= 0.50   (half average strength)
  flip          : opposite |forecast| >= 1.00 closes the position and opens
                  the reverse ("confidence" mode); weaker opposite signals
                  are ignored. "always" mode flips on any opposite signal
                  >= enter threshold (v4's `always` flip mode).
  confidence    : low [0.5,1.0), med [1.0,1.5), high >= 1.5 (journal only)
  ATR           : 14 completed H4 bars (Wilder)
  stops         : SL 2.0x ATR; trail 2.0x ATR behind peak after +1.0x ATR;
                  TP 6.0x ATR (only in the "sltp" exit variant)
  sizing        : risk 1% of current equity per trade over the 2x ATR stop
                  distance, quantized to 0.01 lots (100 oz contract)
  execution     : decisions on completed bar close, filled at NEXT bar open
                  +/- (spread/2 + slippage); slippage 1.0 pip (0.1 USD);
                  intrabar SL/TP checked against high/low, SL-first when
                  both are touched in one bar (conservative)

Exit variants (all pre-registered, all reported):
  flip   — signal flips only; sizing stop is virtual (not enforced)
  trail  — hard SL + trailing stop, no TP (let winners run)
  sltp   — hard SL + trailing stop + 6x ATR take-profit

Leakage surface: no fitted components; signal uses closes <= t (EWMAC with
shift(1) expanding scalars); execution strictly next-bar. Guarded by
tests/test_v5_xau_trend.py causality checks.

Do NOT add v4 hedge_loss/zone_recovery style double-sided positions here:
the v4 performance claims for those modes came from the contaminated signal
stack (see data/HONEST_VALIDATION.md on v4-externaldata), and paired
opposite positions double margin for zero net exposure on a small account.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.cta.signals import ewmac
from src.v5.h4_cta import H4_SPEEDS

PIP = 0.1                 # XAUUSD pip in price units (backward compat)
CONTRACT = 100.0          # oz per 1.0 lot (backward compat)
VOL_MIN, VOL_STEP = 0.01, 0.01
MAX_LOT = 20.0

# Per-symbol contract specs. quote_jpy: PnL accrues in JPY and is converted
# to USD at the exit-bar price.
SPECS = {
    "XAUUSD": dict(pip=0.1, contract=100.0, quote_jpy=False),
    "EURUSD": dict(pip=1e-4, contract=100_000.0, quote_jpy=False),
    "GBPUSD": dict(pip=1e-4, contract=100_000.0, quote_jpy=False),
    "USDJPY": dict(pip=1e-2, contract=100_000.0, quote_jpy=True),
}

PARAMS = dict(
    enter_thresh=0.50,
    flip_thresh=1.00,
    atr_period=14,
    sl_atr=2.0,
    trail_atr=2.0,
    trail_activation_atr=1.0,
    tp_atr=6.0,
    risk_frac=0.01,
    conf_risk_scale=None,  # e.g. {"low": 0.5, "med": 1.0, "high": 1.5}
    slippage_pips=1.0,
    spread_cost_mult=1.0,
    entry_delay_bars=1,
)


def confidence_bucket(strength: float) -> str:
    a = abs(strength)
    if a >= 1.5:
        return "high"
    if a >= 1.0:
        return "med"
    return "low"


def xau_signal(close: pd.Series) -> pd.Series:
    """Continuous trend forecast from closes <= t (|1.0| ~ average strength)."""
    return ewmac(close.to_frame("XAUUSD"), speeds=H4_SPEEDS)["XAUUSD"]


def wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR of COMPLETED bars: value at t uses bars <= t."""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period).mean()


def _round_lot(x: float) -> float:
    mag = round(round(abs(x) / VOL_STEP) * VOL_STEP, 8)
    if mag < VOL_MIN:
        return 0.0
    return float(np.sign(x) * min(mag, MAX_LOT))


def run_trades(df: pd.DataFrame, *, equity0: float = 3000.0,
               exit_mode: str = "trail", flip_mode: str = "confidence",
               params: dict | None = None, symbol: str = "XAUUSD") -> dict:
    """Bar-loop trade simulation. df: H4 OHLC + spread (pips), time-indexed.

    Returns dict(trades=DataFrame, equity=Series, signals=Series).
    """
    if exit_mode not in ("flip", "trail", "sltp"):
        raise ValueError(f"unknown exit_mode {exit_mode!r}")
    if flip_mode not in ("confidence", "always"):
        raise ValueError(f"unknown flip_mode {flip_mode!r}")
    p = {**PARAMS, **(params or {})}
    spec = SPECS[symbol]
    pip, contract, quote_jpy = spec["pip"], spec["contract"], spec["quote_jpy"]

    sig = xau_signal(df["close"]).values
    atr = wilder_atr(df, p["atr_period"]).values
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    spread_px = np.maximum(df["spread"].values,
                           np.nanmedian(df["spread"].values)) * pip * p["spread_cost_mult"]
    half_cost = spread_px / 2.0 + p["slippage_pips"] * pip
    idx = df.index

    eq = equity0
    equity = np.full(len(df), np.nan)
    trades: list[dict] = []
    pos = None            # dict(dir, lots, entry, sl, tp, peak, opened_t, conf)
    pending = None        # dict(dir, strength) decided last bar, filled this open

    def fill_price(t, direction, side):
        """side=+1 pay the offer (open/buy or close/short-cover)."""
        return o[t] + side * direction * half_cost[t]

    def close_position(t, price, reason):
        nonlocal eq, pos
        pnl = (price - pos["entry"]) * pos["dir"] * pos["lots"] * contract
        if quote_jpy:
            pnl /= price  # JPY -> USD at exit price
        eq += pnl
        trades.append(dict(
            open_time=pos["opened_t"], close_time=idx[t],
            direction="buy" if pos["dir"] > 0 else "sell",
            lots=pos["lots"], entry=round(pos["entry"], 2),
            exit=round(price, 2), pnl=round(pnl, 2),
            r_multiple=round(pnl / pos["risk_usd"], 2) if pos["risk_usd"] else np.nan,
            confidence=pos["conf"], exit_reason=reason))
        pos = None

    for t in range(len(df)):
        # 1) fill the pending decision at this bar's open (after the delay)
        if pending is not None and pending["wait"] > 0:
            pending["wait"] -= 1
        elif pending is not None:
            d = pending["dir"]
            if pos is not None:
                close_position(t, fill_price(t, pos["dir"], -1), "flip")
            if d != 0 and np.isfinite(atr[t - 1]):
                sl_dist = p["sl_atr"] * atr[t - 1]
                entry = fill_price(t, d, +1)
                risk = p["risk_frac"]
                if p["conf_risk_scale"]:
                    risk *= p["conf_risk_scale"][confidence_bucket(pending["strength"])]
                loss_per_lot = sl_dist * contract  # quote ccy
                if quote_jpy:
                    loss_per_lot /= entry          # -> USD
                lots = _round_lot((risk * eq) / loss_per_lot)
                if lots > 0:
                    pos = dict(
                        dir=d, lots=lots, entry=entry,
                        sl=entry - d * sl_dist if exit_mode != "flip" else None,
                        tp=entry + d * p["tp_atr"] * atr[t - 1]
                        if exit_mode == "sltp" else None,
                        peak=entry, trail_on=False,
                        atr_at_entry=atr[t - 1],
                        risk_usd=(sl_dist * contract * lots / entry
                                  if quote_jpy else sl_dist * contract * lots),
                        opened_t=idx[t], conf=confidence_bucket(pending["strength"]))
            pending = None

        # 2) intrabar exit checks on the current bar (SL first: conservative)
        if pos is not None and pos["sl"] is not None:
            hit_sl = (l[t] <= pos["sl"]) if pos["dir"] > 0 else (h[t] >= pos["sl"])
            hit_tp = (pos["tp"] is not None and
                      ((h[t] >= pos["tp"]) if pos["dir"] > 0 else (l[t] <= pos["tp"])))
            if hit_sl:
                close_position(t, pos["sl"] - pos["dir"] * half_cost[t],
                               "trail_stop" if pos["trail_on"] else "stop_loss")
            elif hit_tp:
                close_position(t, pos["tp"] - pos["dir"] * half_cost[t], "take_profit")

        # 3) trail update from the completed bar's extreme
        if pos is not None and exit_mode in ("trail", "sltp"):
            ext = h[t] if pos["dir"] > 0 else l[t]
            pos["peak"] = max(pos["peak"], ext) if pos["dir"] > 0 else min(pos["peak"], ext)
            gain = (pos["peak"] - pos["entry"]) * pos["dir"]
            if gain >= p["trail_activation_atr"] * pos["atr_at_entry"]:
                pos["trail_on"] = True
                new_sl = pos["peak"] - pos["dir"] * p["trail_atr"] * pos["atr_at_entry"]
                if (new_sl - pos["sl"]) * pos["dir"] > 0:  # only tighten
                    pos["sl"] = new_sl

        # 4) signal decision on this completed bar -> filled next open
        s = sig[t]
        # note: a pending set on the final bar is never filled in-backtest,
        # but IS returned so the live runner can act on it at the next open
        if np.isfinite(s) and np.isfinite(atr[t]):
            d = 1 if s > 0 else -1
            strong = abs(s) >= p["enter_thresh"]
            if pos is None and pending is None and strong:
                pending = dict(dir=d, strength=s, wait=p["entry_delay_bars"] - 1)
            elif pos is not None and d != pos["dir"] and strong:
                thresh = (p["flip_thresh"] if flip_mode == "confidence"
                          else p["enter_thresh"])
                if abs(s) >= thresh:
                    pending = dict(dir=d, strength=s, wait=p["entry_delay_bars"] - 1)

        # 5) mark equity (closed + floating)
        floating = ((c[t] - pos["entry"]) * pos["dir"] * pos["lots"] * contract
                    if pos is not None else 0.0)
        if quote_jpy and floating:
            floating /= c[t]
        equity[t] = eq + floating

    open_position = dict(pos) if pos is not None else None
    if pos is not None:  # mark final open trade closed at last close for stats
        close_position(len(df) - 1, c[-1], "eod_mark")
        equity[-1] = eq

    return dict(trades=pd.DataFrame(trades),
                equity=pd.Series(equity, index=idx, name="equity"),
                signal=pd.Series(sig, index=idx, name="forecast"),
                open_position=open_position,
                pending=pending)
