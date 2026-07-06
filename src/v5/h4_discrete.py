"""Discrete-lot execution simulation for the V5 H4 book at small equity.

The H4 champion (`src/v5/h4_cta.py`) validates with continuous position
weights. On a real small account the broker quantizes every leg to 0.01-lot
steps, which distorts the vol target two ways: legs too small round to ZERO
(lost diversification) and the smallest tradeable chunk can be LARGER than
the target (forced oversizing). This module replays the same positions under
that quantization so the capital floor is measured, not guessed.

Pre-registered acceptance (declared before the sweep was run): an equity
level is viable when the discrete net Sharpe >= 0.8 x the continuous net
Sharpe at the same vol target.

Causality: the loop is strictly sequential — lots decided at bar t use
equity and prices at bar t and earn bar t+1's return, matching the
`positions.shift(1)` convention of the continuous engine.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.v5.h4_cta import PIP_SIZE

# HFM-style specs for the H4 universe. VERIFY against the live terminal
# (symbol_info.trade_contract_size / volume_min / volume_step) before trading.
H4_SPECS = {
    "EURUSD": dict(contract_size=100_000.0, quote_usd=False),  # per-lot $ = 100k * price
    "GBPUSD": dict(contract_size=100_000.0, quote_usd=False),
    "USDJPY": dict(contract_size=100_000.0, quote_usd=True),   # base is USD: per-lot $ = 100k
    "XAUUSD": dict(contract_size=100.0, quote_usd=False),      # 100 oz: per-lot $ = 100 * price
}
VOL_MIN = 0.01
VOL_STEP = 0.01


def per_lot_usd(symbol: str, price: float) -> float:
    spec = H4_SPECS[symbol]
    return spec["contract_size"] * (1.0 if spec["quote_usd"] else price)


def _round_step(x: float, step: float = VOL_STEP) -> float:
    return round(round(x / step) * step, 8)


def discrete_replay(positions: pd.DataFrame, close: pd.DataFrame,
                    spread: pd.DataFrame, equity0: float,
                    spread_cost_mult: float = 1.0) -> dict:
    """Replay target weights as broker-quantized lots, compounding equity.

    Returns dict with equity (Series), lots (DataFrame), actual weights,
    and quantization diagnostics.
    """
    symbols = list(positions.columns)
    idx = positions.index
    n, k = len(idx), len(symbols)

    w_tgt = positions.fillna(0.0).values
    px = close.values
    ret = close.pct_change(fill_method=None).fillna(0.0).values
    pips = np.array([PIP_SIZE[s] for s in symbols])
    spread_frac = (spread.values * pips / px) * spread_cost_mult

    lots = np.zeros((n, k))
    w_act = np.zeros((n, k))
    equity = np.empty(n)
    eq = equity0
    held = np.zeros(k)
    rounded_zero_bars = np.zeros(k)

    for t in range(n):
        # earn last bar's held exposure over this bar's return, pay costs on
        # lot changes decided THIS bar (cost applies when the change trades)
        if t > 0:
            eq *= 1.0 + float(w_act[t - 1] @ ret[t])
        per_lot = np.array([per_lot_usd(s, px[t, j]) for j, s in enumerate(symbols)])
        ideal = w_tgt[t] * eq / per_lot
        mag = np.array([_round_step(abs(x)) for x in ideal])
        zeroed = (mag < VOL_MIN) & (np.abs(ideal) > 0)
        rounded_zero_bars += zeroed
        mag = np.where(zeroed, 0.0, mag)
        new_lots = np.sign(ideal) * mag
        d_lots = new_lots - held
        cost = float(np.abs(d_lots) * per_lot / eq @ spread_frac[t]) if eq > 0 else 0.0
        eq *= 1.0 - cost
        held = new_lots
        lots[t] = held
        w_act[t] = held * per_lot / eq if eq > 0 else 0.0
        equity[t] = eq
        if eq <= 0:  # blown account: freeze
            lots[t:] = 0.0
            w_act[t:] = 0.0
            equity[t:] = eq
            break

    eq_series = pd.Series(equity, index=idx, name="equity")
    lots_df = pd.DataFrame(lots, index=idx, columns=symbols)
    return dict(
        equity=eq_series,
        lots=lots_df,
        actual_weights=pd.DataFrame(w_act, index=idx, columns=symbols),
        rounded_zero_frac={s: round(float(rounded_zero_bars[j] / n), 4)
                           for j, s in enumerate(symbols)},
        lot_changes=int((lots_df.diff().abs() > 1e-9).sum().sum()),
    )


def target_lots_today(weights: dict, equity: float, prices: dict) -> dict:
    """One-shot lot targets for the live runner (same rounding as the sim)."""
    out = {}
    for s, w in weights.items():
        per_lot = per_lot_usd(s, prices[s])
        ideal = w * equity / per_lot
        mag = _round_step(abs(ideal))
        rounded_zero = mag < VOL_MIN and abs(ideal) > 0
        lots = 0.0 if rounded_zero else math.copysign(mag, ideal)
        out[s] = dict(ideal_lots=round(ideal, 4), lots=lots,
                      per_lot_usd=per_lot, notional=round(lots * per_lot, 2),
                      rounded_zero=rounded_zero)
    return out
