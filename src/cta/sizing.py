"""sizing.py — convert vol-scaled CTA units → broker lots.

The engine's position `pos_i` IS the signed notional exposure as a fraction of equity
(portfolio return = Σ pos_i · return_i). So:

    notional_i = pos_i · equity                       # signed USD exposure
    lots_i     = notional_i / (contract_size_i · price_i)   # 1 lot = contract_size × price USD

then rounded to the broker's volume step and clamped to [volume_min, volume_max]. Lot
granularity (min 0.01) distorts the vol target on small accounts — `target_lots` reports the
rounding error per leg and flags legs that round to zero, so the distortion is never silent.

Pure function (specs passed in) → unit-testable without a broker; the runner fills `specs`
either offline (panel close + default contract sizes) or live (connector symbol_info + tick).
"""
from __future__ import annotations
import math

# Best-effort HFM contract sizes (1 lot = contract_size units of the underlying).
# VERIFY against the live terminal (symbol_info.trade_contract_size) before trading — broker/account specific.
DEFAULT_CONTRACT = {
    "XAUUSD":   100.0,     # 100 oz
    "XAGUSD":   1000.0,    # 1000 oz
    "EURUSD":   100000.0,  # 100k base ccy
    "USDJPY":   100000.0,  # 100k base ccy (USD-base: USD notional = contract, not x price)
    "US500.F":  1.0,       # $1 / index point
    "USOIL":    100.0,     # 100 barrels
    "US10YR.F": 100.0,     # HFM: 1 lot = 100 units, volume_min = 1.0
    "#BTCUSD":  1.0,       # 1 BTC
}
DEFAULT_VOL = dict(vol_min=0.01, vol_step=0.01, vol_max=1e6)


def _round_step(x: float, step: float) -> float:
    return round(round(x / step) * step, 8)


def target_lots(units: dict, equity: float, specs: dict) -> dict:
    """units: {key: signed vol-scaled position}; specs[key]={symbol,contract_size,price,vol_min,
    vol_step,vol_max}. Returns per-key dict with ideal/rounded lots, notional (target vs actual),
    and rounding/cap flags."""
    out = {}
    for k, u in units.items():
        s = specs[k]
        per_lot = s["contract_size"] * s["price"]                 # USD notional of 1.0 lot
        target_notional = u * equity                              # signed
        ideal = target_notional / per_lot if per_lot > 0 else 0.0
        step = s.get("vol_step", 0.01); vmin = s.get("vol_min", 0.01); vmax = s.get("vol_max", 1e6)
        mag = _round_step(abs(ideal), step)
        rounded_zero = mag < vmin and abs(ideal) > 0
        capped = mag > vmax
        mag = 0.0 if rounded_zero else min(mag, vmax)
        lots = math.copysign(mag, ideal) if mag > 0 else 0.0
        actual_notional = lots * per_lot
        out[k] = dict(
            symbol=s.get("symbol", k), ideal_lots=round(ideal, 4), lots=round(lots, 2),
            per_lot_notional=per_lot, target_notional=target_notional,
            actual_notional=actual_notional,
            err_frac=(actual_notional - target_notional) / equity if equity else 0.0,
            rounded_zero=rounded_zero, capped=capped,
        )
    return out


def min_viable_equity(units: dict, specs: dict) -> float:
    """Smallest equity at which EVERY non-zero leg reaches at least 1 min-lot (below this the
    smallest-weight legs round to zero and the vol target breaks)."""
    needs = []
    for k, u in units.items():
        if abs(u) < 1e-9:
            continue
        s = specs[k]
        per_lot = s["contract_size"] * s["price"]
        vmin = s.get("vol_min", 0.01)
        needs.append(vmin * per_lot / abs(u))          # equity s.t. |u|*eq = vmin*per_lot
    return max(needs) if needs else 0.0


def gross_exposure(lots_result: dict) -> dict:
    """Aggregate $ exposure + gross leverage from a target_lots result."""
    gross = sum(abs(v["actual_notional"]) for v in lots_result.values())
    net = sum(v["actual_notional"] for v in lots_result.values())
    return dict(gross_notional=gross, net_notional=net)
