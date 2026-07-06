"""V5 lever pipeline — the single shared position builder for basket strategies.

Backtest (`scripts/v5_universe_backtest.py`), tests, and the live runner
(`scripts/v5_basket_runner.py`) all build positions through `lever_positions`
so the deployed book is byte-identical to the validated backtest. Never use
`src.cta.strategy.champion_positions` in new code: its buffer band width is a
full-sample statistic (lookahead); this pipeline ends in the V5 causal buffer.

Pre-registered lever parameters (declared before the Phase A runs; no sweeps):

  carry_w      = 0.5    50/50 blend of carry into FX-leg signals only
  ml_min_rows  = 2500   pooled rows before the first Ridge fit (default 8000
                        is unreachable early on a 5-instrument panel)
  ml_purge_mult= 2.0    label purge width vs the 1.6 default (conservative)
  regime gates use the module defaults in src/cta/regime.py

Every lever is past-only: regime gates shift(1), carry ffills monthly rates
known at month start, ml_forecast trains on a purged expanding window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.cta.ml_combine import ml_forecast
from src.cta.regime import regime_gate
from src.cta.signals import combine, ewmac, fx_carry, xsmom
from src.cta.strategy import rebalance_hold
from src.cta.universe import FX_PAIRS
from src.v5.h4_cta import buffer_band_causal, cluster_inv_vol, vol_target_h4

ANN = np.sqrt(252)
RATES_FILE = Path("data/rates_3m.csv")

SPEED_SETS = {
    "fast": ((8, 32), (16, 64), (32, 128), (64, 256)),
    "slow": ((32, 128), (64, 256)),
}

DEFAULT_CFG = dict(
    speeds="slow",
    sleeve="combined",
    rebalance="monthly",
    buffer_frac=0.4,
    target_vol=0.10,
    vol_halflife=42,
    regime="none",
    carry=False,
    carry_w=0.5,
    ml_combine=False,
    ml_min_rows=2500,
    ml_purge_mult=2.0,
)


def load_rates(path: str | Path = RATES_FILE) -> pd.DataFrame:
    """Monthly 3m rates panel indexed by date (columns = currency codes)."""
    rates = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    return rates


def carry_signal(close: pd.DataFrame, kept: list, rates: pd.DataFrame) -> pd.DataFrame:
    """FX carry panel for the kept aliases; asserts required currencies exist."""
    needed = {ccy for a in kept if a in FX_PAIRS for ccy in FX_PAIRS[a]}
    missing = needed - set(rates.columns)
    if missing:
        raise ValueError(f"rates file missing currencies {sorted(missing)}")
    return fx_carry(close.index, rates, FX_PAIRS, kept)


def lever_positions(close: pd.DataFrame, kept: list, classes: dict,
                    cfg: dict | None = None) -> pd.DataFrame:
    """Champion position construction with optional pre-registered levers.

    With every lever at its default the output is byte-identical to the
    validated basket5 champion path (regression-guarded in
    tests/test_v5_levers.py).
    """
    c = {**DEFAULT_CFG, **(cfg or {})}
    returns = close.pct_change(fill_method=None)
    speeds = SPEED_SETS[c["speeds"]] if isinstance(c["speeds"], str) else c["speeds"]

    if c["ml_combine"]:
        sig = ml_forecast(close, returns, min_train_rows=c["ml_min_rows"],
                          purge_mult=c["ml_purge_mult"]).fillna(0.0)
    else:
        sig = ewmac(close, speeds=speeds)
        if c["sleeve"] == "combined":
            sig = combine(sig, xsmom(close))

    if c["carry"]:
        car = carry_signal(close, kept, load_rates())
        fx_cols = [a for a in kept if a in FX_PAIRS]
        # blend on FX columns only: fx_carry is zero for non-FX aliases and a
        # whole-panel blend would halve every non-FX signal
        sig[fx_cols] = combine(sig[fx_cols], car[fx_cols], w=1.0 - c["carry_w"])

    sig = regime_gate(close, returns, sig, mode=c["regime"])

    raw = cluster_inv_vol(sig, returns, classes, c["target_vol"],
                          c["vol_halflife"], ann=ANN)
    pos = vol_target_h4(raw, returns, c["target_vol"], c["vol_halflife"], ann=ANN)
    pos = rebalance_hold(pos, c["rebalance"])
    return buffer_band_causal(pos, c["buffer_frac"])
