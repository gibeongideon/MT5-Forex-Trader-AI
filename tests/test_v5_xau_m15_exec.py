"""Leakage and regression guards for the M15-execution XAUUSD engine."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v5.xau_m15_exec import h4_features, ltf_trend, resample_h4, run_trades_m15

CONF = {"low": 0.5, "med": 1.0, "high": 1.5}


def _m15(n=30_000, seed=7, drift=1.5e-5):
    rng = np.random.default_rng(seed)
    close = 2000.0 * np.exp(np.cumsum(rng.normal(drift, 8e-4, n)))
    idx = pd.date_range("2022-01-04", periods=n, freq="15min")
    o = np.roll(close, 1); o[0] = close[0]
    h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 3e-4, n)))
    l = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 3e-4, n)))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": close,
                         "spread": 3.4}, index=idx)


def test_forming_h4_bar_is_invisible():
    m15 = _m15(2000)
    f = h4_features(m15)
    h4 = resample_h4(m15)
    close_time = h4.index + pd.Timedelta(hours=4)
    ld = f["last_done"]
    for j in (0, 137, 1024, 1999):
        i = ld[j]
        if i >= 0:
            assert close_time[i] <= m15.index[j]
        if i + 1 < len(close_time):
            assert close_time[i + 1] > m15.index[j]


def test_future_mutation_leaves_earlier_decisions_identical():
    m15 = _m15()
    cutoff = 24_000
    for kwargs in (dict(limit_k=None), dict(limit_k=0.5),
                   dict(limit_k=None, trail_source="m15"),
                   dict(limit_k=None, confirm_rule="1h")):
        base = run_trades_m15(m15.iloc[:cutoff], params={"conf_risk_scale": CONF},
                              **kwargs)
        mut = m15.copy()
        mut.iloc[cutoff:] *= 1.3
        changed = run_trades_m15(mut.iloc[:cutoff],
                                 params={"conf_risk_scale": CONF}, **kwargs)
        pd.testing.assert_frame_equal(base["trades"], changed["trades"])


def test_trail_never_seeds_from_pre_entry_extreme():
    """Regression: a huge favorable extreme BEFORE entry must not activate
    the trail instantly (caught live by the E1<->E0 reconciliation)."""
    n = 6000
    idx = pd.date_range("2022-01-04", periods=n, freq="15min")
    close = np.full(n, 2000.0) + np.linspace(0, 60, n)      # slow uptrend
    spike = 5600                                             # deep dip pre-entry
    close[spike - 32:spike] = 1900.0                         # prior H4 windows
    o = np.roll(close, 1); o[0] = close[0]
    h = np.maximum(o, close) + 0.3
    l = np.minimum(o, close) - 0.3
    m15 = pd.DataFrame({"open": o, "high": h, "low": l, "close": close,
                        "spread": 3.4}, index=idx)
    res = run_trades_m15(m15, params={"conf_risk_scale": CONF})
    t = res["trades"]
    if len(t):
        dur = (t["close_time"] - t["open_time"]).dt.total_seconds() / 3600
        fast_trails = t[(t["exit_reason"] == "trail_stop") & (dur < 4)]
        assert len(fast_trails) == 0, fast_trails.to_string()


def test_limit_ttl_converts_to_market():
    m15 = _m15(30_000, seed=3)                 # enough H4 bars for EWMAC warmup
    base = run_trades_m15(m15, params={"conf_risk_scale": CONF})
    assert len(base["trades"]) > 0             # sanity: signal fires at all
    res = run_trades_m15(m15, limit_k=5.0,     # unfillable limit -> always TTL
                         params={"conf_risk_scale": CONF})
    assert len(res["trades"]) > 0              # entries still happen via TTL


def test_costs_hurt():
    m15 = _m15()
    cheap = run_trades_m15(m15, params={"conf_risk_scale": CONF,
                                        "spread_cost_mult": 0.5})
    dear = run_trades_m15(m15, params={"conf_risk_scale": CONF,
                                       "spread_cost_mult": 3.0})
    assert dear["equity"].dropna().iloc[-1] < cheap["equity"].dropna().iloc[-1]


def test_ltf_trend_is_completed_bar_aligned():
    m15 = _m15(3000)
    trend, last_done = ltf_trend(m15, "1h")
    mut = m15.copy()
    mut.iloc[2500:] *= 1.5
    trend2, _ = ltf_trend(mut, "1h")
    # completed 1H bars strictly before the mutation are unchanged
    n_safe = int((last_done[2500] or 0))
    assert np.array_equal(trend[:n_safe], trend2[:n_safe], equal_nan=True)


@pytest.mark.skipif(not Path("data/XAUUSD_M15_spliced.csv").exists(),
                    reason="spliced data not present")
def test_data_gate_spliced_matches_h4_csv():
    m15 = pd.read_csv("data/XAUUSD_M15_spliced.csv", parse_dates=["time"],
                      index_col="time").sort_index()
    h4csv = pd.read_csv("data/XAUUSD_H4_long.csv", parse_dates=["time"],
                        index_col="time").sort_index()
    r = resample_h4(m15)
    both = r.join(h4csv[["open", "high", "low", "close"]], rsuffix="_csv",
                  how="inner")
    win = (both.index >= "2023-04-12") & (both.index < "2023-09-30")
    d = np.abs(both[["open", "high", "low", "close"]].values
               - both[["open_csv", "high_csv", "low_csv", "close_csv"]].values
               ).max(axis=1)
    assert (d[~win] > 0.01).sum() == 0
