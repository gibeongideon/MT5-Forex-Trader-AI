"""Causality + correctness guards for the XAUUSD probabilistic-sizing program.

Covers the new modules: xau_exog (exogenous proxy features), vol_forecast
(EWMA-RV / HAR-RV), prob_sizing (risk-multiplier maps), dsr_pbo (statistics),
and xau_meta_oos (fold-local OOS ensemble). The overriding contract is that no
feature/probability at bar t may change when a strictly-future bar is mutated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.dsr_pbo import (deflated_sharpe_ratio, pbo_cscv,
                                    probabilistic_sharpe_ratio)
from src.features.vol_forecast import (ewma_vol, har_fit_predict,
                                       har_rv_features, vol_target_scale)
from src.features.xau_exog import add_xau_exog_features
from src.v5.prob_sizing import kelly_mult, prob_gate, prob_to_risk_mult
from src.v5.xau_meta_oos import MetaOOSConfig, generate_meta_oos


def _h4(n=4000, seed=11):
    rng = np.random.default_rng(seed)
    close = 1800.0 * np.exp(np.cumsum(rng.normal(1e-4, 6e-3, n)))
    idx = pd.date_range("2016-01-01", periods=n, freq="4h")
    o = np.roll(close, 1); o[0] = close[0]
    h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 2e-3, n)))
    l = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 2e-3, n)))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": close,
                         "spread": 3.0, "tick_volume": 1000.0}, index=idx)


def _fx_csvs(tmp_path):
    """Write minimal H4_long CSVs so xau_exog has proxy legs to load."""
    idx = pd.date_range("2016-01-01", periods=4000, freq="4h")
    for sym, seed in (("USDJPY", 1), ("EURUSD", 2), ("GBPUSD", 3)):
        rng = np.random.default_rng(seed)
        close = np.exp(np.cumsum(rng.normal(0, 4e-3, len(idx))))
        pd.DataFrame({"time": idx, "open": close, "high": close * 1.001,
                      "low": close * 0.999, "close": close, "spread": 1.0}
                     ).to_csv(tmp_path / f"{sym}_H4_long.csv", index=False)
    return tmp_path


# ── xau_exog causality ────────────────────────────────────────────────────────
def test_exog_future_mutation_leaves_past_identical(tmp_path):
    data_dir = _fx_csvs(tmp_path)
    xau = _h4()
    base = add_xau_exog_features(xau, data_dir=data_dir)
    cut = 3000
    xau2 = xau.copy()
    xau2.iloc[cut:] *= 1.05                       # mutate the future
    mutated = add_xau_exog_features(xau2, data_dir=data_dir)
    pd.testing.assert_frame_equal(base.iloc[:cut], mutated.iloc[:cut])


def test_exog_runs_without_csvs():
    xau = _h4()
    out = add_xau_exog_features(xau, data_dir="/nonexistent")
    assert len(out) == len(xau)
    assert (out["usd_strength_ret_1"] == 0.0).all()   # graceful zero-fill


# ── vol_forecast causality ────────────────────────────────────────────────────
def test_ewma_vol_is_past_only():
    xau = _h4()
    base = ewma_vol(xau["close"])
    cut = 3000
    c2 = xau["close"].copy(); c2.iloc[cut:] *= 1.1
    mut = ewma_vol(c2)
    pd.testing.assert_series_equal(base.iloc[:cut], mut.iloc[:cut])


def test_har_features_and_fit_are_past_only():
    xau = _h4()
    feat = har_rv_features(xau["close"])
    # regressors must be strictly lagged: rv_d at t excludes bar t's move
    cut = 3000
    c2 = xau["close"].copy(); c2.iloc[cut:] *= 1.1
    feat2 = har_rv_features(c2)
    pd.testing.assert_series_equal(feat["rv_d"].iloc[:cut], feat2["rv_d"].iloc[:cut])
    mask = pd.Series(feat.index < xau.index[cut], index=feat.index)
    pred = har_fit_predict(feat, mask)
    assert (pred.dropna() >= 0).all()             # volatility non-negative


def test_vol_target_scale_bounds():
    sigma = pd.Series([0.0, 0.01, 0.02, np.nan, 1e9])
    m = vol_target_scale(sigma, 0.01, floor=0.25, cap=3.0)
    assert (m >= 0.25).all() and (m <= 3.0).all()
    assert m.iloc[3] == 1.0                        # NaN -> neutral


# ── prob_sizing maps ──────────────────────────────────────────────────────────
def test_prob_to_risk_mult_monotone_and_clipped():
    p = pd.Series([0.0, 0.2, 0.5, 0.8, 1.0, np.nan])
    m = prob_to_risk_mult(p, lo=0.2, hi=0.8, out_lo=0.5, out_hi=1.5)
    assert m.iloc[0] == 0.5 and m.iloc[4] == 1.5
    assert m.iloc[1] < m.iloc[2] < m.iloc[3]
    assert m.iloc[5] == 1.0                        # NaN -> neutral


def test_prob_gate_and_kelly():
    p = pd.Series([0.3, 0.6, np.nan])
    g = prob_gate(p, 0.5)
    assert list(g[:2]) == [0.0, 1.0] and g.iloc[2] == 1.0
    k = kelly_mult(pd.Series([0.4, 0.7]), payoff_ratio=2.0, fraction=0.5, cap=1.5)
    assert (k >= 0).all() and (k <= 1.5).all()


# ── dsr / pbo ─────────────────────────────────────────────────────────────────
def test_psr_dsr_ranges():
    rng = np.random.default_rng(0)
    good = rng.normal(0.001, 0.01, 500)
    assert 0.0 <= probabilistic_sharpe_ratio(good) <= 1.0
    d = deflated_sharpe_ratio(good, np.array([0.02, 0.05, 0.1, 0.03]))
    assert d["n_trials"] == 4 and d["sr_benchmark"] >= 0.0


def test_pbo_bounds():
    rng = np.random.default_rng(1)
    M = rng.normal(0, 0.01, size=(400, 6))
    res = pbo_cscv(M, n_partitions=8)
    assert 0.0 <= res.pbo <= 1.0 and res.n_splits > 0


# ── meta OOS: strictly out-of-sample, no future leak ──────────────────────────
def test_meta_oos_is_out_of_sample():
    rng = np.random.default_rng(3)
    n = 800
    open_t = pd.date_range("2016-06-01", periods=n, freq="2D")
    close_t = open_t + pd.Timedelta(days=1)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series((rng.random(n) < 0.45).astype(int))
    res = generate_meta_oos(X, y, pd.Series(close_t), pd.Series(open_t),
                            MetaOOSConfig(start_year=2018, min_train=50))
    # pre-2018 trades never enter a test window → NaN (no in-sample scoring)
    pre = pd.Series(open_t).dt.year < 2018
    assert res.probs[pre.values].isna().all()
    assert res.probs[~pre.values].notna().any()
    assert np.isfinite(res.mean_auc) or res.n_folds == 0
