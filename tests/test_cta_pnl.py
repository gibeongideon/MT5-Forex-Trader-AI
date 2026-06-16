"""Keystone tests — P&L correctness + lookahead guards. Must be green before backtests."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.cta.pnl import portfolio_pnl
from src.cta.signals import tsmom, ewmac
from src.cta.portfolio import inv_vol_weights, vol_target


def _frame(vals, cols=("A", "B")):
    idx = pd.date_range("2020-01-01", periods=len(vals), freq="D")
    return pd.DataFrame(vals, index=idx, columns=list(cols))


def test_known_pnl_value():
    # positions held over NEXT day's return; gross = pos[t-1]*ret[t]
    pos = _frame([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    ret = _frame([[0.0, 0.0], [0.10, 0.0], [-0.05, 0.0]])
    spread = _frame([[0, 0], [0, 0], [0, 0]])
    close = _frame([[100, 100], [110, 100], [104, 100]])
    pip = pd.Series({"A": 1e-4, "B": 1e-4})
    out = portfolio_pnl(pos, ret, spread, pip, close)
    # day1: pos[day0]=1 * ret[day1]=0.10 → +0.10 ; day2: 1 * -0.05 → -0.05
    assert abs(out["gross"].iloc[1] - 0.10) < 1e-12
    assert abs(out["gross"].iloc[2] - (-0.05)) < 1e-12
    assert pd.isna(out["gross"].iloc[0]) or out["gross"].iloc[0] == 0  # no prior pos


def test_lookahead_guard():
    # Shifting returns must change P&L — proves pos.shift(1) actually bites.
    pos = _frame([[1, 1]] * 6)
    ret = _frame(np.random.default_rng(0).normal(0, 0.01, (6, 2)))
    spread = _frame(np.zeros((6, 2))); close = _frame(np.full((6, 2), 100.0))
    pip = pd.Series({"A": 1e-4, "B": 1e-4})
    base = portfolio_pnl(pos, ret, spread, pip, close)["gross"]
    shifted = portfolio_pnl(pos, ret.shift(1), spread, pip, close)["gross"]
    assert not np.allclose(base.fillna(0), shifted.fillna(0)), "P&L invariant to return shift → lookahead!"


def test_zero_signal_zero_pnl():
    pos = _frame(np.zeros((5, 2)))
    ret = _frame(np.random.default_rng(1).normal(0, 0.01, (5, 2)))
    spread = _frame(np.ones((5, 2))); close = _frame(np.full((5, 2), 100.0))
    pip = pd.Series({"A": 1e-4, "B": 1e-4})
    out = portfolio_pnl(pos, ret, spread, pip, close)
    assert abs(out["net"].fillna(0).sum()) < 1e-12
    assert abs(out["turnover"].fillna(0).sum()) < 1e-12


def test_cost_monotonic():
    pos = _frame([[0, 0], [1, 0], [0, 0]])   # one round trip
    ret = _frame(np.zeros((3, 2))); close = _frame(np.full((3, 2), 100.0))
    pip = pd.Series({"A": 1e-4, "B": 1e-4})
    s1 = portfolio_pnl(pos, ret, _frame([[1, 1]] * 3), pip, close)["net"].sum()
    s2 = portfolio_pnl(pos, ret, _frame([[2, 2]] * 3), pip, close)["net"].sum()
    assert s2 < s1 < 1e-9 and abs(s2 - 2 * s1) < 1e-9  # double spread → double cost


def test_tsmom_trend():
    # a clean uptrend → tsmom = +1 once enough history
    close = pd.DataFrame({"A": np.arange(1, 400) * 1.0},
                         index=pd.date_range("2019-01-01", periods=399, freq="D"))
    s = tsmom(close)
    assert s["A"].iloc[-1] == 1.0


def test_ewmac_trend_and_cap():
    n = 500
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    # clean uptrend + small noise → forecast should be strongly positive, within cap
    up = pd.DataFrame({"A": 100 * (1.0008 ** np.arange(n))
                            + np.random.default_rng(0).normal(0, 0.1, n)}, index=idx)
    f = ewmac(up)
    assert f["A"].iloc[-1] > 0.3, f"uptrend forecast too weak: {f['A'].iloc[-1]}"
    assert f["A"].abs().max() <= 2.0 + 1e-9, "forecast exceeds ±cap/target (±2.0)"
    # flat/choppy mean-reverting series → forecast near zero on average
    rng = np.random.default_rng(1)
    chop = pd.DataFrame({"A": 100 + np.cumsum(rng.normal(0, 0.01, n)) * 0
                              + rng.normal(0, 0.5, n)}, index=idx)
    fc = ewmac(chop)
    assert abs(fc["A"].iloc[60:].mean()) < 0.5, "choppy forecast not near zero"


def test_vol_target_hits_target():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2018-01-01", periods=1500, freq="D")
    ret = pd.DataFrame(rng.normal(0, 0.01, (1500, 3)), index=idx, columns=list("ABC"))
    sig = pd.DataFrame(1.0, index=idx, columns=list("ABC"))
    w = inv_vol_weights(sig, ret)
    pos = vol_target(w, ret, target=0.10)
    realized = (pos.shift(1) * ret).sum(axis=1).iloc[400:].std() * np.sqrt(252)
    assert 0.06 < realized < 0.16, f"realized vol {realized:.3f} far from 0.10 target"
