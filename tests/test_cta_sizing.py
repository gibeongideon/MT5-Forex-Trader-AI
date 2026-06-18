"""Tests for src/cta/sizing.py — units → lots conversion."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cta.sizing import target_lots, min_viable_equity, gross_exposure


def _spec(cs, price, vmin=0.01, vstep=0.01, vmax=1e6, sym="X"):
    return dict(symbol=sym, contract_size=cs, price=price, vol_min=vmin, vol_step=vstep, vol_max=vmax)


def test_basic_conversion():
    # pos 0.5 of $100k = $50k notional; 1 lot = 100*100 = $10k → 5 lots
    r = target_lots({"A": 0.5}, 100_000, {"A": _spec(100, 100)})["A"]
    assert r["ideal_lots"] == 5.0 and r["lots"] == 5.0
    assert r["target_notional"] == 50_000 and r["actual_notional"] == 50_000


def test_sign_preserved():
    r = target_lots({"A": -0.5}, 100_000, {"A": _spec(100, 100)})["A"]
    assert r["lots"] == -5.0 and r["actual_notional"] == -50_000


def test_rounds_to_zero_flagged():
    # tiny exposure → below min lot → rounds to 0 and is flagged
    r = target_lots({"A": 0.0005}, 10_000, {"A": _spec(100, 100)})["A"]
    assert r["lots"] == 0.0 and r["rounded_zero"] is True


def test_step_rounding():
    # ideal 3.327 lots with 0.01 step → 3.33
    r = target_lots({"A": 1.0}, 33_270, {"A": _spec(100, 100)})["A"]
    assert r["lots"] == 3.33


def test_vol_max_cap():
    r = target_lots({"A": 10.0}, 1_000_000, {"A": _spec(100, 100, vmax=2.0)})["A"]
    assert r["lots"] == 2.0 and r["capped"] is True


def test_min_viable_equity():
    # leg |u|=0.05, per_lot=$10k, vmin=0.01 → need 0.01*10000/0.05 = $2,000
    specs = {"A": _spec(100, 100)}
    assert abs(min_viable_equity({"A": 0.05}, specs) - 2000.0) < 1e-6


def test_gross_exposure():
    res = target_lots({"A": 0.5, "B": -0.3}, 100_000,
                      {"A": _spec(100, 100), "B": _spec(100, 100)})
    g = gross_exposure(res)
    assert abs(g["gross_notional"] - 80_000) < 1 and abs(g["net_notional"] - 20_000) < 1
