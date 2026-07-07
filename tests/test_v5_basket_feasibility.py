"""Guards for the small-account basket feasibility harness.

Verifies the two load-bearing claims: (1) it reuses the validated causal engine
so a candidate's stats match a direct lever_positions/mtm run, and (2) Sharpe is
invariant to target_vol while min_viable_equity scales ~1/target_vol (the knob
the tiered recommendation relies on).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cta.sizing import min_viable_equity

# import the script module by path (scripts/ is not a package)
_spec = importlib.util.spec_from_file_location(
    "v5_basket_feasibility", ROOT / "scripts" / "v5_basket_feasibility.py")
feas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feas)


def _have_data(aliases):
    return all((ROOT / "data" / f"{a}_D1_long.csv").exists() for a in aliases)


CAND = ["GOLD", "UST10Y", "SPX"]
pytestmark = pytest.mark.skipif(not _have_data(CAND), reason="D1 panel data absent")


def test_sharpe_invariant_to_target_vol():
    """Doubling target_vol must not move the cost-adjusted Sharpe materially."""
    _, _, _, pnl_a = feas.backtest(CAND, "2012-01-01", 0.10)
    _, _, _, pnl_b = feas.backtest(CAND, "2012-01-01", 0.20)
    sa = pnl_a["net"].mean() / pnl_a["net"].std()
    sb = pnl_b["net"].mean() / pnl_b["net"].std()
    assert abs(sa - sb) < 0.02


def test_min_viable_scales_inversely_with_vol():
    """min_viable_equity(2x vol) ~= 0.5 * min_viable_equity(vol)."""
    kept, close, pos1, _ = feas.backtest(CAND, "2012-01-01", 0.10)
    _, _, pos2, _ = feas.backtest(CAND, "2012-01-01", 0.20)
    specs = feas.specs_for(kept, close)
    mve1, _, _ = feas.deployability(kept, close, pos1, specs, [10000])
    mve2, _, _ = feas.deployability(kept, close, pos2, specs, [10000])
    assert mve1 > 0 and mve2 > 0
    assert 0.4 < (mve2 / mve1) < 0.65        # ~0.5, allowing rounding/caps


def test_reuses_validated_engine_path():
    """A candidate's net series equals a direct lever_positions/mtm run."""
    from src.cta.panel import asset_classes, build_panels
    from src.v5.h4_cta import mtm_pnl_price_units
    from src.v5.levers import lever_positions

    kept, close, pos, pnl = feas.backtest(CAND, "2012-01-01", 0.10)
    c2, s2, k2 = build_panels(CAND, tf="D1")
    cfg = dict(feas.LEVER_BASE, target_vol=0.10)
    pos2 = lever_positions(c2, k2, asset_classes(k2), cfg)
    pnl2 = mtm_pnl_price_units(pos2, c2, s2).loc["2012-01-01":]
    pd.testing.assert_series_equal(pnl["net"], pnl2["net"])


def test_specs_cover_all_kept_legs():
    kept, close, _, _ = feas.backtest(CAND, "2012-01-01", 0.10)
    specs = feas.specs_for(kept, close)
    assert set(specs) == set(kept)
    for a in kept:
        assert specs[a]["contract_size"] > 0 and specs[a]["price"] > 0
