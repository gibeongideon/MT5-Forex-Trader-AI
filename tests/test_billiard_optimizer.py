"""
Tests for BilliardOptimizer — no MT5 / bridge required.

Run:
    conda run -n envmt5 python -m pytest tests/test_billiard_optimizer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.billiard_optimizer import BilliardOptimizer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sphere(x: np.ndarray) -> float:
    """Sphere function: max at x=0. Returns -sum(x^2) so maximum = 0."""
    return -float(np.sum(x ** 2))


def _neg_sphere(x: np.ndarray) -> float:
    """Returns sum(x^2) — used with minimize()."""
    return float(np.sum(x ** 2))


def _unimodal_peak(x: np.ndarray) -> float:
    """Single peak at x=[0.5, 0.5]. Returns -(sum((x-0.5)^2))."""
    return -float(np.sum((x - 0.5) ** 2))


def _constant(x: np.ndarray) -> float:
    return 1.0


# ── Basic API ─────────────────────────────────────────────────────────────────

class TestBilliardOptimizerAPI:

    def test_returns_tuple(self):
        opt = BilliardOptimizer(n_agents=10, n_pockets=5, n_iters=5, seed=0)
        result = opt.optimize(_sphere, [(-1.0, 1.0), (-1.0, 1.0)])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_array_and_float(self):
        opt = BilliardOptimizer(n_agents=10, n_pockets=5, n_iters=5, seed=0)
        params, score = opt.optimize(_sphere, [(-1.0, 1.0)])
        assert isinstance(params, np.ndarray)
        assert isinstance(score, float)

    def test_params_length_matches_bounds(self):
        bounds = [(-2.0, 2.0), (0.0, 1.0), (-5.0, 5.0)]
        opt = BilliardOptimizer(n_agents=10, n_pockets=5, n_iters=5, seed=0)
        params, _ = opt.optimize(_sphere, bounds)
        assert len(params) == len(bounds)

    def test_n_pockets_gt_n_agents_raises(self):
        with pytest.raises(ValueError):
            BilliardOptimizer(n_agents=10, n_pockets=20)


# ── Bounds ────────────────────────────────────────────────────────────────────

class TestBoundsRespected:

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_params_within_bounds(self, seed):
        bounds = [(-1.0, 1.0), (2.0, 5.0), (-10.0, -5.0)]
        opt = BilliardOptimizer(n_agents=20, n_pockets=10, n_iters=20, seed=seed)
        params, _ = opt.optimize(_sphere, bounds)
        for i, (lo, hi) in enumerate(bounds):
            assert lo <= params[i] <= hi, (
                f"Param {i}={params[i]:.4f} out of bounds [{lo}, {hi}]"
            )

    def test_single_dim_within_bounds(self):
        bounds = [(3.0, 7.0)]
        opt = BilliardOptimizer(n_agents=10, n_pockets=5, n_iters=10, seed=0)
        params, _ = opt.optimize(lambda x: -abs(x[0] - 5.0), bounds)
        assert 3.0 <= params[0] <= 7.0


# ── Optimization quality ──────────────────────────────────────────────────────

class TestOptimizationQuality:

    def test_sphere_finds_near_zero(self):
        bounds = [(-5.0, 5.0)] * 3
        opt = BilliardOptimizer(n_agents=50, n_pockets=25, n_iters=200, seed=42)
        params, score = opt.optimize(_sphere, bounds)
        # Best score should be close to 0 (sphere max = 0 at origin)
        assert score > -1.0, f"score={score:.4f}, expected > -1.0"

    def test_unimodal_peak_near_05(self):
        bounds = [(0.0, 1.0)] * 2
        opt = BilliardOptimizer(n_agents=30, n_pockets=15, n_iters=150, seed=42)
        params, _ = opt.optimize(_unimodal_peak, bounds)
        for p in params:
            assert abs(p - 0.5) < 0.2, (
                f"param={p:.4f} too far from 0.5 (expected within ±0.2)"
            )

    def test_constant_objective_returns_constant(self):
        bounds = [(-1.0, 1.0)] * 2
        opt = BilliardOptimizer(n_agents=10, n_pockets=5, n_iters=10, seed=0)
        _, score = opt.optimize(_constant, bounds)
        assert score == pytest.approx(1.0)


# ── Minimize wrapper ──────────────────────────────────────────────────────────

class TestMinimize:

    def test_minimize_finds_minimum(self):
        bounds = [(-5.0, 5.0)] * 2
        opt = BilliardOptimizer(n_agents=30, n_pockets=15, n_iters=100, seed=0)
        params, score = opt.minimize(_neg_sphere, bounds)
        # Minimum of sum(x^2) is 0 at origin
        assert score < 2.0, f"score={score:.4f}, expected close to 0"

    def test_minimize_returns_positive_score(self):
        bounds = [(-1.0, 1.0)] * 2
        opt = BilliardOptimizer(n_agents=10, n_pockets=5, n_iters=10, seed=0)
        _, score = opt.minimize(_neg_sphere, bounds)
        assert score >= 0.0, "Sphere function is always >= 0"


# ── Reproducibility ───────────────────────────────────────────────────────────

class TestReproducibility:

    def test_same_seed_same_result(self):
        bounds = [(-2.0, 2.0)] * 3
        opt1 = BilliardOptimizer(n_agents=20, n_pockets=10, n_iters=30, seed=7)
        opt2 = BilliardOptimizer(n_agents=20, n_pockets=10, n_iters=30, seed=7)
        p1, s1 = opt1.optimize(_sphere, bounds)
        p2, s2 = opt2.optimize(_sphere, bounds)
        np.testing.assert_array_equal(p1, p2)
        assert s1 == s2

    def test_different_seeds_may_differ(self):
        bounds = [(-2.0, 2.0)] * 3
        opt1 = BilliardOptimizer(n_agents=20, n_pockets=10, n_iters=30, seed=1)
        opt2 = BilliardOptimizer(n_agents=20, n_pockets=10, n_iters=30, seed=999)
        p1, _ = opt1.optimize(_sphere, bounds)
        p2, _ = opt2.optimize(_sphere, bounds)
        # Different seeds should (almost certainly) produce different paths
        # — not guaranteed to differ but is statistically expected
        # Just check no crash
        assert len(p1) == len(p2)


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_one_iteration(self):
        bounds = [(-1.0, 1.0)] * 2
        opt = BilliardOptimizer(n_agents=10, n_pockets=5, n_iters=1, seed=0)
        params, score = opt.optimize(_sphere, bounds)
        assert len(params) == 2
        assert np.isfinite(score)

    def test_one_dimension(self):
        bounds = [(-3.0, 3.0)]
        opt = BilliardOptimizer(n_agents=20, n_pockets=10, n_iters=50, seed=0)
        params, score = opt.optimize(lambda x: -(x[0] - 1.5) ** 2, bounds)
        assert abs(params[0] - 1.5) < 1.0

    def test_n_pockets_equals_n_agents(self):
        bounds = [(-1.0, 1.0)] * 2
        opt = BilliardOptimizer(n_agents=10, n_pockets=10, n_iters=5, seed=0)
        params, score = opt.optimize(_sphere, bounds)
        assert np.isfinite(score)
