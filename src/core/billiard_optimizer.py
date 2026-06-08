"""
Billiards Optimization Algorithm (BOA) — modified variant — based on READ 10.

Population-based optimizer for small/medium dimensional problems (3–10 params).
Benchmark rank 8/45 at 62.19% — solid choice for hyperparameter tuning.

Modified update rule (BOAm, from Andrey Dik, Jan 2026):
    X_new = X + rnd[0,1] × (Pocket − X) × I
    where I ∈ {1, 2} chosen uniformly at random

Each "ball" (agent) moves toward a randomly chosen "pocket" (top-N best
solutions seen so far), scaled by a random factor I that either tracks or
overshoots the target.

Usage
-----
    from src.core.billiard_optimizer import BilliardOptimizer

    opt = BilliardOptimizer(n_agents=50, n_pockets=25, n_iters=100)

    def objective(params):
        lr, depth, subsample = params
        # return cross-val score (higher is better)
        ...

    bounds = [(0.01, 0.3), (3, 10), (0.5, 1.0)]
    best_params, best_score = opt.optimize(objective, bounds)
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


class BilliardOptimizer:
    """
    Modified Billiards Optimization Algorithm (BOAm).

    Parameters
    ----------
    n_agents : int
        Number of balls (population size). Default 50 matches article optimum.
    n_pockets : int
        Number of pockets (top-N solutions used as targets). Default 25.
    n_iters : int
        Number of optimization iterations.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_agents:  int            = 50,
        n_pockets: int            = 25,
        n_iters:   int            = 100,
        seed:      int | None     = None,
    ) -> None:
        if n_pockets > n_agents:
            raise ValueError(f"n_pockets ({n_pockets}) must be <= n_agents ({n_agents})")
        self.n_agents  = n_agents
        self.n_pockets = n_pockets
        self.n_iters   = n_iters
        self._rng      = np.random.default_rng(seed)

    # ── Public API ─────────────────────────────────────────────────────────────

    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float],
        bounds:       Sequence[tuple[float, float]],
    ) -> tuple[np.ndarray, float]:
        """
        Maximize `objective_fn` over the parameter space defined by `bounds`.

        Parameters
        ----------
        objective_fn : callable
            Function that takes a 1-D array of parameter values and returns a
            scalar score. Higher scores are better.
        bounds : list of (min, max) tuples
            One tuple per parameter. The optimizer clips all agents to these
            bounds at every iteration.

        Returns
        -------
        best_params : np.ndarray
            Parameter values achieving the highest score.
        best_score : float
            Objective value at `best_params`.
        """
        lo  = np.array([b[0] for b in bounds], dtype=float)
        hi  = np.array([b[1] for b in bounds], dtype=float)
        dim = len(bounds)

        # Initialise agents uniformly within bounds
        agents = self._rng.uniform(lo, hi, size=(self.n_agents, dim))
        scores = np.array([float(objective_fn(a)) for a in agents])

        # Personal bests (each agent remembers its own best position/score)
        best_agents = agents.copy()
        best_scores = scores.copy()

        for _ in range(self.n_iters):
            # Top n_pockets personal bests become the pocket targets
            pocket_ids = np.argsort(best_scores)[-self.n_pockets:]

            for i in range(self.n_agents):
                # Pick a random pocket
                p_id   = self._rng.choice(pocket_ids)
                pocket = best_agents[p_id]

                # BOAm update: X_new = X + rnd × (Pocket − X) × I
                I         = self._rng.choice([1, 2])
                r         = self._rng.random()
                agent_new = agents[i] + r * (pocket - agents[i]) * I
                agent_new = np.clip(agent_new, lo, hi)

                score = float(objective_fn(agent_new))
                if score > best_scores[i]:
                    best_agents[i] = agent_new.copy()
                    best_scores[i] = score

            # All agents move to their personal best positions each iteration
            agents = best_agents.copy()

        best_idx = int(np.argmax(best_scores))
        return best_agents[best_idx].copy(), float(best_scores[best_idx])

    def minimize(
        self,
        objective_fn: Callable[[np.ndarray], float],
        bounds:       Sequence[tuple[float, float]],
    ) -> tuple[np.ndarray, float]:
        """Minimize `objective_fn` (wraps `optimize` with negated function)."""
        neg_best, neg_score = self.optimize(
            lambda x: -float(objective_fn(x)), bounds
        )
        return neg_best, -neg_score
