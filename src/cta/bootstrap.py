"""Block-bootstrap CI on annualized Sharpe (single source of truth).

Block resampling (~21d) respects daily-return autocorrelation — important for a
momentum strategy where returns are serially dependent.
"""
from __future__ import annotations
import numpy as np


def block_bootstrap_sharpe(daily, block: int = 21, n: int = 10000,
                           ppy: int = 252, seed: int = 42):
    d = np.asarray(daily, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < block * 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(len(d) / block))
    smax = len(d) - block
    shs = []
    for _ in range(n):
        starts = rng.integers(0, smax + 1, nblocks)
        samp = np.concatenate([d[s:s + block] for s in starts])[:len(d)]
        sd = samp.std(ddof=1)
        if sd > 1e-12:
            shs.append(samp.mean() / sd * np.sqrt(ppy))
    if not shs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(shs, 2.5)), float(np.percentile(shs, 97.5)))
