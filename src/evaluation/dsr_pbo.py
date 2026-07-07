"""Deflated Sharpe Ratio + Probability of Backtest Overfitting — V5.

Fills a gap: the repo had bootstrap CIs (`src/cta/bootstrap.py`) but no
multiple-testing / overfitting correction. Both metrics here are from
Bailey & López de Prado and are what `AGENT_INSTRUCTIONS.MD` Phase 6 asks for.

  deflated_sharpe_ratio  — probability the true Sharpe > 0 after correcting the
                           observed SR for non-normal returns (skew/kurtosis),
                           sample length, and the number of trials that were run
                           to find it (variance of the trial SRs).
  pbo_cscv               — Combinatorially-Symmetric Cross-Validation estimate
                           of P(the config picked as best in-sample is below
                           median out-of-sample) = probability of overfitting.

Neither needs statsmodels; only numpy/scipy.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import norm


def _sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 1e-12 else 0.0


def probabilistic_sharpe_ratio(
    returns: np.ndarray, sr_benchmark: float = 0.0
) -> float:
    """P(true SR > sr_benchmark) for the observed (non-annualized) return series.

    Uses the skew/kurtosis-adjusted standard error of the Sharpe estimator
    (Bailey & López de Prado 2012). `sr_benchmark` and the returned SR are in the
    SAME per-observation units as `returns`.
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 8:
        return float("nan")
    sr = _sharpe(r)
    sd = r.std(ddof=1)
    skew = float(((r - r.mean()) ** 3).mean() / sd ** 3) if sd > 1e-12 else 0.0
    kurt = float(((r - r.mean()) ** 4).mean() / sd ** 4) if sd > 1e-12 else 3.0
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    returns: np.ndarray, trial_sharpes: np.ndarray
) -> dict:
    """Deflated Sharpe Ratio: PSR against a benchmark inflated by multiple trials.

    The benchmark SR is the expected maximum of `n_trials` independent trial SRs
    given their observed cross-sectional variance (Bailey & López de Prado 2014).
    `returns` and the trial SRs must be in the same per-observation units.
    Returns {dsr, sr_benchmark, n_trials}.
    """
    trials = np.asarray(trial_sharpes, float)
    trials = trials[np.isfinite(trials)]
    m = len(trials)
    var_sr = float(trials.var(ddof=1)) if m > 1 else 0.0
    if m <= 1 or var_sr <= 0:
        sr_star = 0.0
    else:
        euler = 0.5772156649015329
        e_max = ((1 - euler) * norm.ppf(1 - 1.0 / m)
                 + euler * norm.ppf(1 - 1.0 / (m * np.e)))
        sr_star = np.sqrt(var_sr) * e_max
    return {
        "dsr": probabilistic_sharpe_ratio(returns, sr_benchmark=sr_star),
        "sr_benchmark": float(sr_star),
        "n_trials": int(m),
    }


@dataclass
class PBOResult:
    pbo: float
    n_splits: int
    logits: list


def pbo_cscv(perf_matrix: np.ndarray, n_partitions: int = 10) -> PBOResult:
    """Probability of Backtest Overfitting via CSCV (López de Prado 2015).

    `perf_matrix` shape (T, N): T time observations (e.g. per-bar returns) for
    each of N candidate configurations. Rows are split into `n_partitions` equal
    blocks; for every way to choose half the blocks as IS and the rest as OOS,
    pick the best config IS and record its OOS rank. PBO = fraction of splits
    where the IS-best lands below the OOS median (logit < 0).
    """
    M = np.asarray(perf_matrix, float)
    T, N = M.shape
    if N < 2:
        return PBOResult(pbo=float("nan"), n_splits=0, logits=[])
    S = n_partitions - (n_partitions % 2)
    if S < 2:
        S = 2
    edges = np.array_split(np.arange(T), S)
    blocks = [M[e, :] for e in edges if len(e) > 0]
    S = len(blocks)
    logits: list[float] = []
    for is_idx in combinations(range(S), S // 2):
        is_set = set(is_idx)
        J = np.vstack([blocks[i] for i in range(S) if i in is_set])
        Jbar = np.vstack([blocks[i] for i in range(S) if i not in is_set])
        is_perf = np.array([_sharpe(J[:, c]) for c in range(N)])
        oos_perf = np.array([_sharpe(Jbar[:, c]) for c in range(N)])
        best = int(np.argmax(is_perf))
        # OOS relative rank of the IS-best config in [1/(N+1), N/(N+1)]
        rank = (np.sum(oos_perf <= oos_perf[best])) / (N + 1.0)
        rank = min(max(rank, 1.0 / (N + 1.0)), N / (N + 1.0))
        logits.append(float(np.log(rank / (1.0 - rank))))
    if not logits:
        return PBOResult(pbo=float("nan"), n_splits=0, logits=[])
    pbo = float(np.mean([1.0 if lg < 0 else 0.0 for lg in logits]))
    return PBOResult(pbo=pbo, n_splits=len(logits), logits=logits)
