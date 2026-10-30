"""Two-sample statistics on per-epoch alpha power (EC vs EO).

The unit of observation is one clean epoch. Epochs are non-overlapping, but
neighbouring epochs of the same recording are not strictly independent (slow
drifts in arousal, electrode impedance), so p-values should be read as a
within-session, single-subject test, not as population inference.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mann_whitney_greater(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """One-sided Mann-Whitney U test of H1: X stochastically greater than Y.

    Returns (U, p). U counts pairs with x > y (+0.5 for ties); exact p-values
    for small samples without ties, normal approximation with tie correction
    otherwise (SciPy ``method='auto'``).
    """
    res = stats.mannwhitneyu(x, y, alternative="greater", method="auto")
    return float(res.statistic), float(res.pvalue)


def cliffs_delta(x: np.ndarray, y: np.ndarray, u: float | None = None) -> float:
    """Cliff's delta = P(X > Y) - P(X < Y), in [-1, 1] (Cliff, 1993).

    Equivalent to the rank-biserial correlation: delta = 2 U / (n_x n_y) - 1.
    """
    x, y = np.asarray(x), np.asarray(y)
    if u is None:
        u = stats.mannwhitneyu(x, y, alternative="greater").statistic
    return float(2.0 * u / (x.size * y.size) - 1.0)


def permutation_test_mean(
    x: np.ndarray, y: np.ndarray, n_perm: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """One-sided permutation test of mean(x) - mean(y) > 0.

    The p-value uses the (b + 1) / (m + 1) estimator so it is never zero
    (Phipson & Smyth, 2010). Returns (observed difference, p).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = x.mean() - y.mean()
    pool = np.concatenate([x, y])
    rng = np.random.default_rng(seed)
    nx = x.size
    count = 0
    chunk = 1000
    for start in range(0, n_perm, chunk):
        m = min(chunk, n_perm - start)
        perm = pool[np.argsort(rng.random((m, pool.size)), axis=1)]
        diff = perm[:, :nx].mean(axis=1) - perm[:, nx:].mean(axis=1)
        count += int(np.sum(diff >= obs - 1e-12))
    return float(obs), (count + 1) / (n_perm + 1)


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (float(centre - half), float(centre + half))
