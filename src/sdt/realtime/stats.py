"""The statistics of sections 7 and 8 of the realtime-benchmark plan.

**Primary test.** For a series d_1, ..., d_T of differences d_t(2):

1. d-bar is the mean of the series.
2. The periodogram at frequency lambda is I(lambda) = |sum_t (d_t - d-bar) exp(-i lambda t)|^2 / (2 pi T).
3. The long-run variance is 2 pi times the mean of I(lambda_j) over the m lowest Fourier frequencies lambda_j = 2 pi j
   / T, j = 1, ..., m, with m the largest integer whose cube is at most T (2 for T = 26 and T = 19, 1 for T = 7).
4. The standard error is the square root of the long-run variance divided by T, and the statistic is d-bar divided by
   it, compared with a t-distribution with 2m degrees of freedom (fixed-m asymptotics; Coroneo and Iacone, 2020).
5. The test is two-sided at 5%, and the 95% interval is d-bar plus or minus the critical value times the standard
   error.

**Secondary analysis.** For the curve d-bar(h), h = 1 to 12, over common target months: rho is the Spearman
correlation of h and d-bar(h); the slope is the least-squares slope of d-bar(h) on h; the breakeven is the smallest h at
which the linearly interpolated curve turns positive, below 1 if d-bar(1) > 0 and above 12 if it never does. Their 95%
intervals come from one moving-block bootstrap of target months (blocks of 3, 1,000 draws, seed 0) that recomputes the
whole curve in each draw. The interval ends are order statistics of the draws (`inverted_cdf`), so a breakeven stored as
-inf (below 1) or +inf (above 12) sorts at the ends and leaves that end of the interval open.
"""
import numpy as np
from scipy import stats

LEVEL = 0.05
N_DRAWS, BLOCK, SEED = 1000, 3, 0


def bandwidth(T):
    """m: the largest integer whose cube is at most T."""
    m = int(round(T ** (1 / 3)))
    while m ** 3 > T:
        m -= 1
    while (m + 1) ** 3 <= T:
        m += 1
    return m


def daniell_variance(d):
    """The long-run variance of `d` and the bandwidth m (steps 2 and 3)."""
    d = np.asarray(d, float)
    T, m = len(d), bandwidth(len(d))
    lam = 2 * np.pi * np.arange(1, m + 1) / T
    dft = np.exp(-1j * np.outer(lam, np.arange(1, T + 1))) @ (d - d.mean())
    periodogram = np.abs(dft) ** 2 / (2 * np.pi * T)
    return 2 * np.pi * periodogram.mean(), m


def diebold_mariano(d):
    """Steps 1 to 5 for the series `d`."""
    d = np.asarray(d, float)
    if len(d) < 2 or np.isnan(d).any():
        raise ValueError('the test needs at least two differences and no missing ones')
    variance, m = daniell_variance(d)
    se, df = np.sqrt(variance / len(d)), 2 * m
    statistic = d.mean() / se
    critical = stats.t.ppf(1 - LEVEL / 2, df)
    return {'T': len(d), 'm': m, 'df': df, 'mean': d.mean(), 'se': se, 'statistic': statistic,
            'p': 2 * stats.t.sf(abs(statistic), df), 'critical': critical,
            'lo': d.mean() - critical * se, 'hi': d.mean() + critical * se}


def outcome(test):
    """The decision rule of section 7."""
    if test['p'] < LEVEL:
        return 'twins more accurate' if test['mean'] > 0 else 'simple models more accurate'
    return 'neither shown more accurate'


def breakeven(h, curve):
    """The smallest h at which the linearly interpolated curve turns positive: -inf if curve[0] > 0, +inf if never."""
    h, curve = np.asarray(h, float), np.asarray(curve, float)
    if curve[0] > 0:
        return -np.inf
    for j in range(len(curve) - 1):
        if curve[j] <= 0 < curve[j + 1]:
            return h[j] + (0 - curve[j]) / (curve[j + 1] - curve[j]) * (h[j + 1] - h[j])
    return np.inf


def summaries(h, curve):
    """rho, slope and breakeven of one curve."""
    return {'rho': stats.spearmanr(h, curve).statistic, 'slope': np.polyfit(h, curve, 1)[0],
            'breakeven': breakeven(h, curve)}


def block_draws(T, n_draws=N_DRAWS, block=BLOCK, seed=SEED):
    """Row indices of T target months for each draw of a moving-block bootstrap: blocks of `block` consecutive months,
    starting anywhere a whole block fits, drawn with replacement and joined until T months are filled."""
    rng = np.random.default_rng(seed)
    n_blocks = -(-T // block)
    starts = rng.integers(0, T - block + 1, size=(n_draws, n_blocks))
    return (starts[:, :, None] + np.arange(block)).reshape(n_draws, -1)[:, :T]


def curve_analysis(D, h):
    """The secondary analysis for a matrix D of d_t(h): one row per common target month in order, one column per
    horizon in `h`. Returns the curve, the three summaries with their intervals, and the summaries of every draw."""
    D = np.asarray(D, float)
    if np.isnan(D).any():
        raise ValueError('every target month needs d_t(h) at every horizon')
    curve = D.mean(axis=0)
    point = summaries(h, curve)
    idx = block_draws(len(D))
    drawn = [summaries(h, D[i].mean(axis=0)) for i in idx]
    out = {}
    for k, v in point.items():
        x = np.array([s[k] for s in drawn])
        lo, hi = np.quantile(x, [LEVEL / 2, 1 - LEVEL / 2], method='inverted_cdf')
        out[k] = {'estimate': v, 'lo': lo, 'hi': hi}
    return curve, out, drawn
