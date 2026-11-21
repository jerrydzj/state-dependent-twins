import numpy as np
import pytest

from sdt.realtime.stats import (bandwidth, block_draws, breakeven, curve_analysis, daniell_variance, diebold_mariano,
                                outcome)


def test_bandwidth_is_the_integer_cube_root():
    assert [bandwidth(T) for T in [1, 7, 8, 19, 26, 27, 63, 64]] == [1, 1, 2, 2, 2, 3, 3, 4]


def test_daniell_variance_averages_the_lowest_periodogram_ordinates():
    d = np.random.default_rng(0).normal(size=26)
    I = np.abs(np.fft.fft(d - d.mean())) ** 2 / (2 * np.pi * len(d))      # same modulus as the plan's sum over t
    variance, m = daniell_variance(d)
    assert m == 2 and variance == pytest.approx(2 * np.pi * I[1:3].mean())


def test_diebold_mariano_uses_t_with_2m_degrees_of_freedom():
    d = np.random.default_rng(1).normal(0.3, 1, size=26)
    r = diebold_mariano(d)
    assert r['df'] == 4 and r['critical'] == pytest.approx(2.776, abs=1e-3)
    assert r['lo'] < r['mean'] < r['hi'] and (r['p'] < 0.05) == (abs(r['statistic']) > r['critical'])
    assert r['hi'] - r['mean'] == pytest.approx(r['critical'] * r['se'])
    assert diebold_mariano(d[:7])['df'] == 2


def test_outcome():
    assert outcome({'p': 0.01, 'mean': 0.2}) == 'twins more accurate'
    assert outcome({'p': 0.01, 'mean': -0.2}) == 'simple models more accurate'
    assert outcome({'p': 0.2, 'mean': 0.2}) == 'neither shown more accurate'


def test_breakeven():
    h = np.arange(1, 13)
    assert breakeven(h, h - 2.5) == pytest.approx(2.5)
    assert breakeven(h, h - 0.5) == -np.inf and breakeven(h, -h) == np.inf
    assert breakeven([1, 2, 3], [-1, 0, 1]) == 2                     # turns positive after touching 0 at h = 2


def test_block_draws_join_blocks_of_consecutive_months():
    idx = block_draws(16)
    assert idx.shape == (1000, 16) and idx.min() >= 0 and idx.max() <= 15
    blocks = idx[:, :15].reshape(1000, 5, 3)
    assert (np.diff(blocks, axis=2) == 1).all() and (block_draws(16) == idx).all()


def test_curve_analysis_on_a_rising_curve():
    h = np.arange(1, 13)
    D = np.tile(0.01 * (h - 4.5), (16, 1)) + np.random.default_rng(2).normal(0, 0.001, size=(16, 12))
    curve, summary, drawn = curve_analysis(D, h)
    assert summary['rho']['estimate'] == pytest.approx(1) and summary['slope']['estimate'] == pytest.approx(0.01, abs=1e-3)
    assert summary['breakeven']['estimate'] == pytest.approx(4.5, abs=0.1) and len(drawn) == 1000
    assert all(r['lo'] <= r['estimate'] <= r['hi'] for r in summary.values())
