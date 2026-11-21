import numpy as np
import pandas as pd
import pytest

from sdt.data.clean import CLASSES
from sdt.twin.calibration import apply_calibration, class_probs, fit_calibration, to_classes
from sdt.twin.prompts import RAW


def test_to_classes_maps_options_in_form_order():
    out = to_classes(np.array([[0.5, 0.3, 0.2]]), 'BUS12')
    assert out.columns.tolist() == CLASSES and out.iloc[0].tolist() == [0.5, 0.3, 0.2]


def test_to_classes_gives_dur_the_given_middle_share():
    out = to_classes(np.array([[0.6, 0.4]]), 'DUR', mid_share=0.01)
    assert np.allclose(out.iloc[0].tolist(), [0.594, 0.01, 0.396]) and np.isclose(out.sum(axis=1)[0], 1)
    with pytest.raises(ValueError):
        to_classes(np.array([[0.6, 0.4]]), 'DUR')


def test_class_probs_divide_by_the_mass():
    raw = pd.DataFrame([[0.4, 0.2, 0.2], [0.5, 0.3, np.nan]], columns=RAW, index=[3, 5])
    out = class_probs(raw.iloc[[0]], 'BUS12')
    assert np.allclose(out.iloc[0].tolist(), [0.5, 0.25, 0.25, 0.8])
    dur = class_probs(raw.iloc[[1]], 'DUR', mid_share=0.01)
    assert dur.index.tolist() == [5] and np.allclose(dur.iloc[0].tolist(), [0.61875, 0.01, 0.37125, 0.8])


def synthetic(a_true, b_true, n=20_000, seed=0):
    """Class probabilities p, answers drawn from softmax(a_true * log p + b_true), and survey-like weights."""
    rng = np.random.default_rng(seed)
    p = pd.DataFrame(rng.dirichlet([2, 2, 2], n), columns=CLASSES)
    q = apply_calibration(p, pd.Series([a_true, *b_true], index=['a'] + CLASSES))
    y = np.array(CLASSES)[(q.cumsum(axis=1).to_numpy() < rng.random(n)[:, None]).sum(axis=1)]
    return p, y, rng.lognormal(0, 0.5, n)


@pytest.mark.parametrize('a_true', [0.4, 1.0, 1.8])
def test_fit_calibration_recovers_a_known_scale_and_shift(a_true):
    p, y, w = synthetic(a_true, [0.7, 0.0, -0.4])
    c = fit_calibration(p, y, w)
    assert c['mid'] == 0 and np.allclose(c[['a', 'fav', 'unf']], [a_true, 0.7, -0.4], atol=0.08)


def test_shift_only_keeps_the_scale_at_one():
    p, y, w = synthetic(1.0, [0.7, 0.0, -0.4])
    c = fit_calibration(p, y, w, scale=False)
    assert c['a'] == 1 and np.allclose(c[['fav', 'unf']], [0.7, -0.4], atol=0.06)


@pytest.mark.parametrize('scale', [False, True])
def test_calibration_matches_weighted_answer_shares(scale):
    # At the maximum, the weighted mean calibrated probability of each class equals its weighted share of answers
    p, y, w = synthetic(0.5, [0.0, 0.0, 0.0], seed=1)
    q = apply_calibration(p, fit_calibration(p, y, w, scale))
    actual = [np.average(y == c, weights=w) for c in CLASSES]
    assert np.allclose(np.average(q, axis=0, weights=w), actual, atol=1e-6)


def test_scale_never_reverses_the_twin():
    # Answers that run against p would want a < 0; the bound stops at a = 0, the baseline shares
    p, y, w = synthetic(-1.0, [0.0, 0.0, 0.0], seed=2)
    assert fit_calibration(p, y, w)['a'] == 0


def test_identity_calibration_leaves_probabilities_unchanged():
    p = pd.DataFrame([[0.2, 0.5, 0.3], [0.594, 0.01, 0.396]], columns=CLASSES, index=[7, 9])
    q = apply_calibration(p, pd.Series([1.0, 0, 0, 0], index=['a'] + CLASSES))
    assert q.index.tolist() == [7, 9] and np.allclose(q, p)
