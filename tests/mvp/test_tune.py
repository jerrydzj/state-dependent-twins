import numpy as np
import pandas as pd
import pytest

from sdt.mvp.clean import CLASSES
from sdt.mvp.encode import mnl_features
from sdt.mvp.models import Ridge, gb, mnl
from sdt.mvp.tune import N_FOLDS, folds, month_folds, tune, tune_ridge
from sdt.mvp.windows import TRAIN_END


def fake_train(months, n=60, seed=0):
    """Random clean-table rows: n interviews per month, web from Apr 2024."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in months:
        for _ in range(n):
            rows.append({'YYYYMM': m, 'WT': rng.uniform(0.5, 1.5), 'web': int(m >= 202404),
                         'PAGO': rng.choice(['fav', 'mid', 'unf']), 'YTL5': rng.integers(1, 6), 'HOMEOWN': 'own',
                         'INVEST': 1, 'AGE': rng.integers(18, 90), 'REGION': 'S', 'SEX': 'male', 'MARRY': 'married',
                         'NUMKID': 0, 'NUMADT': 2, 'EDUC': 3, 'POLAFF': rng.choice(['Rep', 'Dem', 'Ind'])})
    df = pd.DataFrame(rows)
    for c in ['HOMEOWN', 'REGION', 'SEX', 'MARRY', 'POLAFF', 'PAGO']:
        df[c] = df[c].astype('category')
    return df


def test_folds_block_months_and_spread_web():
    months = [y * 100 + m for y in range(2017, 2025) for m in range(1, 13) if 201702 <= y * 100 + m <= TRAIN_END]
    train = fake_train(months, n=2)
    out = list(folds(train))
    assert len(out) == N_FOLDS
    assert sorted(m for _, val in out for m in val.YYYYMM.unique()) == months       # each month validated exactly once
    for fit, val in out:
        assert not set(fit.YYYYMM) & set(val.YYYYMM) and fit.web.any()
    assert month_folds(train).loc[202404:].nunique() == 4                          # 4 web months in 4 different folds


def test_tune_returns_a_grid_setting():
    months = [y * 100 + m for y in range(2017, 2025) for m in range(1, 13) if 201702 <= y * 100 + m <= TRAIN_END]
    grid = [{'max_leaf_nodes': 4, 'min_samples_leaf': 20}, {'max_leaf_nodes': 8, 'min_samples_leaf': 20}]
    best, curves = tune(fake_train(months, n=8), 'PAGO', grid=grid, max_trees=20)
    assert {k: best[k] for k in grid[0]} in grid and 1 <= best['max_iter'] <= 20
    assert curves.shape == (2, 20) and curves.min(axis=None) == curves.loc[tuple(best[k] for k in grid[0]), best['max_iter']]


def test_gb_has_no_random_early_stopping():
    assert gb().early_stopping is False


def ridge_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({'a': rng.normal(5, 3, n), 'b': rng.integers(0, 2, n).astype(float)})
    logits = np.column_stack([np.zeros(n), 0.4 * X.a - 2, 1.5 * X.b - 1])
    P = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array(CLASSES)[[rng.choice(3, p=p) for p in P]]
    return X, y, rng.uniform(0.5, 1.5, n)


def test_ridge_without_penalty_is_the_mnl():
    X, y, w = ridge_data()
    P = Ridge(np.inf).fit(X, y, sample_weight=w).predict_proba(X)
    assert np.abs(P - mnl().fit(X, y, sample_weight=w).predict_proba(X)).max() < 1e-5


def test_ridge_standardises_on_training_rows():
    X, y, w = ridge_data()
    m = Ridge(1.0).fit(X, y, sample_weight=w)
    assert m.scaler_.mean_ == pytest.approx(np.average(X, axis=0, weights=w))
    shares = np.array([w[y == c].sum() for c in m.classes_]) / w.sum()
    P = Ridge(1e-8).fit(X, y, sample_weight=w).predict_proba(X)      # a huge penalty leaves only the intercepts
    assert np.abs(P - shares).max() < 1e-3


def test_tune_ridge_returns_a_grid_value():
    months = [y * 100 + m for y in range(2017, 2025) for m in range(1, 13) if 201702 <= y * 100 + m <= TRAIN_END]
    grid = [1e-3, 1.0, np.inf]
    C, loss = tune_ridge(fake_train(months, n=8), 'PAGO', mnl_features, grid=grid)
    assert C in grid and loss.index.tolist() == grid and loss.idxmin() == C
