import numpy as np
import pandas as pd
import pytest

from sdt.data.clean import CLASSES, OUT_PATH
from sdt.features import state_table
from sdt.realtime.months import between
from sdt.realtime.two_stage import (NotConverged, StageOne, fit_ridge, predict_ridge, probabilities, respondent_terms,
                                    stage_two)

C_GRID = [0.01, 1.0, np.inf]


@pytest.fixture(scope='module')
def sample():
    """Real interviews to July 2024 (30 per month), with answers drawn from a known two-stage model."""
    if not OUT_PATH.exists():
        pytest.skip('data/processed/clean.parquet not built')
    df = pd.read_parquet(OUT_PATH)
    d = df[df.YYYYMM <= 202407].groupby(['YYYYMM', 'web']).head(30).reset_index(drop=True)
    rng = np.random.default_rng(0)
    months = sorted(d.YYYYMM.unique())
    level = pd.DataFrame(rng.normal(0, 0.5, size=(len(months), 2)), index=months)
    R = respondent_terms(d, 'D').to_numpy(float)
    beta = np.zeros((R.shape[1], 2))
    beta[:3] = [[0.8, -0.5], [-0.4, 0.6], [0.3, 0.3]]
    eta = level.loc[d.YYYYMM].to_numpy() + 0.7 * d.web.to_numpy()[:, None] + R @ beta
    P = probabilities(eta)
    d['PAGO'] = np.array(CLASSES)[(P.cumsum(axis=1) < rng.random(len(d))[:, None]).sum(axis=1)]
    return d


def test_unpenalized_fit_matches_every_month_s_shares(sample):
    # With free month intercepts, the score equations make each month's mean predicted probabilities its shares
    one = StageOne(sample, 'PAGO', 'D', [np.inf])
    eta = one.levels.loc[sample.YYYYMM].to_numpy() + one.log_odds(sample)
    P = pd.DataFrame(probabilities(eta), columns=CLASSES).assign(m=sample.YYYYMM.to_numpy(), w=sample.WT.to_numpy())
    Y = pd.get_dummies(sample.PAGO)[CLASSES].astype(float).assign(m=sample.YYYYMM.to_numpy(), w=sample.WT.to_numpy())
    share = lambda t: t[CLASSES].mul(t.w, axis=0).groupby(t.m).sum().div(t.w.groupby(t.m).sum(), axis=0)
    assert np.allclose(share(P), share(Y), atol=1e-4)
    assert one.dropped == []


def test_penalty_is_tuned_and_shrinks_the_respondent_terms(sample):
    one = StageOne(sample, 'PAGO', 'D', C_GRID)
    assert one.C in C_GRID and len(one.cv) == 3 and np.isfinite(one.cv).all()
    strong, none = StageOne(sample, 'PAGO', 'D', [0.01]), StageOne(sample, 'PAGO', 'D', [np.inf])
    assert np.abs(strong.beta).sum() < np.abs(none.beta).sum()


def test_a_penalty_that_does_not_converge_is_never_chosen(sample, monkeypatch):
    fit = StageOne.fit

    def no_maximum_without_penalty(R, month, web, Y, w, rows, n_months, C, start):
        if np.isinf(C):
            raise NotConverged('no maximum')
        return fit(R, month, web, Y, w, rows, n_months, C, start)
    monkeypatch.setattr(StageOne, 'fit', staticmethod(no_maximum_without_penalty))
    one = StageOne(sample, 'PAGO', 'D', [1.0, np.inf])
    assert one.C == 1.0 and np.isinf(one.cv[1]) and np.isfinite(one.cv[0])
    with pytest.raises(NotConverged):
        StageOne(sample, 'PAGO', 'D', [np.inf])


def test_folds_put_each_month_and_answer_in_every_fold_s_fitting_rows(sample):
    month = sample.YYYYMM.to_numpy()
    Y = pd.get_dummies(sample.PAGO)[CLASSES].to_numpy(float)
    fold = StageOne.folds(month, Y, 5, 0)
    cells = pd.DataFrame({'m': month, 'y': Y.argmax(axis=1), 'f': fold}).groupby(['m', 'y']).f
    assert (cells.nunique() >= np.minimum(cells.size(), 2)).all()      # a cell of two or more spans two folds
    assert (np.bincount(fold) >= len(fold) // 5).all()


def test_a_month_without_some_answer_has_no_level(sample):
    d = sample.copy()
    d.loc[(d.YYYYMM == 202001) & (d.PAGO == 'mid'), 'PAGO'] = 'fav'
    one = StageOne(d, 'PAGO', 'D', [1.0])
    assert one.dropped == [202001] and 202001 not in one.levels.index and len(one.levels) == d.YYYYMM.nunique() - 1


def test_respondent_terms_of_the_p_spec_add_the_party_products(sample):
    D, DP = respondent_terms(sample, 'D'), respondent_terms(sample, 'D+P')
    assert 'web' not in D and 'pres_R' not in DP and set(DP.columns) - set(D.columns) == {
        'POLAFF=Rep:pres_R', 'POLAFF=Dem:pres_R', 'POLAFF=Rep:winner_R', 'POLAFF=Dem:winner_R'}


def test_ridge_with_no_penalty_is_least_squares():
    X = np.random.default_rng(3).normal(size=(40, 2))
    y = 1 + X @ [2.0, -1.0]
    assert np.allclose(predict_ridge(fit_ridge(X, y, 0.0), X), y)
    assert np.abs(fit_ridge(X, y, 1e6)[3]).max() < 0.01


def stage_two_inputs(levels_of, last=202405):
    months = between(201702, last)
    S = state_table(tuple(between(201702, 202407)))
    levels = pd.DataFrame({c: levels_of(S.loc[months]) for c in ['fav', 'unf']}, index=pd.Index(months, name='month'))
    fold = pd.Series(np.arange(len(months)) % 5, index=months)
    return levels, fold, S.loc[[202406, 202407]]


STATE_EFFECT = lambda S: 0.2 * S.unrate - 0.1 * S.cpi


def test_stage_two_level_regression_predicts_at_the_target_state():
    levels, fold, target = stage_two_inputs(lambda S: 0.5 + STATE_EFFECT(S))
    level, chosen = stage_two(levels, 'demographics_macro', 'level-regression', [0.0], fold, target)
    assert np.allclose(level.fav, 0.5 + STATE_EFFECT(target)) and chosen == {'fav': 0.0, 'unf': 0.0}


def test_anchored_change_adds_the_state_effect_and_no_trend():
    # Levels with a trend of 0.01 a month: the regression's intercept takes the trend, and the forecast leaves it out
    levels, fold, target = stage_two_inputs(lambda S: 0.5 + 0.01 * np.arange(len(S)) + STATE_EFFECT(S))
    change, _ = stage_two(levels, 'demographics_macro', 'anchored-change', [0.0], fold, target)
    anchor = levels.loc[202405, 'unf']
    S = state_table(tuple(between(201702, 202407)))
    assert np.allclose(change.unf, anchor + STATE_EFFECT(target) - STATE_EFFECT(S.loc[[202405]]).iloc[0])


def test_anchored_change_starts_from_the_latest_month_with_a_level():
    levels, fold, target = stage_two_inputs(lambda S: 0.5 + STATE_EFFECT(S))
    without_may = levels.drop(index=202405)
    change, _ = stage_two(without_may, 'demographics_macro', 'anchored-change', [0.0], fold, target)
    S = state_table(tuple(between(201702, 202407)))
    expected = levels.loc[202404, 'fav'] + STATE_EFFECT(target) - STATE_EFFECT(S.loc[[202404]]).iloc[0]
    assert np.allclose(change.fav, expected)


def test_stage_two_smoothing_runs_over_web_months_only():
    levels, fold, target = stage_two_inputs(lambda S: np.arange(len(S), dtype=float))
    web = levels.loc[202404:].fav.to_numpy()                             # April and May 2024
    level, chosen = stage_two(levels, 'demographics', 0.5, [0.0], fold, target)
    assert chosen is None and np.allclose(level.fav, 0.5 * web[1] + 0.5 * web[0])
    with pytest.raises(ValueError, match='no web month'):
        stage_two(levels.loc[:202403], 'demographics', 0.5, [0.0], fold, target)
