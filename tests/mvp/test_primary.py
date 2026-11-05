import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from sdt.mvp.clean import CLASSES, OUT_PATH
from sdt.mvp.primary import dissimilarity, outcome, primary_test, spearman, state_dissimilarity
from sdt.mvp.score import score_predictions

MONTHS = [202410, 202411, 202412, 202501, 202502]                         # crosses a year


def test_dissimilarity_is_the_standardised_distance_to_the_nearest_training_row():
    train = pd.DataFrame({'x': [0.0, 2.0, 4.0], 'p': [0, 0, 1]})          # x: mean 2, SD 1.633; p: mean 1/3, SD 0.471
    evaluation = pd.DataFrame({'x': [2.0, 4.0], 'p': [1, 0]}, index=[202408, 202409])
    z = lambda x, p: np.array([(x - 2) / np.sqrt(8 / 3), (p - 1 / 3) / np.sqrt(2 / 9)])
    nearest = lambda x, p: min(np.linalg.norm(z(x, p) - z(*r)) for r in train.itertuples(index=False))
    assert dissimilarity(train, evaluation).tolist() == pytest.approx([nearest(2, 1), nearest(4, 0)])


def test_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(0)
    d = rng.permutation(19).astype(float)
    A = rng.integers(0, 6, size=(50, 19)).astype(float)                    # many ties
    assert spearman(d, A) == pytest.approx([spearmanr(d, a).statistic for a in A])
    assert spearman(d, A[0]) == pytest.approx(spearmanr(d, A[0]).statistic)


@pytest.mark.parametrize('lo, hi, expected', [
    ([0.1, 0.2], [0.5, 0.6], 'support'),
    ([0.1, -0.6], [0.5, -0.1], 'contradiction'),                         # below 0 against either comparator
    ([0.1, -0.2], [0.5, 0.3], 'inconclusive'),
    ([0.0, 0.2], [0.5, 0.6], 'inconclusive'),                            # an interval touching 0 does not lie above it
])
def test_outcome_rule(lo, hi, expected):
    assert outcome(pd.DataFrame({'lo': lo, 'hi': hi})) == expected


def drifting(twin_tracks, n=400, seed=0):
    """Five months whose favourable share rises by 0.1 a month. The model that tracks the drift predicts each month's
    true shares; the other two predict the first month's for every month."""
    rng = np.random.default_rng(seed)
    month = np.repeat(MONTHS, n)
    truth = np.stack([0.2 + 0.1 * np.arange(5), np.full(5, 0.3), 0.5 - 0.1 * np.arange(5)], axis=1)
    p = truth[np.repeat(np.arange(5), n)]
    df = pd.DataFrame({'CASEID': np.arange(5 * n), 'YYYYMM': month, 'WT': rng.uniform(0.5, 1.5, 5 * n)})
    for item in ['PAGO', 'PEXP']:
        df[item] = [rng.choice(CLASSES, p=q) for q in p]
    tracks, fixed = pd.DataFrame(p, columns=CLASSES), pd.DataFrame(np.tile(truth[0], (5 * n, 1)), columns=CLASSES)
    models = {'twin': tracks if twin_tracks else fixed, 'mnl': fixed if twin_tracks else tracks,
              'baseline': fixed if twin_tracks else tracks}
    pred = pd.concat([P.assign(item=item, CASEID=df.CASEID, model=m)
                      for item in ['PAGO', 'PEXP'] for m, P in models.items()], ignore_index=True)
    return pred, df, pd.Series(np.linspace(0.1, 0.5, 5), index=MONTHS)


@pytest.mark.parametrize('twin_tracks, expected', [(True, 'support'), (False, 'contradiction')])
def test_primary_test_on_a_drift_only_one_model_tracks(twin_tracks, expected):
    pred, df, d = drifting(twin_tracks)
    out = primary_test(pred, df, d, twin='twin', comparators=['mnl', 'baseline'], n_boot=200)
    assert out['outcome'] == expected
    assert out['rho'].observed.tolist() == pytest.approx([1.0 if twin_tracks else -1.0] * 2)
    assert out['rho_draws'].shape == (200, 2) and out['leave_one_out'].shape == (5, 2)
    assert out['elapsed'].tolist() == [0, 1, 2, 3, 4]
    # d rises with time here, so the time sibling equals rho on the sample, on every draw and leaving out each month
    pd.testing.assert_frame_equal(out['rho_time'], out['rho'])
    pd.testing.assert_frame_equal(out['rho_time_draws'], out['rho_draws'])
    pd.testing.assert_frame_equal(out['leave_one_out_time'], out['leave_one_out'])


def test_time_sibling_ignores_d_and_the_outcome_ignores_time():
    pred, df, d = drifting(True)
    shuffled = pd.Series([0.5, 0.3, 0.4, 0.1, 0.2], index=MONTHS)            # d now falls as A_t rises
    a = primary_test(pred, df, d, twin='twin', comparators=['mnl', 'baseline'], n_boot=50)
    b = primary_test(pred, df, shuffled, twin='twin', comparators=['mnl', 'baseline'], n_boot=50)
    pd.testing.assert_frame_equal(a['rho_time'], b['rho_time'])
    assert b['rho'].observed.tolist() == pytest.approx([spearman(shuffled, b['A'][c]) for c in ['mnl', 'baseline']])
    assert a['outcome'] == 'support' and b['outcome'] == 'contradiction'


def test_monthly_distances_average_to_the_distributional_score():
    pred, df, d = drifting(True)
    out = primary_test(pred, df, d, twin='twin', comparators=['mnl', 'baseline'], n_boot=20)
    W = out['distances'].groupby(['item', 'model']).wasserstein.mean()
    for m, g in pred.groupby('model'):
        assert W.xs(m, level='model').to_dict() == pytest.approx(score_predictions(g, df).wasserstein.to_dict())
    by_month = out['distances'].pivot(index='month', columns=['model', 'item'], values='wasserstein')
    assert out['A']['mnl'].to_numpy() == pytest.approx((by_month['mnl'] - by_month['twin']).mean(axis=1).to_numpy())


def test_models_must_predict_the_same_interviews():
    pred, df, d = drifting(True)
    pred = pred.drop(pred.index[(pred.model == 'mnl') & (pred.CASEID == 0)])
    with pytest.raises(ValueError, match='different interviews'):
        primary_test(pred, df, d, twin='twin', comparators=['mnl', 'baseline'], n_boot=20)


@pytest.mark.skipif(not OUT_PATH.exists(), reason='data/processed/mvp_clean.parquet not built')
def test_dissimilarity_of_the_evaluation_window_peaks_at_the_flip():
    # The plan states that d_t peaks in December 2024 and January 2025
    d = state_dissimilarity(pd.read_parquet(OUT_PATH, columns=['YYYYMM']))
    assert len(d) == 19 and d.index[0] == 202408 and d.index[-1] == 202602
    assert set(d.nlargest(2).index) == {202412, 202501}
