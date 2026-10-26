import numpy as np
import pandas as pd
import pytest

from sdt.mvp.clean import clean, load_raw
from sdt.mvp.encode import mnl_features
from sdt.mvp.models import BaselineShares, mnl
from sdt.mvp.score import (GROUPS, PERIODS, bootstrap, cells, distribution_score, evaluate, log_loss, noise_floor, predict,
                           score_predictions, skill, wasserstein)
from sdt.mvp.windows import EVAL_START, HORIZON, TRAIN_END, split
from sdt.mvp.clean import OUT_PATH
from sdt.paths import DATA, RAW_CSV, ROOT


def test_log_loss_by_hand():
    P = [[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]]
    expected = (-np.log(0.5) * 1 - np.log(0.8) * 3) / 4
    assert log_loss(['fav', 'unf'], P, [1, 3]) == pytest.approx(expected)


def test_log_loss_rejects_missing_outcome():
    with pytest.raises(ValueError):
        log_loss(['fav', np.nan], [[1 / 3] * 3] * 2, [1, 1])


def test_skill():
    assert skill(0.9, 0.9) == 0 and skill(0.45, 0.9) == pytest.approx(0.5)


def test_baseline_is_weighted_web_shares():
    X = pd.DataFrame({'web': [1, 1, 1, 0]})
    m = BaselineShares().fit(X, ['fav', 'unf', 'unf', 'mid'], sample_weight=[2, 1, 1, 100])
    assert m.shares_ == pytest.approx([0.5, 0, 0.5])                 # the phone row is ignored
    assert (predict(m, X) == m.shares_).all()
    assert BaselineShares(web_only=False).fit(X, ['fav', 'unf', 'unf', 'mid']).shares_ == pytest.approx([.25, .25, .5])


def test_windows_match_interview_dates():
    win = pd.read_csv(DATA / 'external' / 'sca' / 'interview_windows.csv', parse_dates=['interview_end'])
    assert win.loc[win.interview_end < HORIZON, 'YYYYMM'].max() == TRAIN_END
    assert win.YYYYMM[win.YYYYMM > TRAIN_END].min() == EVAL_START
    train, test = split(pd.DataFrame({'YYYYMM': [201702, 202407, 202408, 202602]}))
    assert train.YYYYMM.tolist() == [201702, 202407] and test.YYYYMM.tolist() == [202408, 202602]


@pytest.mark.skipif(not RAW_CSV.exists(), reason='data/data.csv not downloaded')
def test_real_data_baseline():
    df = clean(load_raw())
    scores = evaluate(BaselineShares(), mnl_features, df)
    assert scores.index.tolist() == ['PAGO', 'PEXP', 'BUS12', 'BUS5', 'DUR']
    assert scores.columns.tolist() == ['log_loss', 'wasserstein']
    loss = scores.log_loss
    # No constant can beat the evaluation window's own shares, so the baseline's loss must be at least theirs
    _, test = split(df)
    oracle = evaluate(BaselineShares(), mnl_features, pd.concat([test.assign(YYYYMM=TRAIN_END, CASEID=test.CASEID + 10**7), test])).log_loss
    assert (oracle <= loss).all() and (oracle > 0).all()


def test_mnl_is_unpenalised_weighted_mle():
    # With one binary feature the MLE reproduces the weighted class shares in each group exactly; a penalty would not
    X = pd.DataFrame({'web': [0, 0, 0, 1, 1, 1, 1]})
    y = ['fav', 'mid', 'unf', 'fav', 'fav', 'unf', 'mid']
    w = [1, 1, 2, 1, 3, 2, 2]
    P = predict(mnl().fit(X, y, sample_weight=w), X)
    assert P[0] == pytest.approx([0.25, 0.25, 0.5], abs=1e-4) and P[3] == pytest.approx([0.5, 0.25, 0.25], abs=1e-4)


def test_wasserstein_by_hand():
    # Predicted 20/30/50 vs actual 16/35/49: half the fav gap (0.04) plus half the unf gap (0.01)
    assert wasserstein([0.2, 0.3, 0.5], [0.16, 0.35, 0.49]) == pytest.approx(0.025)
    assert wasserstein([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0
    assert wasserstein([1, 0, 0], [0, 0, 1]) == 1                                  # opposite ends of the scale
    assert wasserstein([1 / 3] * 3, [1, 0, 0]) == pytest.approx(0.5)               # uniform guess vs all fav


def test_distribution_score_averages_months():
    y = ['fav', 'unf', 'fav', 'fav']
    month = [1, 1, 2, 2]
    P = [[0.5, 0, 0.5]] * 4                       # month 1 actual 50/0/50: distance 0; month 2 actual 100/0/0: 0.5
    assert distribution_score(y, P, [1, 1, 1, 1], month) == pytest.approx(0.25)
    assert distribution_score(y, P, [1, 3, 1, 1], month) == pytest.approx(0.125 + 0.25)   # month 1 actual 25/0/75


def test_noise_floor_shrinks_with_sample_size():
    rng = np.random.default_rng(1)
    floors = []
    for n in (100, 1600):
        y = rng.choice(['fav', 'mid', 'unf'], size=3 * n)
        floors.append(noise_floor(y, np.ones(3 * n), np.repeat([1, 2, 3], n), n_boot=100))
    assert 0 < floors[1] < floors[0] / 2                                           # about 1/sqrt(n): 4x smaller


def test_bootstrap_is_paired():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({'CASEID': np.arange(n), 'YYYYMM': np.repeat([202409, 202410, 202411], n // 3),
                       'WT': rng.uniform(0.5, 1.5, n), 'PAGO': rng.choice(['fav', 'mid', 'unf'], n, p=[.5, .3, .2])})
    base = pd.DataFrame({'item': 'PAGO', 'CASEID': df.CASEID, 'fav': 0.4, 'mid': 0.3, 'unf': 0.3})
    good = base.assign(fav=0.5, mid=0.3, unf=0.2)
    pred = pd.concat([base.assign(model='baseline'), good.assign(model='mnl'), base.assign(model='gb')])
    ci = bootstrap(pred, df, n_boot=200).set_index('model')
    assert ci.loc['gb', 'log_loss_skill_lo'] == ci.loc['gb', 'log_loss_skill_hi'] == 0     # same as the baseline
    assert ci.loc['mnl', 'log_loss_skill_lo'] > 0                                           # the true shares win
    point = score_predictions(pred[pred.model == 'mnl'], df).loc['PAGO']
    for s in ['log_loss', 'wasserstein']:
        assert ci.loc['mnl', f'{s}_lo'] <= point[s] <= ci.loc['mnl', f'{s}_hi']
    assert ci.loc['mnl-gb', 'log_loss_hi'] < 0                                              # mnl's loss is lower


def test_cells_partition_the_evaluation_window():
    months = [202407, 202408, 202410, 202411, 202501, 202502, 202602]
    df = pd.DataFrame({'CASEID': range(14), 'YYYYMM': months * 2,
                       'POLAFF': pd.Categorical(['Rep', 'Dem', 'Ind', None, 'Rep', 'Dem', 'Ind'] * 2)})
    out = {(p, g): d for p, g, d in cells(df)}
    assert list(out) == [(p, g) for p in ['all', *PERIODS] for g in ['all', *GROUPS]]
    assert set(out['all', 'all'].YYYYMM) == set(months) - {202407}                    # evaluation window only
    assert {p: sorted(set(out[p, 'all'].YYYYMM)) for p in PERIODS} == {
        'Baseline': [202408, 202410], 'Flip': [202411, 202501], 'Tariffs': [202502, 202602]}
    assert sum(len(out[p, 'all']) for p in PERIODS) == len(out['all', 'all'])          # periods don't overlap
    assert sum(len(out['all', g]) for g in GROUPS) == out['all', 'all'].POLAFF.notna().sum()   # non-reporters: 'all' only


FIRST_RUN = ROOT / 'experiments' / 'mvp' / '2026-10-22-supervised-demographics'


@pytest.mark.skipif(not (FIRST_RUN / 'predictions.parquet').exists() or not OUT_PATH.exists(),
                    reason='predictions or clean table not built')
def test_all_cell_reproduces_first_run_scores():
    # The 'all'/'all' cell must equal the scores reported in notebook 03, which predate the cells
    from sdt.mvp.run import score_cell
    run = FIRST_RUN
    df = pd.read_parquet(OUT_PATH)
    _, _, d = next(cells(df))
    pred = pd.read_parquet(run / 'predictions.parquet')
    new = score_cell(pred[pred.CASEID.isin(d.CASEID)], d, pairs=[('mnl', 'gb')])
    old = pd.read_csv(run / 'scores.csv')
    if 'period' in old:
        old = old[(old.period == 'all') & (old.group == 'all')].drop(columns=['period', 'group'])
    key = ['item', 'model']
    pd.testing.assert_frame_equal(new.round(6).set_index(key).sort_index(), old.set_index(key).sort_index()[new.columns.drop(key)],
                                  check_exact=False, atol=1e-6, rtol=0)


def test_score_cell_differences_follow_pairs():
    from sdt.mvp.run import score_cell
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({'CASEID': range(n), 'YYYYMM': rng.choice([202408, 202409], n), 'WT': rng.uniform(0.5, 1.5, n),
                       'PAGO': rng.choice(['fav', 'mid', 'unf'], n)})
    P = lambda: rng.dirichlet([1, 1, 1], n)
    pred = pd.concat([pd.DataFrame(P(), columns=['fav', 'mid', 'unf']).assign(item='PAGO', CASEID=range(n), model=m)
                      for m in ['baseline', 'mnl_demographics', 'mnl_demographics_state']], ignore_index=True)
    out = score_cell(pred, df, pairs=[('mnl_demographics_state', 'mnl_demographics'), ('gb_x', 'gb_y')]).set_index('model')
    assert 'mnl_demographics_state-mnl_demographics' in out.index and 'gb_x-gb_y' not in out.index   # absent pairs skipped
    d = out.loc['mnl_demographics_state-mnl_demographics']
    assert d.log_loss == pytest.approx(out.loc['mnl_demographics_state', 'log_loss'] - out.loc['mnl_demographics', 'log_loss'])
    assert d.log_loss_lo <= d.log_loss <= d.log_loss_hi
