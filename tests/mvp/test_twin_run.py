import numpy as np
import pandas as pd
import pytest

from sdt.mvp import twin_run
from sdt.mvp.clean import CLASSES, OUT_PATH, OUTCOMES
from sdt.mvp.encode import state_table
from sdt.mvp.prompts import FEATURE_SETS
from sdt.mvp.twin import RAW
from sdt.mvp.twin_run import SCENARIOS, Runner, calibration_cv, interview_sets, postprocess, rerun_sample

TRAIN_WEB_MONTHS = (202404, 202405, 202406, 202407)


def test_scenarios():
    S = state_table(TRAIN_WEB_MONTHS)
    flipped = SCENARIOS['pres_R'](S)
    assert (flipped[['pres_R', 'winner_R']] == 1).all(axis=None) and (S[['pres_R', 'winner_R']] == 0).all(axis=None)
    hot = SCENARIOS['cpi_9.1'](S)
    assert (hot.cpi == 9.1).all()
    assert np.allclose(hot.cpi - hot.cpi_d1y, S.cpi - S.cpi_d1y)          # the year-earlier value is unchanged
    assert hot.drop(columns=['cpi', 'cpi_d1y']).equals(S.drop(columns=['cpi', 'cpi_d1y']))


@pytest.fixture(scope='module')
def clean_table():
    if not OUT_PATH.exists():
        pytest.skip('data/processed/mvp_clean.parquet not built')
    return pd.read_parquet(OUT_PATH)


def test_train_web_is_the_last_four_training_months(clean_table):
    d = interview_sets(clean_table, ['train-web'])['train-web']
    assert len(d) == 2257 and tuple(sorted(d.YYYYMM.unique())) == TRAIN_WEB_MONTHS and (d.web == 1).all()


def test_rerun_sample_draws_ten_per_cell(clean_table):
    d = interview_sets(clean_table, ['train-web'])['train-web']
    cells = list(rerun_sample(d))
    assert len(cells) == len(FEATURE_SETS) * len(OUTCOMES) and all(len(s) == 10 for _, _, s in cells)
    assert len({tuple(s.CASEID) for _, _, s in cells}) == len(cells)               # each cell has its own draw
    assert [tuple(s.CASEID) for _, _, s in rerun_sample(d)] == [tuple(s.CASEID) for _, _, s in cells]


def test_runner_skips_chunks_on_disk(tmp_path, clean_table):
    runner = Runner(tmp_path, 'unused', 8)
    runner.model = lambda dtype: pytest.fail('the model must not be loaded for a chunk on disk')
    (tmp_path / 'chunks' / 'base--actual--train-web--demographics--PAGO.parquet').touch()
    runner.chunk('base', 'actual', 'train-web', 'demographics', 'PAGO', clean_table.head(2))


def synthetic_raw(sets, seed=0):
    rng = np.random.default_rng(seed)
    return pd.concat([pd.DataFrame(rng.dirichlet([1, 1, 1], len(d)) * 0.99, columns=RAW).assign(
        stage='base', scenario='actual', interviews=s, feature_set='demographics', item=item, CASEID=d.CASEID.to_numpy())
        for s, d in sets.items() for item in OUTCOMES], ignore_index=True)


def test_postprocess_keeps_answered_interviews_and_fits_calibrations(clean_table):
    sets = interview_sets(clean_table, ['train-web', 'eval'], limit=300)
    pred, calibration = postprocess(synthetic_raw(sets), clean_table)
    assert sorted(pred.model.unique()) == ['twin_demographics', 'twin_scale_demographics', 'twin_shift_demographics']
    assert pred.columns.tolist() == ['item', 'CASEID'] + CLASSES + ['model'] and np.allclose(pred[CLASSES].sum(axis=1), 1)
    key = clean_table.set_index('CASEID')
    for item in OUTCOMES:
        ids = pred[(pred.item == item) & (pred.model == 'twin_demographics')].CASEID
        answered = pd.concat(sets.values()).pipe(lambda d: d[d[item].notna()]).CASEID
        assert sorted(ids) == sorted(answered) and key.loc[ids, item].notna().all()
    assert set(calibration['shift']['demographics']) == set(OUTCOMES) and 0 < calibration['dur_mid_share'] < 0.05
    assert all(c['a'] == 1 for c in calibration['shift']['demographics'].values())


def test_calibration_cv_scores_every_model_on_held_out_months(clean_table):
    d = interview_sets(clean_table, ['train-web'])['train-web']
    sample = d.groupby('YYYYMM').head(150)                                   # four months
    cv = calibration_cv(synthetic_raw({'train-web': sample}), clean_table)
    assert len(cv) == len(OUTCOMES) * 4 and set(cv.model) == {'baseline', 'twin', 'shift', 'scale'}
    assert cv.log_loss.notna().all()
    # random probabilities carry no information, so calibration should land near the baseline shares, far below
    # the uncalibrated twin
    by = cv.pivot_table(index='item', columns='model', values='log_loss')
    assert (by.scale < by.twin).all() and (by.scale - by.baseline).abs().max() < 0.05
