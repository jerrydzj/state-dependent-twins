import json

import numpy as np
import pandas as pd
import pytest

from sdt.data.clean import CLASSES, OUT_PATH, OUTCOMES
from sdt.experiment import sha256
from sdt.mvp import twin_calibration
from sdt.mvp.twin_calibration import calibration_cv, postprocess
from sdt.twin.prompts import RAW

BLOCKS = [(202404, 202407), (202408, 202602)]


def interview_blocks(df, blocks, limit=None):
    """The web interviews of each block of months, as sdt.twin.run selects them (that module needs PyTorch)."""
    out = {f'{a}-{b}': df[(df.web == 1) & df.YYYYMM.between(a, b)] for a, b in blocks}
    return {k: d.head(limit) if limit else d for k, d in out.items()}


@pytest.fixture(scope='module')
def clean_table():
    if not OUT_PATH.exists():
        pytest.skip('data/processed/clean.parquet not built')
    return pd.read_parquet(OUT_PATH)


def training_web(clean_table, per_month=None):
    d = interview_blocks(clean_table, BLOCKS[:1])['202404-202407']
    return d.groupby('YYYYMM').head(per_month) if per_month else d


def synthetic_raw(sets, seed=0):
    rng = np.random.default_rng(seed)
    return pd.concat([pd.DataFrame(rng.dirichlet([1, 1, 1], len(d)) * 0.99, columns=RAW).assign(
        stage='base', scenario='actual', interviews=s, feature_set='demographics', item=item, CASEID=d.CASEID.to_numpy())
        for s, d in sets.items() for item in OUTCOMES], ignore_index=True)


def test_postprocess_keeps_answered_interviews_and_fits_calibrations(clean_table):
    sets = interview_blocks(clean_table, BLOCKS, limit=300)
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
    sample = training_web(clean_table, 150)                                  # four months
    cv = calibration_cv(synthetic_raw({'202404-202407': sample}), clean_table)
    assert len(cv) == len(OUTCOMES) * 4 and set(cv.model) == {'baseline', 'twin', 'shift', 'scale'}
    assert cv.log_loss.notna().all()
    # random probabilities carry no information, so calibration should land near the baseline shares, far below
    # the uncalibrated twin
    by = cv.pivot_table(index='item', columns='model', values='log_loss')
    assert (by.scale < by.twin).all() and (by.scale - by.baseline).abs().max() < 0.05


def test_rescore_records_and_checks_the_source_file(tmp_path, clean_table):
    d = training_web(clean_table, 150)
    source = tmp_path / 'source-run'
    source.mkdir()
    synthetic_raw({'202404-202407': d}).to_parquet(source / 'option_probs.parquet', index=False)
    meta = {'name': 'source-run', 'commit': 'abc', 'model': 'm', 'revision': 'r', 'device': 'gpu',
            'interviews': {'202404-202407': len(d)}, 'stages': ['base']}
    json.dump(meta | {'option_probs_sha256': sha256(source / 'option_probs.parquet')}, open(source / 'run.json', 'w'))
    twin_calibration.rescore('check', 'source-run', out_dir=tmp_path)
    (new,) = [p for p in tmp_path.iterdir() if p.name.endswith('-check')]
    assert {p.name for p in new.iterdir()} == {'predictions.parquet', 'calibration.json', 'calibration_cv.csv', 'run.json'}
    recorded = json.load(open(new / 'run.json'))['source']
    assert recorded['name'] == 'source-run' and recorded['option_probs_sha256'] == sha256(source / 'option_probs.parquet')
    json.dump(meta | {'option_probs_sha256': '0' * 64}, open(source / 'run.json', 'w'))
    with pytest.raises(RuntimeError, match='differs'):
        twin_calibration.rescore('check-again', 'source-run', out_dir=tmp_path)


def test_postprocess_calibrates_on_the_training_window_only(clean_table):
    # Whatever the twin says about evaluation interviews, the calibration stays the same
    sets = interview_blocks(clean_table, BLOCKS, limit=300)
    raw = synthetic_raw(sets)
    later = raw.interviews == '202408-202602'
    changed = raw.assign(p1=np.where(later, 0.9, raw.p1), p2=np.where(later, 0.05, raw.p2),
                         p3=np.where(later, 0.04, raw.p3))
    assert postprocess(raw, clean_table)[1] == postprocess(changed, clean_table)[1]


def test_postprocess_ignores_how_a_run_split_its_blocks(clean_table):
    # The training window is a range of months, so one block spanning both windows calibrates as two blocks do
    raw = synthetic_raw(interview_blocks(clean_table, BLOCKS, limit=300))
    assert postprocess(raw.assign(interviews='202404-202602'), clean_table)[1] == postprocess(raw, clean_table)[1]


def test_postprocess_needs_the_training_window(clean_table):
    raw = synthetic_raw(interview_blocks(clean_table, BLOCKS[1:], limit=300))
    with pytest.raises(ValueError, match='no interviews of the training window'):
        postprocess(raw, clean_table)
