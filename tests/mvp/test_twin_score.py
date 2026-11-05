import json
import shutil

import numpy as np
import pandas as pd
import pytest

from sdt.mvp import run as supervised_run
from sdt.mvp.clean import CLASSES, OUT_PATH, OUTCOMES
from sdt.mvp.primary import COMPARATORS, TWIN
from sdt.mvp.prompts import FEATURE_SETS
from sdt.mvp.twin import RAW
from sdt.mvp.twin_run import postprocess, sha256
from sdt.mvp.twin_score import run
from sdt.mvp.windows import split
from sdt.paths import RAW_CSV


@pytest.fixture(scope='module')
def sources(tmp_path_factory):
    """A small clean table (one phone interview per training month, so that d_t sees all 90 months, the training web
    interviews and 100 interviews of each evaluation month), and a twin run and a supervised run on it with random
    probabilities, laid out as the real runs are."""
    if not OUT_PATH.exists() or not RAW_CSV.exists():
        pytest.skip('data/data.csv or data/processed/mvp_clean.parquet missing')
    clean = pd.read_parquet(OUT_PATH)
    train, test = split(clean)
    sets = {'train-web': train[train.web == 1], 'eval': test.groupby('YYYYMM').head(100)}
    df = pd.concat([train[train.web == 0].groupby('YYYYMM').head(1), *sets.values()], ignore_index=True)
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp('runs')
    data_hash = sha256(RAW_CSV)

    twins = root / 'twins'
    twins.mkdir()
    raw = pd.concat([pd.DataFrame(rng.dirichlet([1, 1, 1], len(d)) * 0.99, columns=RAW).assign(
        stage='base', scenario='actual', interviews=s, feature_set=fs, item=item, CASEID=d.CASEID.to_numpy())
        for s, d in sets.items() for fs in FEATURE_SETS for item in OUTCOMES], ignore_index=True)
    raw.to_parquet(twins / 'option_probs.parquet', index=False)
    json.dump(postprocess(raw, df)[1], open(twins / 'calibration.json', 'w'))
    json.dump({'name': 'twins', 'commit': 'abc', 'data_csv_sha256': data_hash,
               'option_probs_sha256': sha256(twins / 'option_probs.parquet')}, open(twins / 'run.json', 'w'))

    supervised = root / 'supervised'
    supervised.mkdir()
    pred = pd.concat([pd.DataFrame(rng.dirichlet([2, 2, 2], len(a)), columns=CLASSES).assign(
        item=item, CASEID=a.CASEID.to_numpy(), model=m)
        for item in OUTCOMES for a in [sets['eval'][sets['eval'][item].notna()]] for m in COMPARATORS],
        ignore_index=True)
    pred[['item', 'CASEID'] + CLASSES + ['model']].to_parquet(supervised / 'predictions.parquet', index=False)
    supervised_run.score_run(pred, df).to_csv(supervised / 'scores.csv', index=False, float_format='%.6f')
    json.dump({'name': 'supervised', 'commit': 'def', 'data_csv_sha256': data_hash}, open(supervised / 'run.json', 'w'))
    return df, root


def test_run_writes_every_output(sources, tmp_path):
    df, root = sources
    run('check', 'twins', 'supervised', source_dir=root, out_dir=tmp_path, df=df)
    (out,) = tmp_path.iterdir()
    primary = json.load(open(out / 'primary.json'))
    assert primary['twin'] == TWIN and primary['outcome'] in {'support', 'contradiction', 'inconclusive'}
    assert all(r['lo'] <= r['hi'] for r in primary['rho'].values())
    months = pd.read_csv(out / 'primary_months.csv')
    assert len(months) == 19 and months.d.notna().all()
    scores = pd.read_csv(out / 'scores.csv')
    assert {TWIN, f'{TWIN}-mnl_demographics_macro_politics'} <= set(scores.model)
    assert len(pd.read_csv(out / 'rho_draws.csv')) == 1000
    by_period = pd.read_csv(out / 'calibration_by_period.csv')
    assert set(by_period.fitted_on) == {'training web', 'Baseline', 'Flip', 'Tariffs'}
    recorded = json.load(open(root / 'twins' / 'calibration.json'))
    row = by_period[(by_period.fitted_on == 'training web') & (by_period.calibration == 'scale')].iloc[0]
    assert row.a == pytest.approx(recorded['scale'][row.feature_set][row['item']]['a'], abs=1e-6)
    meta = json.load(open(out / 'run.json'))
    assert meta['sources']['twins']['option_probs_sha256'] == sha256(root / 'twins' / 'option_probs.parquet')


def test_run_stops_on_a_changed_probability_file(sources, tmp_path):
    df, root = sources
    copy = tmp_path / 'sources'
    shutil.copytree(root, copy)
    meta = json.load(open(copy / 'twins' / 'run.json'))
    json.dump(meta | {'option_probs_sha256': '0' * 64}, open(copy / 'twins' / 'run.json', 'w'))
    with pytest.raises(RuntimeError, match='differs from the file its run wrote'):
        run('check', 'twins', 'supervised', source_dir=copy, out_dir=tmp_path / 'out', df=df)
    assert not (tmp_path / 'out').exists()                                    # nothing written


def test_run_stops_if_supervised_scores_do_not_reproduce(sources, tmp_path):
    df, root = sources
    copy = tmp_path / 'sources'
    shutil.copytree(root, copy)
    scores = pd.read_csv(copy / 'supervised' / 'scores.csv')
    scores.loc[scores.model == 'baseline', 'log_loss'] += 0.01
    scores.to_csv(copy / 'supervised' / 'scores.csv', index=False, float_format='%.6f')
    with pytest.raises(RuntimeError, match='do not reproduce'):
        run('check', 'twins', 'supervised', source_dir=copy, out_dir=tmp_path / 'out', df=df)
    assert not (tmp_path / 'out').exists()
