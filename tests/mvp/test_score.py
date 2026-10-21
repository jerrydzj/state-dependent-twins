import numpy as np
import pandas as pd
import pytest

from sdt.mvp.clean import clean, load_raw
from sdt.mvp.encode import mnl_features
from sdt.mvp.models import BaselineShares
from sdt.mvp.score import evaluate, log_loss, predict, skill
from sdt.mvp.windows import EVAL_START, HORIZON, TRAIN_END, split
from sdt.paths import DATA, RAW_CSV


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
    loss = evaluate(BaselineShares(), mnl_features, df)
    assert loss.index.tolist() == ['PAGO', 'PEXP', 'BUS12', 'BUS5', 'DUR']
    # No constant can beat the evaluation window's own shares, so the baseline's loss must be at least theirs
    _, test = split(df)
    oracle = evaluate(BaselineShares(), mnl_features, pd.concat([test.assign(YYYYMM=TRAIN_END), test]))
    assert (oracle <= loss).all() and (oracle > 0).all()
