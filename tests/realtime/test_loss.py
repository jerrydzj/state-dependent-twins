import numpy as np
import pandas as pd
import pytest

from sdt.realtime.loss import ITEMS, actual, floors, losses


def toy_table():
    """Two web months and one phone interview; every item answered alike, except a missing PAGO answer."""
    rows = [(1, 202405, 1, 1.0, 'fav'), (2, 202405, 1, 3.0, 'unf'), (3, 202405, 1, 2.0, 'mid'),
            (4, 202405, 0, 9.0, 'fav'), (5, 202406, 1, 1.0, 'fav'), (6, 202406, 1, 1.0, 'fav')]
    df = pd.DataFrame(rows, columns=['CASEID', 'YYYYMM', 'web', 'WT', 'answer'])
    for item in ITEMS:
        df[item] = df.answer
    df.loc[df.CASEID == 3, 'PAGO'] = np.nan
    return df


def test_actual_shares_count_answering_web_interviews_only():
    a = actual(toy_table(), [202405, 202406]).set_index(['month', 'item'])
    assert a.loc[(202405, 'PEXP'), ['fav', 'mid', 'unf']].tolist() == pytest.approx([1 / 6, 2 / 6, 3 / 6])
    assert a.loc[(202405, 'PAGO'), ['fav', 'mid', 'unf']].tolist() == pytest.approx([0.25, 0, 0.75])
    assert a.loc[(202405, 'PEXP'), 'interviews'] == 3 and a.loc[(202405, 'PAGO'), 'interviews'] == 2


def test_floor_is_the_variance_of_the_weighted_shares():
    a = actual(toy_table(), [202405, 202406]).set_index(['month', 'item'])
    w, fav, unf = np.array([1.0, 3.0, 2.0]), np.array([1, 0, 0]), np.array([0, 1, 0])
    expected = sum(np.sum(w ** 2 * (y - np.average(y, weights=w)) ** 2) / w.sum() ** 2 for y in (fav, unf))
    assert a.loc[(202405, 'PEXP'), 'floor'] == pytest.approx(expected)
    assert a.loc[(202406, 'PEXP'), 'floor'] == 0                     # everyone answered fav
    assert floors(a.reset_index()).loc[202406] == 0


def test_loss_averages_cumulative_errors_over_all_five_items():
    a = actual(toy_table(), [202405, 202406])
    f = a[['month', 'item']].assign(candidate='c', data_end=202404, fav=0.5, mid=0.2, unf=0.3)
    L = losses(f, a).set_index('month').loss
    per_item = (0.5 - a.fav) ** 2 + (0.3 - a.unf) ** 2
    assert L.loc[202405] == pytest.approx(per_item[a.month == 202405].mean())
    assert 202406 in L.index and len(losses(f[f.item != 'DUR'], a)) == 0     # a missing item means no loss
