import numpy as np
import pandas as pd
import pytest

from sdt.mvp.clean import RAW_COLUMNS, clean, load_raw
from sdt.paths import RAW_CSV

# One valid raw interview; tests override single cells
BASE = {'CASEID': 1, 'YYYYMM': 202001, 'WT': 1.0, 'ID': 10, 'IDPREV': np.nan, 'DATEPR': np.nan, 'METHOD': 1,
        'PAGO': 1, 'PEXP': 3, 'BUS12': 5, 'BUS5': 1, 'DUR': 1,
        'YTL5': 3, 'HOMEOWN': 1, 'INVEST': 1, 'AGE': 40, 'REGION': 1, 'SEX': 2, 'MARRY': 1, 'NUMKID': 0,
        'NUMADT': 2, 'EDUC': 4, 'POLAFF': 2}


def one(**cells):
    return clean(pd.DataFrame([{**BASE, **cells}]))


def test_base_row_is_complete():
    assert list(BASE) == RAW_COLUMNS
    assert one().notna().drop(columns=['IDPREV', 'DATEPR']).all(axis=None)


@pytest.mark.parametrize('col, code, expected', [
    ('PAGO', 1, 'fav'), ('PAGO', 3, 'mid'), ('PAGO', 5, 'unf'),
    ('BUS12', 2, 'fav'), ('BUS12', 3, 'mid'), ('BUS12', 4, 'unf'),
    ('BUS5', 2, 'fav'), ('BUS5', 4, 'unf'),
    ('DUR', 3, 'mid'),
])
def test_outcome_classes(col, code, expected):
    assert one(**{col: code})[col][0] == expected


@pytest.mark.parametrize('col, code', [
    ('PAGO', 8), ('PEXP', 9), ('BUS12', 8), ('BUS5', 98), ('BUS5', 99), ('DUR', 9),
    ('HOMEOWN', 99), ('NUMADT', 8), ('NUMADT', 9), ('POLAFF', 8), ('POLAFF', 9),
    ('AGE', np.nan), ('SEX', np.nan), ('YTL5', np.nan),
])
def test_missing_codes_become_nan(col, code):
    assert pd.isna(one(**{col: code})[col][0])


@pytest.mark.parametrize('col, code', [('PAGO', 2), ('BUS5', 8), ('MARRY', 2), ('METHOD', 3), ('INVEST', 8)])
def test_unknown_code_raises(col, code):
    with pytest.raises(ValueError, match=col):
        one(**{col: code})


def test_recodes():
    assert [one(EDUC=e).EDUC[0] for e in range(1, 7)] == [1, 1, 2, 3, 4, 5]
    assert one(INVEST=5).INVEST[0] == 0
    assert one(METHOD=2).web[0] == 1
    assert one(POLAFF=3).POLAFF[0] == 'Ind'


def test_window_starts_feb_2017():
    raw = pd.DataFrame([{**BASE, 'YYYYMM': 201701}, {**BASE, 'YYYYMM': 201702}])
    assert clean(raw).YYYYMM.tolist() == [201702]


@pytest.mark.skipif(not RAW_CSV.exists(), reason='data/data.csv not downloaded')
def test_real_data():
    df = clean(load_raw())                                 # raises if any raw code is outside the spec
    assert len(df) == 77_050 and df.CASEID.is_unique
    web = df[df.web == 1]
    assert web.YYYYMM.min() == 202404
    # Missing shares from the pre-cleaning profile (2026-10-15 download)
    assert web.AGE.isna().mean() == pytest.approx(0.128, abs=0.005)
    assert web.POLAFF.isna().mean() == pytest.approx(0.104, abs=0.005)
    assert (web.BUS12 == 'mid').mean() == pytest.approx(0.359, abs=0.005)
