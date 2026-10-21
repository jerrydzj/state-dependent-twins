import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from sdt.mvp.clean import clean, load_raw
from sdt.mvp.encode import MNL_LEVELS, NEVER_MISSING, gb_features, mnl_features
from sdt.paths import RAW_CSV
from test_clean import BASE

MNL_COLUMNS = sum(len(lv) - 1 + (c not in NEVER_MISSING) for c, (lv, _) in MNL_LEVELS.items()) + 2  # + INVEST, web


def rows(*overrides):
    return clean(pd.DataFrame([{**BASE, **o} for o in overrides]))


def test_mnl_columns_fixed_and_complete():
    X = mnl_features(rows({}))
    assert X.shape == (1, MNL_COLUMNS)
    assert X.notna().all(axis=None) and (X.dtypes == float).all()


@pytest.mark.parametrize('col, raw, feature', [
    ('AGE', 18, 'AGE=18-34'), ('AGE', 34, 'AGE=18-34'), ('AGE', 35, 'AGE=35-49'), ('AGE', 97, 'AGE=75+'),
    ('AGE', np.nan, 'AGE=NR'), ('NUMKID', 5, 'NUMKID=3+'), ('NUMADT', 9, 'NUMADT=NR'), ('NUMADT', 5, 'NUMADT=4+'),
    ('EDUC', 2, 'EDUC=1'), ('POLAFF', 8, 'POLAFF=NR'), ('YTL5', np.nan, 'YTL5=NR'), ('SEX', 2, 'SEX=female'),
])
def test_mnl_level(col, raw, feature):
    X = mnl_features(rows({col: raw}))
    own = [f for f in X if f.startswith(col + '=')]
    assert X.loc[0, own][X.loc[0, own] == 1].index.tolist() == [feature]


def test_mnl_reference_is_all_zeros():
    X = mnl_features(rows({'AGE': 55, 'POLAFF': 3}))
    assert X.filter(regex='^(AGE|POLAFF)=').sum(axis=None) == 0


def test_gb_types_and_missing():
    X = gb_features(rows({}, {'AGE': np.nan, 'POLAFF': 9, 'METHOD': 2}))
    assert np.isnan(X.AGE[1]) and pd.isna(X.POLAFF[1]) and X.web.tolist() == [0, 1]
    assert X.AGE.dtype == float and X.POLAFF.dtype == 'category'
    assert X.POLAFF.cat.categories.tolist() == ['Rep', 'Dem', 'Ind']


@pytest.mark.skipif(not RAW_CSV.exists(), reason='data/data.csv not downloaded')
def test_real_data_fits():
    df = clean(load_raw()).sample(5000, random_state=0)
    Xm, Xg = mnl_features(df), gb_features(df)
    assert Xm.shape == (5000, MNL_COLUMNS) and Xm.notna().all(axis=None)
    has_y = df.PAGO.notna()
    HistGradientBoostingClassifier(max_iter=5).fit(Xg[has_y], df.PAGO[has_y])  # accepts NaN inputs and categoricals
