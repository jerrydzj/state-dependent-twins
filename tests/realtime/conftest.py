import pandas as pd
import pytest

from sdt.data.clean import OUT_PATH


@pytest.fixture(scope='session')
def to_july_2024():
    """Real interviews to July 2024: 15 per month and mode, and 60 web interviews per month from April 2024."""
    if not OUT_PATH.exists():
        pytest.skip('data/processed/clean.parquet not built')
    df = pd.read_parquet(OUT_PATH)
    df = df[df.YYYYMM <= 202407]
    web = df[df.web == 1].groupby('YYYYMM').head(60)
    return pd.concat([df[df.web == 0].groupby('YYYYMM').head(15), web]).sort_index().reset_index(drop=True)
