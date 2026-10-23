"""Clean the raw Surveys of Consumers file for the MVP, as described in section 1 of
notebooks/03_mvp_first_run.ipynb.

One row per interview from FIRST_MONTH on. Outcomes are fav/mid/unf, with DK/NA as NaN (models drop them per item).
Ordered inputs are nullable integers, unordered inputs are categoricals, and missing is NaN throughout. Encoding for
each model family happens later, not here.

    python -m sdt.mvp.clean     # writes data/processed/mvp_clean.parquet
"""
import pandas as pd

from sdt.paths import PROCESSED, RAW_CSV

FIRST_MONTH = 201702                                   # W1: first month POLAFF is asked every month
OUT_PATH = PROCESSED / 'mvp_clean.parquet'

CLASSES = ['fav', 'mid', 'unf']
FAV3 = {1: 'fav', 3: 'mid', 5: 'unf'}
FAV5 = {1: 'fav', 2: 'fav', 3: 'mid', 4: 'unf', 5: 'unf'}   # 2 and 4 ("with qualifications") occur only on phone

# Each entry is column: (raw code -> clean value, raw codes that mean missing). Any other code raises.
OUTCOMES = {
    'PAGO': (FAV3, [8, 9]),
    'PEXP': (FAV3, [8, 9]),
    'BUS12': (FAV5, [8, 9]),
    'BUS5': (FAV5, [98, 99]),
    'DUR': (FAV3, [8, 9]),
}
ORDERED = {
    'YTL5': ({q: q for q in range(1, 6)}, []),
    'AGE': ({a: a for a in range(18, 98)}, []),            # 97 = 97 or older
    'NUMKID': ({k: k for k in range(0, 6)}, []),           # 5 = 5 or more
    'NUMADT': ({k: k for k in range(1, 6)}, [8, 9]),       # web missing is blank to Nov 2024, code 9 after
    'EDUC': ({1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}, []),    # 1-2 merged: no HS diploma
}
UNORDERED = {
    'HOMEOWN': ({1: 'own', 2: 'rent'}, [99]),
    'REGION': ({1: 'W', 2: 'NC', 3: 'NE', 4: 'S'}, []),
    'SEX': ({1: 'male', 2: 'female'}, []),
    'MARRY': ({1: 'married', 3: 'divorced', 4: 'widowed', 5: 'never'}, []),
    'POLAFF': ({1: 'Rep', 2: 'Dem', 3: 'Ind'}, [8, 9]),
}
BINARY = {
    'INVEST': ({1: 1, 5: 0}, []),                          # 5 pools no, DK and NA
    'METHOD': ({1: 0, 2: 1}, []),                          # becomes `web`; codes 3-4 end before 2017
}
DESIGN = ['CASEID', 'YYYYMM', 'WT', 'ID', 'IDPREV', 'DATEPR']
INPUTS = ['YTL5', 'HOMEOWN', 'INVEST', 'AGE', 'REGION', 'SEX', 'MARRY', 'NUMKID', 'NUMADT', 'EDUC', 'POLAFF']
RAW_COLUMNS = DESIGN + ['METHOD'] + list(OUTCOMES) + INPUTS


def recode(s, mapping, missing):
    """Map raw codes to clean values and missing codes to NaN. Raise on any code the spec doesn't cover."""
    unknown = set(s.dropna().unique()) - set(mapping) - set(missing)
    if unknown:
        raise ValueError(f'{s.name}: codes not in spec: {sorted(unknown)}')
    return s.map(mapping)


def clean(raw):
    d = raw[raw.YYYYMM >= FIRST_MONTH]
    out = d[DESIGN].astype({'IDPREV': 'Int64', 'DATEPR': 'Int64'})
    out['web'] = recode(d.METHOD, *BINARY['METHOD']).astype('int8')
    for c, (m, na) in OUTCOMES.items():
        out[c] = pd.Categorical(recode(d[c], m, na), categories=CLASSES)
    for c in INPUTS:
        if c in ORDERED:
            out[c] = recode(d[c], *ORDERED[c]).astype('Int8')
        elif c in UNORDERED:
            m, na = UNORDERED[c]
            out[c] = pd.Categorical(recode(d[c], m, na), categories=list(m.values()))
        else:
            out[c] = recode(d[c], *BINARY[c]).astype('int8')
    return out.reset_index(drop=True)


def load_raw():
    return pd.read_csv(RAW_CSV, skipinitialspace=True, usecols=RAW_COLUMNS)


if __name__ == '__main__':
    df = clean(load_raw())
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f'{len(df):,} interviews, {df.YYYYMM.min()}-{df.YYYYMM.max()} -> {OUT_PATH.relative_to(OUT_PATH.parents[2])}')
