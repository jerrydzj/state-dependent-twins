"""Turn the clean MVP table into features for each model family, as described in section 1 of
notebooks/03_mvp_first_run.ipynb.

Each encoder takes the output of `clean.clean` and returns one feature row per interview. Feature columns are fixed by
the spec, never inferred from the data, so training and evaluation rows always line up. The demographics encoders
(`mnl_features`, `gb_features`) use the respondent attributes. The others add state variables of `state.py` for each
interview's survey month: the 8 macro variables (`*_demographics_macro_features`), the 2 president variables
(`*_demographics_politics_features`), or all 10 (`*_demographics_macro_politics_features`).
"""
from functools import cache

import pandas as pd

from sdt.mvp.clean import INPUTS, ORDERED, UNORDERED
from sdt.mvp.state import MACRO, POLITICS, VARIABLES, build

AGE_BINS, AGE_LABELS = [17, 34, 49, 64, 74, 97], ['18-34', '35-49', '50-64', '65-74', '75+']
BUCKET = {  # ordered inputs bucketed for MNL: clean value -> level
    'AGE': lambda s: pd.cut(s.astype(float), AGE_BINS, labels=AGE_LABELS).astype(object),
    'NUMKID': lambda s: s.astype(float).clip(upper=3).map({0: '0', 1: '1', 2: '2', 3: '3+'}),
    'NUMADT': lambda s: s.astype(float).clip(upper=4).map({1: '1', 2: '2', 3: '3', 4: '4+'}),
}

# MNL one-hot: column -> (levels, reference). Every level but the reference gets a 0/1 column, plus NR for missing
# (REGION is never missing). The reference is the most common level, so each coefficient compares a group with the
# largest one.
MNL_LEVELS = {
    'YTL5': ([1, 2, 3, 4, 5], 5),
    'HOMEOWN': (['own', 'rent'], 'own'),
    'AGE': (AGE_LABELS, '50-64'),
    'REGION': (['W', 'NC', 'NE', 'S'], 'S'),
    'SEX': (['male', 'female'], 'male'),
    'MARRY': (['married', 'divorced', 'widowed', 'never'], 'married'),
    'NUMKID': (['0', '1', '2', '3+'], '0'),
    'NUMADT': (['1', '2', '3', '4+'], '2'),
    'EDUC': ([1, 2, 3, 4, 5], 4),
    'POLAFF': (['Rep', 'Dem', 'Ind'], 'Ind'),
}
NEVER_MISSING = {'REGION'}
BINARY = ['INVEST', 'web']


def one_hot(s, levels, reference, nr):
    """0/1 columns for every level except the reference, plus `NR` for missing. Raise on values outside `levels`."""
    unknown = set(s.dropna().unique()) - set(levels)
    if unknown or (not nr and s.isna().any()):
        raise ValueError(f'{s.name}: unexpected values {sorted(unknown) or "NaN"}')
    cols = {f'{s.name}={v}': (s == v).fillna(False) for v in levels if v != reference}
    if nr:
        cols[f'{s.name}=NR'] = s.isna()
    return pd.DataFrame(cols, index=s.index).astype(float)


def mnl_features(df):
    """One-hot inputs with an NR level, plus INVEST and web as 0/1. All float, no NaN."""
    parts = []
    for c in INPUTS:
        if c in MNL_LEVELS:
            s = BUCKET[c](df[c]) if c in BUCKET else df[c].astype(float if c in ORDERED else object)
            parts.append(one_hot(s.rename(c), *MNL_LEVELS[c], nr=c not in NEVER_MISSING))
        else:
            parts.append(df[[c]].astype(float))
    parts.append(df[['web']].astype(float))
    return pd.concat(parts, axis=1)


def gb_features(df):
    """Ordered inputs as floats with NaN, unordered as pandas categoricals (read natively by HistGradientBoosting's
    default categorical_features='from_dtype'), INVEST and web as 0/1."""
    out = pd.DataFrame(index=df.index)
    for c in INPUTS:
        if c in ORDERED:
            out[c] = df[c].astype(float)
        elif c in UNORDERED:
            out[c] = df[c].astype(pd.CategoricalDtype(list(UNORDERED[c][0].values())))
        else:
            out[c] = df[c].astype('int8')
    out['web'] = df['web'].astype('int8')
    return out


PARTY_X_PRESIDENT = [(p, s) for s in ['pres_R', 'winner_R'] for p in ['Rep', 'Dem']]   # independents and NR: reference


@cache
def state_table(months):
    return build(months).set_index('YYYYMM')[VARIABLES]


def state_features(df):
    """The state variables of each row's survey month, as floats."""
    return state_table(tuple(sorted(df.YYYYMM.unique()))).loc[df.YYYYMM].set_index(df.index).astype(float)


def mnl_with_state(df, columns):
    """mnl_features plus the given state variables, linear, and respondent party x each president variable if the
    president variables are included."""
    S = state_features(df)
    X = pd.concat([mnl_features(df), S[columns]], axis=1)
    if set(POLITICS) <= set(columns):
        for p, s in PARTY_X_PRESIDENT:
            X[f'POLAFF={p}:{s}'] = X[f'POLAFF={p}'] * S[s]
    return X


def gb_with_state(df, columns):
    """gb_features plus the given state variables; trees find their own interactions."""
    return pd.concat([gb_features(df), state_features(df)[columns]], axis=1)


def mnl_demographics_macro_features(df):
    return mnl_with_state(df, MACRO)


def mnl_demographics_politics_features(df):
    return mnl_with_state(df, POLITICS)


def mnl_demographics_macro_politics_features(df):
    return mnl_with_state(df, VARIABLES)


def gb_demographics_macro_features(df):
    return gb_with_state(df, MACRO)


def gb_demographics_politics_features(df):
    return gb_with_state(df, POLITICS)


def gb_demographics_macro_politics_features(df):
    return gb_with_state(df, VARIABLES)
