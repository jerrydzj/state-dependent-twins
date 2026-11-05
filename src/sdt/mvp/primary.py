"""The primary test of the mvp-twin preregistration (docs/prereg/mvp-twin.md, section 4).

1. Dissimilarity d_t: for each evaluation month, the Euclidean distance from its 10 state variables to those of the
   nearest training month, with each variable standardised by its mean and standard deviation over the 90 training
   months. The standard deviation is the population one; any other choice multiplies every distance by the same
   factor, so it leaves the ranks of d_t unchanged.
2. Monthly distributional score: the Wasserstein distance between a model's predicted and the actual weighted answer
   shares of all respondents who answered the item in month t (score.wasserstein).
3. Advantage A_t against a comparator: the comparator's monthly score minus the shifted twin's, averaged over the five
   items.
4. Statistic: the Spearman correlation rho between d_t and A_t across the 19 evaluation months.
5. Interval: the 2.5th and 97.5th percentiles of rho over 1,000 bootstrap draws (score.draw_counts, seed 0). Each draw
   redraws every month's interviews with replacement, and every item and model is scored on that same draw.
6. Outcome: support if rho's interval lies above 0 against both comparators, contradiction if it lies below 0 against
   either, and inconclusive otherwise.

Exploratory, not in the plan: rho with each month left out in turn, which shows whether one month carries the result.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sdt.mvp import encode
from sdt.mvp.clean import CLASSES
from sdt.mvp.score import class_index, draw_counts, wasserstein
from sdt.mvp.windows import split

TWIN = 'twin_shift_demographics_macro_politics'
COMPARATORS = ['mnl_demographics_macro_politics', 'baseline']


def dissimilarity(train, evaluation):
    """d_t for each row of `evaluation`: the Euclidean distance to the nearest row of `train`, with every column
    standardised by its mean and population standard deviation over the rows of `train`."""
    mean, sd = train.mean(), train.std(ddof=0)
    if (sd == 0).any():
        raise ValueError(f'constant over the training rows: {list(sd.index[sd == 0])}')
    Z, E = ((train - mean) / sd).to_numpy(float), ((evaluation - mean) / sd).to_numpy(float)
    return pd.Series(np.sqrt(((E[:, None] - Z[None]) ** 2).sum(axis=-1)).min(axis=1), index=evaluation.index, name='d')


def state_dissimilarity(df):
    """d_t for the evaluation months of the clean table `df`, from the 10 state variables of state.py."""
    S = encode.state_table(tuple(sorted(df.YYYYMM.unique())))
    train, test = split(df)
    return dissimilarity(S.loc[sorted(train.YYYYMM.unique())], S.loc[sorted(test.YYYYMM.unique())])


def monthly_distances(pred, df, draws, models):
    """Wasserstein distance per item, model and month of `df`, on the sample and on every draw of `draws`
    (score.draw_counts). Returns a table of the sample's distances (item, model, month, wasserstein) and a dict
    (item, model) -> array of draws x months, months ascending. Every model must predict the same interviews."""
    key = df.set_index('CASEID')
    observed, by_draw = [], {}
    for item, g in pred[pred.model.isin(models)].groupby('item', sort=False):
        P = {m: gm.set_index('CASEID') for m, gm in g.groupby('model', sort=False)}
        if set(P) != set(models):
            raise ValueError(f'{item}: no predictions from {sorted(set(models) - set(P))}')
        ids = P[models[0]].index
        if any(set(p.index) != set(ids) for p in P.values()):
            raise ValueError(f'{item}: the models predict different interviews')
        a = key.loc[ids]
        Y = np.eye(len(CLASSES))[class_index(a[item].astype(str))]
        w, month = a.WT.to_numpy(float), a.YYYYMM.to_numpy()
        counts = draws.loc[ids].to_numpy().T                      # times each interview is drawn, one row per draw
        Pm = {m: P[m].loc[ids, CLASSES].to_numpy() for m in models}
        months = np.unique(month)
        for m in models:
            by_draw[item, m] = np.empty((len(counts), len(months)))
        for j, t in enumerate(months):
            s = month == t
            actual = w[s] @ Y[s] / w[s].sum()
            cw = counts[:, s] * w[s]                                 # each draw's weight on each respondent
            total = cw.sum(axis=1, keepdims=True)
            if (total == 0).any():
                raise ValueError(f'{item}, {t}: a draw contains none of these interviews, so its shares are undefined')
            drawn_actual = cw @ Y[s] / total
            for m in models:
                observed.append({'item': item, 'model': m, 'month': t,
                                 'wasserstein': wasserstein(w[s] @ Pm[m][s] / w[s].sum(), actual)})
                by_draw[item, m][:, j] = wasserstein(cw @ Pm[m][s] / total, drawn_actual)
    return pd.DataFrame(observed), by_draw


def advantages(observed, by_draw, twin, comparators):
    """A_t against each comparator: its monthly distance minus the twin's, averaged over items. Returns the sample's
    A_t (months x comparators) and a dict comparator -> array of draws x months."""
    W = observed.pivot(index='month', columns=['model', 'item'], values='wasserstein')
    if W.isna().any().any():
        raise ValueError('every item must be scored in every month')
    items = list(W[twin].columns)
    A = pd.DataFrame({c: (W[c][items] - W[twin][items]).mean(axis=1) for c in comparators})
    drawn = {c: np.mean([by_draw[i, c] - by_draw[i, twin] for i in items], axis=0) for c in comparators}
    return A, drawn


def spearman(d, A):
    """Spearman correlation of the vector `d` with `A`, or with each row of `A` if it is 2-D. Ties get average ranks,
    as in scipy.stats.spearmanr."""
    rd, rA = rankdata(d), rankdata(A, axis=-1)
    rd, rA = rd - rd.mean(), rA - rA.mean(axis=-1, keepdims=True)
    return (rA @ rd) / np.sqrt((rA ** 2).sum(axis=-1) * (rd ** 2).sum())


def outcome(rho):
    """The plan's decision rule, from a table with `lo` and `hi` per comparator."""
    if (rho.hi < 0).any():
        return 'contradiction'
    return 'support' if (rho.lo > 0).all() else 'inconclusive'


def primary_test(pred, df, d, twin=TWIN, comparators=COMPARATORS, n_boot=1000, seed=0, draws=None):
    """Steps 2 to 6 for the predictions `pred` (the score format with a `model` column) of the evaluation interviews
    `df`, with d_t given as `d` (a Series indexed by month). `draws` defaults to score.draw_counts of `df`. Returns a
    dict of:

    - d and A: d_t, and A_t per comparator (months x comparators).
    - rho: per comparator, rho on the sample (`observed`) and its interval (`lo`, `hi`).
    - rho_draws: rho on every draw (draws x comparators).
    - leave_one_out: rho with each month left out (exploratory; left-out month x comparators).
    - distances: the sample's monthly distance per item and model.
    - outcome: support, contradiction or inconclusive.
    """
    if draws is None:
        draws = draw_counts(df, n_boot, seed)
    observed, by_draw = monthly_distances(pred, df, draws, [twin, *comparators])
    A, drawn = advantages(observed, by_draw, twin, comparators)
    if not np.array_equal(d.index, A.index):
        raise ValueError('d must be indexed by the scored months, ascending')
    dv = d.to_numpy(float)
    rho_draws = pd.DataFrame({c: spearman(dv, drawn[c]) for c in comparators})
    rho = pd.DataFrame({'observed': [spearman(dv, A[c].to_numpy()) for c in comparators],
                        'lo': [np.percentile(rho_draws[c], 2.5) for c in comparators],
                        'hi': [np.percentile(rho_draws[c], 97.5) for c in comparators]},
                       index=pd.Index(comparators, name='comparator'))
    keep = {t: A.index != t for t in A.index}
    leave_one_out = pd.DataFrame({c: {t: spearman(dv[k], A[c].to_numpy()[k]) for t, k in keep.items()}
                                  for c in comparators}).rename_axis('left_out')
    return {'d': d, 'A': A, 'rho': rho, 'rho_draws': rho_draws, 'leave_one_out': leave_one_out,
            'distances': observed, 'outcome': outcome(rho)}
