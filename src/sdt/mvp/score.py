"""Fit on the training window, score on the evaluation window, at two levels:

- Individual: weighted log loss of each respondent's predicted probabilities against their answer.
- Distributional: Wasserstein distance between each month's predicted and actual weighted answer shares, averaged over
  months (as in Kolluri et al. 2025, with months in place of experimental conditions).

Both are compared with the baseline shares through `skill`, and with a uniform guess and a bootstrap noise floor
through `benchmarks`.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone

from sdt.mvp.clean import CLASSES, OUTCOMES
from sdt.mvp.windows import split


def class_index(y):
    """Position of each answer in CLASSES. Raise on missing or unknown answers."""
    idx = pd.Series(np.asarray(y, dtype=object)).map({c: i for i, c in enumerate(CLASSES)})
    if idx.isna().any():
        raise ValueError('outcomes must be non-missing and in CLASSES')
    return idx.to_numpy(int)


def log_loss(y, P, w):
    """Weighted mean of -log(probability given to the answer that happened). P's columns follow CLASSES."""
    idx = class_index(y)
    p = np.asarray(P)[np.arange(len(idx)), idx]
    return np.average(-np.log(np.clip(p, 1e-15, 1)), weights=w)


def wasserstein(p, q):
    """Distance between distributions over CLASSES placed at 0, 0.5 and 1 on the response scale: the area between
    their cumulative shares. With three equally spaced classes this is half the fav gap plus half the unf gap."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return 0.5 * (np.abs(p[..., 0] - q[..., 0]) + np.abs(p[..., 2] - q[..., 2]))


def monthly_shares(P, w, month):
    """Weighted average of the rows of P within each month; one row per month."""
    P, w = np.asarray(P, float), np.asarray(w, float)
    return pd.DataFrame(P * w[:, None]).groupby(np.asarray(month)).sum().div(
        pd.Series(w).groupby(np.asarray(month)).sum(), axis=0)


def distribution_score(y, P, w, month):
    """Mean over months of the Wasserstein distance between predicted and actual weighted shares."""
    actual = monthly_shares(np.eye(len(CLASSES))[class_index(y)], w, month)
    return wasserstein(monthly_shares(P, w, month).to_numpy(), actual.to_numpy()).mean()


def noise_floor(y, w, month, n_boot=200, seed=0):
    """Score of a model that knew each month's true shares, which still differ from the observed shares by sampling
    noise. For each month, redraw its interviews with replacement (same count), take the distance between the redrawn
    and observed shares, and average over redraws, then over months."""
    rng = np.random.default_rng(seed)
    Y, w, month = np.eye(len(CLASSES))[class_index(y)], np.asarray(w, float), np.asarray(month)
    out = []
    for m in np.unique(month):
        Ym, wm = Y[month == m], w[month == m]
        observed = wm @ Ym / wm.sum()
        draw = rng.integers(0, len(wm), size=(n_boot, len(wm)))
        redrawn = np.einsum('bn,bnk->bk', wm[draw], Ym[draw]) / wm[draw].sum(axis=1, keepdims=True)
        out.append(wasserstein(redrawn, observed).mean())
    return np.mean(out)


def skill(score, baseline_score):
    """1 - score / baseline score: 0 matches the baseline, above 0 beats it, 1 would be perfect."""
    return 1 - score / baseline_score


def item_data(df, item, encode):
    """Features, outcome and weight for the rows that answered `item` (DK/NA are dropped per item)."""
    d = df[df[item].notna()]
    return encode(d), d[item].astype(str), d.WT


def predict(model, X):
    """predict_proba with columns reordered to CLASSES."""
    P = model.predict_proba(X)
    return P[:, [list(model.classes_).index(c) for c in CLASSES]]


def fit_predict(model, encode, df, items=tuple(OUTCOMES)):
    """Fit a fresh copy of `model` per item on the training window and predict the evaluation rows that answered the
    item. One row per (item, interview): CASEID and the probability of each class. `model` can also be a dict of one
    model per item (e.g. tuned settings)."""
    train, test = split(df)
    out = []
    for item in items:
        X, y, w = item_data(train, item, encode)
        m = clone(model[item] if isinstance(model, dict) else model).fit(X, y, sample_weight=w)
        X, y, _ = item_data(test, item, encode)
        out.append(pd.DataFrame(predict(m, X), columns=CLASSES).assign(item=item, CASEID=test.loc[y.index, 'CASEID'].to_numpy()))
    return pd.concat(out, ignore_index=True)[['item', 'CASEID'] + CLASSES]


def score_predictions(pred, df):
    """Log loss and Wasserstein distance per item for saved predictions, with answers and weights from the clean table."""
    key = df.set_index('CASEID')
    rows = {}
    for item, g in pred.groupby('item', sort=False):
        a = key.loc[g.CASEID]
        y, P = a[item].astype(str), g[CLASSES].to_numpy()
        rows[item] = {'log_loss': log_loss(y, P, a.WT), 'wasserstein': distribution_score(y, P, a.WT, a.YYYYMM)}
    return pd.DataFrame(rows).T


def evaluate(model, encode, df, items=tuple(OUTCOMES)):
    """Fit on the training window and return evaluation log loss and Wasserstein distance per item."""
    return score_predictions(fit_predict(model, encode, df, items), df)


def benchmarks(df, items=tuple(OUTCOMES), n_boot=200, seed=0):
    """Per item on the evaluation window: the uniform guess (1/3 each) on both scores, and the bootstrap noise floor
    on the distributional score."""
    _, test = split(df)
    rows = {}
    for item in items:
        d = test[test[item].notna()]
        y, w, month = d[item].astype(str), d.WT, d.YYYYMM
        uniform = np.full((len(d), len(CLASSES)), 1 / len(CLASSES))
        rows[item] = {'uniform_log_loss': log_loss(y, uniform, w),
                      'uniform_wasserstein': distribution_score(y, uniform, w, month),
                      'floor_wasserstein': noise_floor(y, w, month, n_boot, seed)}
    return pd.DataFrame(rows).T
