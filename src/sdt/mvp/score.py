"""Fit on the training window, score on the evaluation window. Individual-level scores from notebooks/02_mvp_plan.ipynb
section 5: weighted log loss per item, and skill against the baseline shares."""
import numpy as np
import pandas as pd
from sklearn.base import clone

from sdt.mvp.clean import CLASSES, OUTCOMES
from sdt.mvp.windows import split


def log_loss(y, P, w):
    """Weighted mean of -log(probability given to the answer that happened). P's columns follow CLASSES."""
    idx = pd.Series(np.asarray(y, dtype=object)).map({c: i for i, c in enumerate(CLASSES)})
    if idx.isna().any():
        raise ValueError('outcomes must be non-missing and in CLASSES')
    p = np.asarray(P)[np.arange(len(idx)), idx.to_numpy(int)]
    return np.average(-np.log(np.clip(p, 1e-15, 1)), weights=w)


def skill(loss, baseline_loss):
    """1 - loss / baseline loss: 0 matches the baseline, above 0 beats it, 1 would be perfect."""
    return 1 - loss / baseline_loss


def item_data(df, item, encode):
    """Features, outcome and weight for the rows that answered `item` (DK/NA are dropped per item)."""
    d = df[df[item].notna()]
    return encode(d), d[item].astype(str), d.WT


def predict(model, X):
    """predict_proba with columns reordered to CLASSES."""
    P = model.predict_proba(X)
    return P[:, [list(model.classes_).index(c) for c in CLASSES]]


def evaluate(model, encode, df, items=tuple(OUTCOMES)):
    """Fit a fresh copy of `model` per item on the training window; return evaluation log loss per item. `model` can
    also be a dict of one model per item (e.g. tuned settings)."""
    train, test = split(df)
    loss = {}
    for item in items:
        X, y, w = item_data(train, item, encode)
        m = clone(model[item] if isinstance(model, dict) else model).fit(X, y, sample_weight=w)
        X, y, w = item_data(test, item, encode)
        loss[item] = log_loss(y, predict(m, X), w)
    return pd.Series(loss, name='log_loss')
