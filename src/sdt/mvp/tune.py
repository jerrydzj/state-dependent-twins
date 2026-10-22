"""Month-blocked cross-validation of gradient boosting inside the training window.

Whole months are assigned to N_FOLDS folds, so no month is split between fitting and validation (once state variables
are added, a model can't memorise a held-out month). Phone-only and web months are shuffled separately and dealt round
robin, so web months spread across folds and every fit sees web rows. One fit per setting gives the loss after every
tree count via staged predictions. The chosen setting minimises weighted log loss averaged over the folds.
"""
import numpy as np
import pandas as pd

from sdt.mvp.clean import CLASSES
from sdt.mvp.encode import gb_features
from sdt.mvp.models import gb
from sdt.mvp.score import item_data

GRID = [{'max_leaf_nodes': leaves, 'min_samples_leaf': rows} for leaves in (4, 8, 16) for rows in (50, 200, 800)]
MAX_TREES = 1000
N_FOLDS, SEED = 5, 0


def month_folds(train):
    """Fold number (0 to N_FOLDS - 1) for each training month."""
    rng = np.random.default_rng(SEED)
    has_web = train.groupby('YYYYMM').web.max()
    fold = {}
    for months in (has_web.index[has_web == 0], has_web.index[has_web == 1]):
        for i, m in enumerate(rng.permutation(months)):
            fold[m] = i % N_FOLDS
    return pd.Series(fold).sort_index()


def folds(train):
    """(fit rows, validation rows) per fold, all modes."""
    fold = train.YYYYMM.map(month_folds(train))
    for k in range(N_FOLDS):
        yield train[fold != k], train[fold == k]


def staged_loss(model, X, y, w):
    """Weighted log loss after each boosting round."""
    cols = [list(model.classes_).index(c) for c in CLASSES]
    idx = pd.Series(np.asarray(y)).map({c: i for i, c in enumerate(CLASSES)}).to_numpy(int)
    rows = np.arange(len(idx))
    return np.array([np.average(-np.log(np.clip(P[:, cols][rows, idx], 1e-15, 1)), weights=w)
                     for P in model.staged_predict_proba(X)])


def tune(train, item, grid=GRID, max_trees=MAX_TREES):
    """Return the best setting (including max_iter) and the fold-averaged validation curves, one row per setting."""
    curves = np.zeros((len(grid), max_trees))
    for fit, val in folds(train):
        X, y, w = item_data(fit, item, gb_features)
        Xv, yv, wv = item_data(val, item, gb_features)
        for i, params in enumerate(grid):
            model = gb(max_iter=max_trees, **params).fit(X, y, sample_weight=w)
            curves[i] += staged_loss(model, Xv, yv, wv) / N_FOLDS
    i, t = np.unravel_index(curves.argmin(), curves.shape)
    curves = pd.DataFrame(curves, index=pd.MultiIndex.from_frame(pd.DataFrame(grid)), columns=np.arange(1, max_trees + 1))
    return {**grid[i], 'max_iter': int(t) + 1}, curves
