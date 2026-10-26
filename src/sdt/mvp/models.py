"""MVP models. Each follows scikit-learn's interface: fit(X, y, sample_weight) and predict_proba(X), with classes_ in
the order of the predict_proba columns."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sdt.mvp.clean import CLASSES


def mnl():
    """Multinomial logit by unpenalised, weighted maximum likelihood (C=inf turns the L2 penalty off). Use with
    encode.mnl_features."""
    return LogisticRegression(C=np.inf, max_iter=10_000, tol=1e-8)     # state fits need up to ~5,000 iterations


def gb(**params):
    """Gradient boosting with learning rate 0.05 and no built-in early stopping (it would hold out random rows, mixing
    months and people). Tree count and size come from tune.tune. Use with encode.gb_features."""
    return HistGradientBoostingClassifier(learning_rate=0.05, early_stopping=False, random_state=0, **params)


class BaselineShares(ClassifierMixin, BaseEstimator):
    """Predicts the same distribution for everyone: the weighted answer shares of the training rows. With
    web_only=True (the MVP baseline), only web rows count, since every evaluation month is web."""

    def __init__(self, web_only=True):
        self.web_only = web_only

    def fit(self, X, y, sample_weight=None):
        y, w = np.asarray(y), np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        if self.web_only:
            keep = np.asarray(X['web']) == 1
            y, w = y[keep], w[keep]
        self.classes_ = np.array(CLASSES)
        self.shares_ = np.array([w[y == c].sum() for c in CLASSES]) / w.sum()
        return self

    def predict_proba(self, X):
        return np.tile(self.shares_, (len(X), 1))
