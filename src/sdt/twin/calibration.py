"""From the twin's raw option probabilities to class probabilities, and their calibration.

`class_probs` does steps 4 and 5 of readout.py: it divides the raw option probabilities by the answer mass and maps
options to classes (`to_classes`), giving DUR, whose web form has no middle option, a given middle share.

Calibration (`fit_calibration`, `apply_calibration`) fits the twin's probabilities to recorded answers by
bias-corrected temperature scaling (Alexandari, Kundaje and Shrikumar, ICML 2020): temperature scaling (Guo et al.,
ICML 2017) plus a bias per class. The calibrated probabilities are softmax(a * log p + b), with b = (b_fav, 0, b_unf),
so each class's calibrated log-odds against the middle class are a times its raw log-odds plus its b:

- b, the shift, adds the same amount to each class's log-odds in every prediction. At the fit, the twin's weighted mean
  probability of each class equals that class's weighted share in the fitting data. b cannot tell the survey form's
  effect from the model's errors about opinions, so it absorbs any constant part of either.
- a, the scale, multiplies every difference in log-odds between two predictions by the same factor: a < 1 makes the
  twin less sure and a > 1 more sure. a is kept at 0 or above, so it can damp or amplify the model's differences but
  never reverse them.

With scale=False, a is fixed at 1, so every difference in log-odds between predictions stays the model's: the shifted
twin. Both are fitted by weighted maximum likelihood, separately for each model, prompt, feature set and item, on
interviews that each study chooses.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sdt.data.clean import CLASSES
from sdt.twin.prompts import OPTIONS, RAW


def to_classes(p, item, mid_share=None):
    """Option probabilities (rows summing to 1, columns in the web form's order) -> a DataFrame of CLASSES."""
    options = OPTIONS[item]
    out = pd.DataFrame(p, columns=options)
    if 'mid' not in options:
        if mid_share is None:
            raise ValueError(f'{item} has no middle option: pass mid_share')
        out = (out * (1 - mid_share)).assign(mid=mid_share)
    return out[CLASSES]


def class_probs(raw, item, mid_share=None):
    """Raw option probabilities of `item` (columns RAW) -> CLASSES probabilities and the answer mass, indexed like
    `raw`: divide by the mass (steps 4 and 5 of readout.py)."""
    p = raw[RAW[:len(OPTIONS[item])]].to_numpy(float)
    mass = p.sum(axis=1)
    return to_classes(p / mass[:, None], item, mid_share).assign(mass=mass).set_axis(raw.index)


def _calibrated(logp, a, b):
    """softmax(a * logp + b) row by row, with b = (b_fav, 0, b_unf)."""
    z = a * logp + np.array([b[0], 0.0, b[1]])
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def fit_calibration(p, y, w, scale=True):
    """The scale a and shift b that maximise the weighted log-likelihood of answers `y` (class labels) under
    softmax(a * log p + b), where `p` holds CLASSES probabilities; with scale=False, a = 1. Returned as a Series with
    index ['a'] + CLASSES. The likelihood is concave in (a, b), so the maximum is unique."""
    logp = np.log(np.clip(np.asarray(p[CLASSES], float), 1e-12, None))   # clip: guard against log(0)
    onehot = (np.asarray(y)[:, None] == np.array(CLASSES)).astype(float)
    missing = [c for c, n in zip(CLASSES, onehot.sum(axis=0)) if n == 0 and c != 'mid']
    if missing:     # the likelihood then keeps rising as that class's b falls, so no finite b maximises it
        raise ValueError(f'no answers in {missing}: the calibration has no maximum')
    w = np.asarray(w, float) / np.sum(w)

    def loss(theta):
        a, b = (theta[0], theta[1:]) if scale else (1.0, theta)
        q = _calibrated(logp, a, b)
        nll = -(w * np.log((q * onehot).sum(axis=1))).sum()
        resid = w[:, None] * (q - onehot)
        grad = resid.sum(axis=0)[[0, 2]]                                # d nll / d (b_fav, b_unf)
        if scale:
            grad = np.r_[(resid * logp).sum(), grad]                    # d nll / d a
        return nll, grad

    start, bounds = (np.array([1.0, 0, 0]), [(0, None), (None, None), (None, None)]) if scale else (np.zeros(2), None)
    fit = minimize(loss, start, jac=True, method='L-BFGS-B', bounds=bounds, options={'gtol': 1e-8})
    if not fit.success:
        raise RuntimeError(f'calibration did not converge: {fit.message}')
    a, b = (fit.x[0], fit.x[1:]) if scale else (1.0, fit.x)
    return pd.Series([a, b[0], 0.0, b[1]], index=['a'] + CLASSES)


def apply_calibration(p, c):
    """softmax(a * log p + b) for a DataFrame `p` of CLASSES probabilities and a calibration `c` from fit_calibration."""
    logp = np.log(np.clip(np.asarray(p[CLASSES], float), 1e-12, None))
    return pd.DataFrame(_calibrated(logp, c['a'], c[['fav', 'unf']].to_numpy()), columns=CLASSES, index=p.index)
