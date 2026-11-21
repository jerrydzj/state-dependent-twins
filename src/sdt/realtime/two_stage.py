"""The two-stage ridge of the realtime-benchmark plan (section 6, candidate 6): 26 candidates.

**Stage 1**, for each item, on every interview from February 2017 to data end e:

1. A multinomial logit gives, for respondent i in month m, the log-odds of fav and of unf against mid as
   level_(m,c) + web_c x web_i + r_i . beta_c, for c = fav, unf. The month intercepts level_(m,c) are the month's
   levels, and web_c is the web indicator's coefficient; both are unpenalized.
2. The respondent terms r_i are the 11 attributes in the logit encoding with NR (features.mnl_features without `web`),
   plus, on input sets with P, the four party x president products. Each column is standardized to weighted mean 0
   and standard deviation 1 on the fitting rows, as in models.Ridge.
3. The fit minimizes the WT-weighted negative log-likelihood plus |beta|^2 / (2C), with C on the scale of
   scikit-learn's penalized logit, so the C grid is that of notebook 04. Only the fav and unf contrasts against mid
   are penalized, whereas scikit-learn's multinomial logit penalizes one coefficient vector per class.
4. C is tuned by 5-fold cross-validation over interviews. After a shuffle (seed 0), interviews are dealt to folds in
   turn within each month and answer, so a month and answer with at least two interviews appears in the fitting rows
   of every fold. Each fold fits the C grid from the strongest penalty to none, starting each fit from the previous
   one that converged. A C whose fit does not converge in some fold, as happens without a penalty when an attribute
   level predicts its answers perfectly, gets an infinite loss. The chosen C has the lowest weighted log loss averaged
   over the folds, and the model is refitted on every interview of the months that have levels (step 5).
5. A month in which some answer has no interviews has no finite level, because the likelihood keeps rising as that
   intercept moves. Such a month is left out of the fit (out of a fold's fit, and its validation rows, if the answer
   is missing from the fold's fitting rows) and has no level.

D and D+M share stage 1, because M enters only in stage 2, and so do D+P and D+M+P.

**Stage 2** forecasts each contrast's level at a target month t from the levels of the months up to e, in one of
three ways:

- `smooth-a<alpha>`: exponential smoothing over the levels of the web months, in order, starting from the first web
  month's level. The forecast is the smoothed level at e.
- `level-regression`: a ridge regression of every month's level on the input set's state variables. The forecast is
  its prediction at month t's state variables.
- `anchored-change`: the level at e plus beta . (x_t - x_e), where beta comes from a ridge regression, with an
  unpenalized intercept, of every month-to-month change in level on the change in the state variables x. The
  forecast leaves the intercept out, so it carries no trend forward. If month e has no level, the latest month with
  one takes its place.

Both regressions use only months with levels. Their state variables are standardized on the regression's rows, and
the penalty is tuned over `stage2_alphas` (strongest first, then none) by month-blocked cross-validation with the
folds of tune.month_folds on the stage-1 interviews, minimizing mean squared error averaged over the folds.

If stage 1 has no maximum or does not converge at the chosen C, or no web month has a level, the candidates concerned
have no fit at e, and the unit records why under `errors`.

**Prediction.** A respondent's probabilities are the softmax of the forecast level plus web_c (every target interview
is web) plus r_i . beta_c for fav and unf, and 0 for mid.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sdt import tune
from sdt.data.clean import CLASSES
from sdt.data.state import POLITICS
from sdt.features import FEATURE_SETS, mnl_demographics_politics_features, mnl_features, state_table
from sdt.realtime import candidates as names
from sdt.realtime.loss import web_answers, weighted_shares
from sdt.realtime.months import FIRST_WEB, add, targets

SPECS = {'D': ['demographics', 'demographics_macro'], 'D+P': ['demographics_politics', 'demographics_macro_politics']}
CONTRASTS = ['fav', 'unf']


class NotConverged(RuntimeError):
    pass


def respondent_terms(d, spec):
    """The respondent columns of stage 1 for spec 'D' or 'D+P'."""
    if spec == 'D':
        return mnl_features(d).drop(columns='web')
    return mnl_demographics_politics_features(d).drop(columns=['web', *POLITICS])


def one_hot(y):
    return (np.asarray(y, dtype=object)[:, None] == np.array(CLASSES)).astype(float)


def identified(month, Y, w):
    """Months in which every answer has positive weight."""
    weight = pd.DataFrame(Y * w[:, None]).groupby(month).sum()
    return set(weight.index[(weight > 0).all(axis=1)])


def objective(theta, month, web, R, Y, w, n_months, inv_C, total):
    """The stage-1 objective, divided by the total weight, and its gradient."""
    p = R.shape[1]
    level = theta[:2 * n_months].reshape(n_months, 2)
    web_coef = theta[2 * n_months:2 * n_months + 2]
    beta = theta[2 * n_months + 2:].reshape(p, 2)
    eta = level[month] + web[:, None] * web_coef + R @ beta             # log-odds of fav and unf against mid
    top = np.maximum(eta.max(axis=1), 0)
    ef, eu, em = np.exp(eta[:, 0] - top), np.exp(eta[:, 1] - top), np.exp(-top)
    norm = ef + em + eu
    log_lik = Y[:, 0] * eta[:, 0] + Y[:, 2] * eta[:, 1] - top - np.log(norm)
    value = -(w @ log_lik) + 0.5 * inv_C * np.sum(beta ** 2)
    resid = np.column_stack([ef / norm - Y[:, 0], eu / norm - Y[:, 2]]) * w[:, None]
    g_level = np.column_stack([np.bincount(month, resid[:, k], n_months) for k in range(2)])
    grad = np.concatenate([g_level.ravel(), web @ resid, (R.T @ resid + inv_C * beta).ravel()])
    return value / total, grad / total


def probabilities(eta):
    """Class probabilities (columns CLASSES) from the log-odds of fav and unf against mid."""
    z = np.column_stack([eta[:, 0], np.zeros(len(eta)), eta[:, 1]])
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


class StageOne:
    """Stage 1 for one item and spec on the interviews `d` (steps 1 to 5)."""

    def __init__(self, d, item, spec, C_grid, n_folds=5, seed=0):
        self.spec = spec
        months = np.array(sorted(d.YYYYMM.unique()))
        month = np.searchsorted(months, d.YYYYMM.to_numpy())
        web = d.web.to_numpy(float)
        R = respondent_terms(d, spec)
        self.columns = list(R.columns)
        R = R.to_numpy(float)
        Y, w = one_hot(d[item].astype(str)), d.WT.to_numpy(float)
        fold = self.folds(month, Y, n_folds, seed)
        cv = np.zeros(len(C_grid))
        for k in range(n_folds):
            fit, val = fold != k, fold == k
            keep = np.isin(month, sorted(identified(month[fit], Y[fit], w[fit])))
            fit, val = fit & keep, val & keep
            theta = None
            for j, C in enumerate(C_grid):
                try:
                    theta, mean, sd = self.fit(R, month, web, Y, w, fit, len(months), C, theta)
                except NotConverged:
                    cv[j] = np.inf
                    continue
                eta = self.eta(theta, month[val], web[val], (R[val] - mean) / sd, len(months))
                p = (probabilities(eta) * Y[val]).sum(axis=1)
                cv[j] += np.average(-np.log(np.clip(p, 1e-15, 1)), weights=w[val]) / n_folds
        if np.isinf(cv).all():
            raise NotConverged('stage 1 converged at no C of the grid')
        self.cv = cv
        self.C = C_grid[int(cv.argmin())]
        ok = sorted(identified(month, Y, w))
        rows = np.isin(month, ok)
        theta, self.mean, self.sd = self.fit(R, month, web, Y, w, rows, len(months), self.C, None)
        n = len(months)
        level = theta[:2 * n].reshape(n, 2)
        self.levels = pd.DataFrame(level, index=months, columns=CONTRASTS).loc[months[ok]].rename_axis('month')
        self.dropped = [int(m) for m in months if m not in set(months[ok])]
        self.web_coef = theta[2 * n:2 * n + 2]
        self.beta = theta[2 * n + 2:].reshape(-1, 2)

    @staticmethod
    def folds(month, Y, n_folds, seed):
        """Fold of each interview: dealt in turn within each month and answer, after a shuffle."""
        key = np.random.default_rng(seed).random(len(month))
        order = np.lexsort((key, Y.argmax(axis=1), month))
        fold = np.empty(len(month), int)
        fold[order] = np.arange(len(month)) % n_folds
        return fold

    @staticmethod
    def eta(theta, month, web, Z, n_months):
        level = theta[:2 * n_months].reshape(n_months, 2)
        web_coef, beta = theta[2 * n_months:2 * n_months + 2], theta[2 * n_months + 2:].reshape(-1, 2)
        return level[month] + web[:, None] * web_coef + Z @ beta

    @staticmethod
    def fit(R, month, web, Y, w, rows, n_months, C, start):
        """Fit on `rows` with penalty C from `start` (None: each month's observed log-odds, other terms 0). Returns
        the parameters and the standardization of R."""
        mean = np.average(R[rows], axis=0, weights=w[rows])
        sd = np.sqrt(np.average((R[rows] - mean) ** 2, axis=0, weights=w[rows]))
        sd[sd == 0] = 1
        Z = (R[rows] - mean) / sd
        if start is None:
            counts = pd.DataFrame(Y[rows] * w[rows, None]).groupby(month[rows]).sum()
            counts = counts.reindex(range(n_months)).fillna(1.0)       # months without rows: start at even odds
            counts = counts.to_numpy() + 1e-3
            start = np.concatenate([np.log(counts[:, [0, 2]] / counts[:, [1]]).ravel(), np.zeros(2 + 2 * R.shape[1])])
        inv_C = 0.0 if np.isinf(C) else 1 / C
        args = (month[rows], web[rows], Z, Y[rows], w[rows], n_months, inv_C, w[rows].sum())
        # ftol 1e-15 lets the fit stop on the gradient (gtol), so the score equations of step 1 hold closely
        res = minimize(objective, start, args=args, jac=True, method='L-BFGS-B',
                       options={'maxiter': 20_000, 'maxfun': 20_000, 'ftol': 1e-15, 'gtol': 1e-8})
        if not res.success:
            raise NotConverged(f'stage 1 did not converge at C = {C}: {res.message}')
        return res.x, mean, sd

    def log_odds(self, d):
        """web_c x web_i + r_i . beta_c for the interviews d (rows x CONTRASTS)."""
        R = respondent_terms(d, self.spec)[self.columns].to_numpy(float)
        return d.web.to_numpy(float)[:, None] * self.web_coef + ((R - self.mean) / self.sd) @ self.beta


def fit_ridge(X, y, alpha):
    """Ridge regression with an unpenalized intercept on standardized X: (mean, sd, y-bar, beta)."""
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1
    Z = (X - mean) / sd
    A = np.vstack([Z, np.sqrt(alpha) * np.eye(X.shape[1])])
    b = np.concatenate([y - y.mean(), np.zeros(X.shape[1])])
    return mean, sd, y.mean(), np.linalg.lstsq(A, b, rcond=None)[0]


def predict_ridge(model, X):
    mean, sd, ybar, beta = model
    return ybar + ((X - mean) / sd) @ beta


def tuned_ridge(X, y, fold, alphas):
    """The ridge regression with the penalty of lowest mean squared error averaged over the month folds, and that
    penalty."""
    loss = np.zeros(len(alphas))
    used = [k for k in np.unique(fold) if (fold == k).any() and (fold != k).sum() > 1]
    for j, a in enumerate(alphas):
        for k in used:
            model = fit_ridge(X[fold != k], y[fold != k], a)
            loss[j] += np.mean((predict_ridge(model, X[fold == k]) - y[fold == k]) ** 2) / len(used)
    alpha = alphas[int(loss.argmin())]
    return fit_ridge(X, y, alpha), alpha


def stage_two(levels, fs, option, alphas, month_fold, state):
    """The forecast level of each contrast at each month in `state` (the state variables of the target months), and
    the chosen penalty per contrast (None for smoothing). `option` is 'level-regression', 'anchored-change' or a
    smoothing alpha. Raises ValueError if smoothing finds no web month with a level."""
    if isinstance(option, float):
        web = levels[levels.index >= FIRST_WEB]
        if web.empty:
            raise ValueError('no web month has a level')
        level = web.iloc[0].to_numpy()
        for m in web.index[1:]:
            level = option * web.loc[m].to_numpy() + (1 - option) * level
        return pd.DataFrame([level] * len(state), index=state.index, columns=CONTRASTS), None
    columns = FEATURE_SETS[fs]
    S = state_table(tuple(sorted(set(levels.index) | set(state.index))))[columns]
    if option == 'level-regression':
        X, rows = S.loc[levels.index].to_numpy(float), levels.index
    else:
        rows = [m for m in levels.index if add(m, -1) in levels.index]
        X = (S.loc[rows].to_numpy(float) - S.loc[[add(m, -1) for m in rows]].to_numpy(float))
    fold = month_fold.loc[rows].to_numpy()
    out, chosen = {}, {}
    for c in CONTRASTS:
        y = levels.loc[rows, c].to_numpy() if option == 'level-regression' else (
            levels.loc[rows, c].to_numpy() - levels.loc[[add(m, -1) for m in rows], c].to_numpy())
        model, chosen[c] = tuned_ridge(X, y, fold, alphas)
        if option == 'level-regression':
            out[c] = predict_ridge(model, state[columns].to_numpy(float))
        else:
            anchor = levels.index[-1]                                    # e, or the latest month before it with a level
            mean, sd, _, beta = model
            change = (state[columns].to_numpy(float) - S.loc[anchor].to_numpy(float)) / sd
            out[c] = levels.loc[anchor, c] + change @ beta
    return pd.DataFrame(out, index=state.index), chosen


def fit_forecast(ctx, e, spec, item):
    """Forecast rows of every two-stage candidate of `spec` for one item at data end e, and the fitted settings."""
    train = ctx.df[ctx.df.YYYYMM <= e]
    d = train[train[item].notna()]
    try:
        one = StageOne(d, item, spec, list(ctx.settings.stage1_C), ctx.settings.stage1_folds, ctx.settings.seed)
    except NotConverged as err:
        return [], {'errors': [f'stage 1: {err}']}
    settings = {'C': None if np.isinf(one.C) else one.C,
                'cv_log_loss': [None if np.isinf(x) else float(x) for x in one.cv],    # None: did not converge
                'months_without_level': one.dropped, 'web_coef': dict(zip(CONTRASTS, one.web_coef.tolist())),
                'stage2': {}, 'errors': []}
    t = web_answers(ctx.df, item, targets(e, ctx.last_month))
    if t.empty:
        return [], settings
    month_fold = tune.month_folds(d)
    target_months = sorted(t.YYYYMM.unique())
    state = state_table(tuple(target_months))
    base = one.log_odds(t)
    rows = []
    for fs in SPECS[spec]:
        for option in names.STAGE_TWO[fs]:
            if option == 'smooth':
                options = [(f'smooth-{names.alpha_name(a)}', float(a)) for a in ctx.settings.alphas]
            else:
                options = [(option, option)]
            for label, opt in options:
                name = names.two_stage(fs, label)
                try:
                    level, chosen = stage_two(one.levels, fs, opt, list(ctx.settings.stage2_alphas), month_fold, state)
                except ValueError as err:
                    settings['errors'].append(f'{name}: {err}')
                    continue
                if chosen is not None:
                    settings['stage2'][name] = {c: float(a) for c, a in chosen.items()}
                P = probabilities(level.loc[t.YYYYMM].to_numpy() + base)
                s = weighted_shares(P, t.WT, t.YYYYMM)
                rows += [{'candidate': name, 'data_end': e, 'month': m, 'item': item, **r.to_dict()}
                         for m, r in s.iterrows()]
    return rows, settings
