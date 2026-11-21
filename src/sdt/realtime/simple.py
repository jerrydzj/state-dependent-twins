"""The simple-model candidates of the realtime-benchmark plan that smooth shares or fit respondents in one stage
(section 6, candidates 3 to 5). The two-stage ridge is in two_stage.py.

- **Exponential smoothing** (candidate 3): for each item and answer, with s_m its weighted share among month m's web
  interviews, the level follows l_m = alpha s_m + (1 - alpha) l_(m-1) from l_(April 2024) = s_(April 2024). Every
  month a fit at data end e forecasts gets the level at e.
- **One-stage ridge logit** (candidate 4): models.Ridge on the input set's logit encoding (features.ENCODERS), fitted
  to every interview from February 2017 to e with weights WT x 0.5^(a / half-life), where a is the interview's age in
  months at e. Its penalty is tuned by tune.tune_ridge on the same weights.
- **Gradient boosting** (candidate 5): models.gb on the input set's boosting encoding with the same weights. Its
  settings are tuned by tune.tune on the same weights at the data ends of months.tuning_end, and refitted with the
  latest tuned settings at the data ends in between.

Each fit predicts the web interviews of every month it forecasts, and its predicted shares are their weighted mean.
"""
import numpy as np

from sdt import tune
from sdt.data.clean import CLASSES
from sdt.features import ENCODERS, item_data
from sdt.models import Ridge, gb
from sdt.realtime import candidates as names
from sdt.realtime.loss import ITEMS, web_answers, weighted_shares
from sdt.realtime.months import FIRST_WEB, between, index, targets


def decayed(df, e, half_life):
    """`df` with WT multiplied by 0.5^(a / half_life), a being each interview's age in months at data end e."""
    if half_life is None:
        return df
    return df.assign(WT=df.WT * 0.5 ** ((index(e) - index(df.YYYYMM)) / half_life))


def training(ctx, e, half_life=None):
    """Every interview from February 2017 to data end e, weighted for the half-life."""
    return decayed(ctx.df[ctx.df.YYYYMM <= e], e, half_life)


def forecast_rows(name, e, item, P, d):
    """Forecast rows of candidate `name` from probabilities P (columns CLASSES) for the target interviews d."""
    s = weighted_shares(P, d.WT, d.YYYYMM)
    return [{'candidate': name, 'data_end': e, 'month': m, 'item': item, **r.to_dict()} for m, r in s.iterrows()]


def smoothing(ctx, e):
    """Forecast rows of every smoothing candidate fitted at data end e."""
    web_months = between(FIRST_WEB, e)
    forecast = targets(e, ctx.last_month)
    rows = []
    for item in ITEMS:
        d = web_answers(ctx.df, item, web_months)
        Y = (d[item].astype(str).to_numpy()[:, None] == np.array(CLASSES)).astype(float)
        s = weighted_shares(Y, d.WT, d.YYYYMM).reindex(web_months)
        if s.isna().any(axis=None):
            raise ValueError(f'{item}: a web month up to {e} has no answers')
        for alpha in ctx.settings.alphas:
            level = s.iloc[0].to_numpy()
            for m in web_months[1:]:
                level = alpha * s.loc[m].to_numpy() + (1 - alpha) * level
            rows += [{'candidate': names.smoothing(alpha), 'data_end': e, 'month': m, 'item': item,
                      **dict(zip(CLASSES, level))} for m in forecast]
    return rows, {}


def predict(model, X):
    P = model.predict_proba(X)
    return P[:, [list(model.classes_).index(c) for c in CLASSES]]


def ridge(ctx, e, fs, half_life, item):
    """Forecast rows of the one-stage ridge logit on feature set `fs` for one item, and its tuned penalty."""
    encode = ENCODERS[fs]['mnl']
    train = training(ctx, e, half_life)
    C, cv = tune.tune_ridge(train, item, encode, list(ctx.settings.ridge_C))
    X, y, w = item_data(train, item, encode)
    model = Ridge(C).fit(X, y, sample_weight=w)
    d = web_answers(ctx.df, item, targets(e, ctx.last_month))
    rows = forecast_rows(names.ridge(fs, half_life), e, item, predict(model, encode(d)), d) if len(d) else []
    return rows, {'C': None if np.isinf(C) else C, 'cv_log_loss': [float(x) for x in cv]}


def gb_tuning(ctx, e, fs, half_life, item):
    """Boosting's tuned settings on feature set `fs` for one item at tuning data end e."""
    best, curves = tune.tune(training(ctx, e, half_life), item, ENCODERS[fs]['gb'], list(ctx.settings.gb_grid),
                             ctx.settings.gb_max_trees)
    grid = curves.min(axis=1).rename('val_log_loss').to_frame().assign(best_trees=curves.idxmin(axis=1))
    return {'chosen': {k: int(v) for k, v in best.items()}, 'grid': grid.reset_index().to_dict('records')}


def gb_fit(ctx, e, fs, half_life, item, setting):
    """Forecast rows of boosting on feature set `fs` for one item, refitted at data end e with `setting`."""
    encode = ENCODERS[fs]['gb']
    X, y, w = item_data(training(ctx, e, half_life), item, encode)
    model = gb(**setting).fit(X, y, sample_weight=w)
    d = web_answers(ctx.df, item, targets(e, ctx.last_month))
    return forecast_rows(names.boosting(fs, half_life), e, item, predict(model, encode(d)), d) if len(d) else []
