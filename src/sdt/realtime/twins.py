"""The 32 twin candidates of the realtime-benchmark plan (section 6, candidates 1 and 2).

At data end e, for each input set, item and k:

1. The calibration months are the latest k web months up to e, or all web months up to e if k is None or exceeds
   their number.
2. DUR's middle share is its weighted share among the web interviews of the calibration months that answered DUR.
3. The zero-shot twin's class probabilities come from the twin run's raw option probabilities (calibration.class_probs)
   with that middle share.
4. The shifted twin adds a constant to the log-odds of fav and another to those of unf, both against mid; the scaled
   twin also multiplies the log-odds by a factor. Both are fitted by weighted maximum likelihood to the answers of the
   calibration months' web interviews (calibration.fit_calibration).
5. Each fitted twin predicts the web interviews of every month the fit forecasts.

A calibration with no maximum (an item with no fav or no unf answer in the calibration months), or one that does not
converge, is left out and recorded under `errors`, so the candidate has no fit at e.
"""
import numpy as np

from sdt.data.clean import CLASSES
from sdt.features import FEATURE_SETS
from sdt.realtime import candidates as names
from sdt.realtime.loss import ITEMS, web_answers, weighted_shares
from sdt.realtime.months import FIRST_WEB, between, targets
from sdt.twin.calibration import apply_calibration, class_probs, fit_calibration

CALIBRATIONS = {'shift': False, 'scale': True}      # candidate model -> whether the scale is fitted


def fit_forecast(ctx, e):
    """Forecast rows (candidate, data_end, month, item and the predicted shares) of every twin candidate fitted at
    data end e, and the fitted a and b of each."""
    web_months = between(FIRST_WEB, e)
    forecast = targets(e, ctx.last_month)
    rows, fits, errors = [], {}, []
    for item in ITEMS:
        d = web_answers(ctx.df, item, web_months + forecast)
        y, w, month = d[item].astype(str).to_numpy(), d.WT.to_numpy(float), d.YYYYMM.to_numpy()
        target = np.isin(month, forecast)
        for fs in FEATURE_SETS:
            raw = ctx.raw[fs, item].loc[d.CASEID].set_axis(d.index)
            for k in ctx.settings.ks:
                fit = np.isin(month, web_months if k is None else web_months[-k:])
                mid = float(np.average(y[fit] == 'mid', weights=w[fit])) if item == 'DUR' else None
                P = class_probs(raw, item, mid)
                for model, scale in CALIBRATIONS.items():
                    name = names.twin(model, fs, k)
                    try:
                        c = fit_calibration(P[fit], y[fit], w[fit], scale)
                    except (ValueError, RuntimeError) as err:
                        errors.append(f'{name}, {item}: {err}')
                        continue
                    fits.setdefault(name, {})[item] = {'a': c['a'], 'fav': c['fav'], 'unf': c['unf'],
                                                       **({'mid_share': mid} if mid is not None else {})}
                    if target.any():
                        Q = apply_calibration(P[target], c)[CLASSES]
                        s = weighted_shares(Q, w[target], month[target])
                        rows += [{'candidate': name, 'data_end': e, 'month': m, 'item': item, **r.to_dict()}
                                 for m, r in s.iterrows()]
    return rows, {**fits, 'errors': errors}
