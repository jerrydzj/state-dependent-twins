"""Shares, losses and noise floors of the realtime-benchmark plan (sections 4 and 7).

Every share counts only web interviews. For an item and month:

1. The actual shares are the `WT`-weighted shares of the answers of the month's web respondents who answered the item.
2. A forecast's predicted shares are its predicted probabilities averaged over the same respondents, weighted by `WT`.
3. The item's loss is the squared error of the cumulative shares: (fav - fav^)^2 + (unf - unf^)^2.
4. The month's loss L is the mean of the item losses over the five items. A forecast missing an item has no loss.

The noise floor of a month is the expected loss of a forecast equal to the population's shares. For each item and for
fav and unf, the variance of the weighted share is estimated as sum w_i^2 (y_i - s)^2 / (sum w_i)^2, where y_i is 1 if
respondent i gave that answer and s is the weighted share. The item's floor is the sum over fav and unf, and the
month's floor is the mean over the five items, as for the loss.
"""
import numpy as np
import pandas as pd

from sdt.data.clean import CLASSES, OUTCOMES

ITEMS = list(OUTCOMES)
SHARES = ['fav', 'mid', 'unf']          # columns of predicted and actual shares, in CLASSES order


def web_answers(df, item, months):
    """The web interviews of `months` that answered `item`, in the table's order."""
    return df[(df.web == 1) & df[item].notna() & df.YYYYMM.isin(months)]


def weighted_shares(P, w, month):
    """Weighted mean of the rows of P (columns CLASSES) within each month: one row per month, months ascending."""
    P, w, month = np.asarray(P, float), np.asarray(w, float), np.asarray(month)
    num = pd.DataFrame(P * w[:, None], columns=CLASSES).groupby(month).sum()
    return num.div(pd.Series(w).groupby(month).sum(), axis=0).rename_axis('month')


def actual(df, months):
    """Per month of `months` and item: the actual shares, the number of interviews and the item's noise floor."""
    rows = []
    for item in ITEMS:
        d = web_answers(df, item, months)
        Y = (d[item].astype(str).to_numpy()[:, None] == np.array(CLASSES)).astype(float)
        w, month = d.WT.to_numpy(float), d.YYYYMM.to_numpy()
        s = weighted_shares(Y, w, month)
        for m, g in pd.DataFrame({'m': month}).groupby('m').groups.items():
            i = np.asarray(g)
            floor = sum(np.sum(w[i] ** 2 * (Y[i, k] - s.loc[m].iloc[k]) ** 2) / w[i].sum() ** 2 for k in (0, 2))
            rows.append({'month': m, 'item': item, **s.loc[m].to_dict(), 'interviews': len(i), 'floor': floor})
    return pd.DataFrame(rows).sort_values(['month', 'item'], ignore_index=True)


def losses(forecasts, actual_shares):
    """L for every forecast with all five items: one row per candidate, data end and month, with column `loss`."""
    f = forecasts.merge(actual_shares[['month', 'item', 'fav', 'unf']], on=['month', 'item'], suffixes=('', '_actual'),
                        validate='many_to_one')
    f['item_loss'] = (f.fav - f.fav_actual) ** 2 + (f.unf - f.unf_actual) ** 2
    g = f.groupby(['candidate', 'data_end', 'month']).item_loss.agg(['mean', 'count'])
    return g[g['count'] == len(ITEMS)]['mean'].rename('loss').reset_index()


def floors(actual_shares):
    """The noise floor of each month: the mean of its item floors over the five items."""
    g = actual_shares.groupby('month').floor.agg(['mean', 'count'])
    if (g['count'] != len(ITEMS)).any():
        raise ValueError('every month needs a floor for every item')
    return g['mean'].rename('floor')
