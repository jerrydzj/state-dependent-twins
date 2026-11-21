"""Steps 4 and 5 of the procedure (section 5 of the realtime-benchmark plan): the picks and the differences d_t(h).

1. **Selection.** At each data end o, a candidate's score is its mean L(c, v - 2, v) over the validation months v of o
   (months.validation_months). A candidate is eligible at o if it has every L its score needs and a fit at o, which
   means a loss for every month the fit at o forecasts. Each class picks its eligible candidate with the lowest score,
   and exact ties go to the candidate listed first (candidates.table).
2. **Comparison.** For target month t and horizon h, with o = t - h, d_t(h) = L(simple pick, o, t) - L(twin pick, o, t).
   A positive d_t(h) means the twin pick is closer to the actual shares.
"""
import numpy as np
import pandas as pd

from sdt.realtime.months import add, targets, validation_months

CLASSES = ['twin', 'simple']


def picks(L, candidates, pick_ends, last_month):
    """One row per data end of `pick_ends` and class: the pick, its score, the number of validation months and of
    eligible candidates. `L` has columns candidate, data_end, month and loss; `candidates` is candidates.table()."""
    loss = L.set_index(['candidate', 'data_end', 'month']).loss.to_dict()
    rows = []
    for o in pick_ends:
        V = validation_months(o)
        forecast = targets(o, last_month)
        for cls in CLASSES:
            best, eligible = None, 0
            for c in candidates[candidates.cls == cls].sort_values('order').candidate:
                needed = [loss.get((c, add(v, -2), v)) for v in V]
                if not V or any(x is None for x in needed) or any((c, o, t) not in loss for t in forecast):
                    continue
                eligible += 1
                score = float(np.mean(needed))
                if best is None or score < best[1]:
                    best = (c, score)
            rows.append({'data_end': o, 'cls': cls, 'candidate': best[0] if best else None,
                         'score': best[1] if best else np.nan, 'validation_months': len(V), 'eligible': eligible})
    return pd.DataFrame(rows)


def differences(L, pick_table, pairs):
    """d_t(h) for each (t, h) in `pairs`: one row per pair with the data end o = t - h, both picks, their losses on
    month t and d."""
    loss = L.set_index(['candidate', 'data_end', 'month']).loss
    pick = pick_table.set_index(['data_end', 'cls']).candidate
    rows = []
    for t, h in pairs:
        o = add(t, -h)
        simple, twin = pick.get((o, 'simple')), pick.get((o, 'twin'))
        if simple is None or twin is None:
            raise ValueError(f'no pick of both classes at data end {o}, needed for month {t} at horizon {h}')
        ls, lt = loss[simple, o, t], loss[twin, o, t]
        rows.append({'month': t, 'horizon': h, 'data_end': o, 'simple': simple, 'twin': twin, 'simple_loss': ls,
                     'twin_loss': lt, 'd': ls - lt})
    return pd.DataFrame(rows)
