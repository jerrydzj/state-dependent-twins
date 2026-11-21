import pandas as pd
import pytest

from sdt.realtime.months import add, targets
from sdt.realtime.procedure import differences, picks

LAST = 202409


def candidate_table(names):
    return pd.DataFrame([{'order': i, 'candidate': c, 'cls': cls} for i, (c, cls) in enumerate(names)])


def full_losses(loss_of):
    """L for every candidate of `loss_of` (candidate -> function of data end and month) at data ends 202404-202408."""
    rows = [{'candidate': c, 'data_end': e, 'month': m, 'loss': f(e, m)}
            for c, f in loss_of.items() for e in [202404, 202405, 202406, 202407, 202408] for m in targets(e, LAST)]
    return pd.DataFrame(rows)


def test_each_class_picks_its_lowest_score_over_the_validation_months():
    # 'b' is better on the operating point only in July and August 2024, so it wins from the August pick on
    cands = candidate_table([('a', 'twin'), ('b', 'twin'), ('s', 'simple')])
    L = full_losses({'a': lambda e, m: 1.1, 'b': lambda e, m: 0.0 if m >= 202407 else 3.0, 's': lambda e, m: 2.0})
    p = picks(L, cands, [202406, 202407, 202408], LAST).set_index(['data_end', 'cls'])
    assert p.loc[(202406, 'twin'), 'candidate'] == 'a' and p.loc[(202406, 'twin'), 'validation_months'] == 1
    assert p.loc[(202407, 'twin'), 'candidate'] == 'a'               # mean of 3 and 0 is 1.5 > 1.1
    assert p.loc[(202408, 'twin'), 'candidate'] == 'b' and p.loc[(202408, 'twin'), 'score'] == pytest.approx(1.0)
    assert p.loc[(202408, 'simple'), 'candidate'] == 's'


def test_ties_go_to_the_first_listed_and_missing_fits_make_a_candidate_ineligible():
    cands = candidate_table([('late', 'twin'), ('early', 'twin'), ('s', 'simple')])
    L = full_losses({'late': lambda e, m: 1.0, 'early': lambda e, m: 1.0, 's': lambda e, m: 1.0})
    p = picks(L, cands, [202408], LAST).set_index('cls')
    assert p.loc['twin', 'candidate'] == 'late' and p.loc['twin', 'eligible'] == 2
    # Without its fit at 202406, 'late' lacks L(c, 202406, 202408) for the August score
    p = picks(L[~((L.candidate == 'late') & (L.data_end == 202406))], cands, [202408], LAST).set_index('cls')
    assert p.loc['twin', 'candidate'] == 'early' and p.loc['twin', 'eligible'] == 1
    # Without its fit at the pick's own data end, it cannot forecast from there
    p = picks(L[~((L.candidate == 'late') & (L.data_end == 202408))], cands, [202408], LAST).set_index('cls')
    assert p.loc['twin', 'candidate'] == 'early'


def test_differences_compare_both_picks_on_the_target_month():
    cands = candidate_table([('a', 'twin'), ('s', 'simple')])
    L = full_losses({'a': lambda e, m: 0.1 * (m - e), 's': lambda e, m: 0.05 * (m - e) + 0.3})
    p = picks(L, cands, [202406, 202407, 202408], LAST)
    d = differences(L, p, [(202408, 2), (202409, 1)]).set_index('month')
    o = add(202408, -2)
    assert d.loc[202408, 'data_end'] == o and d.loc[202408, 'd'] == pytest.approx(
        (0.05 * (202408 - o) + 0.3) - 0.1 * (202408 - o))
    with pytest.raises(ValueError, match='no pick'):
        differences(L, p, [(202407, 3)])                             # no pick at 202404
