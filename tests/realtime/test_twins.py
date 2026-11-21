import numpy as np
import pytest

from realtime_inputs import TINY, random_raw
from sdt.realtime.candidates import Settings
from sdt.realtime.run import Context, twin_inputs
from sdt.realtime.twins import fit_forecast


@pytest.fixture(scope='module')
def ctx(to_july_2024):
    df = to_july_2024
    return Context(df, twin_inputs(random_raw(df), df, 202407), 202407, Settings(**{**TINY, 'ks': (1, 3, 6, None)}))


def test_twins_calibrate_on_the_latest_k_web_months(ctx):
    rows, fits = fit_forecast(ctx, 202405)
    d = ctx.df[(ctx.df.web == 1) & ctx.df.DUR.notna()]
    share = lambda months: np.average(d[d.YYYYMM.isin(months)].DUR == 'mid', weights=d[d.YYYYMM.isin(months)].WT)
    assert fits['shift/D/k1']['DUR']['mid_share'] == pytest.approx(share([202405]))
    assert fits['shift/D/kall']['DUR']['mid_share'] == pytest.approx(share([202404, 202405]))
    assert fits['scale/D+M+P/k1']['PAGO']['a'] != 1 and fits['shift/D+M+P/k1']['PAGO']['a'] == 1


def test_twins_forecast_every_later_month_and_coincide_when_k_exceeds_the_web_months(ctx):
    rows, _ = fit_forecast(ctx, 202405)
    by = {(r['candidate'], r['month'], r['item']): r['fav'] for r in rows}
    assert {m for (_, m, _) in by} == {202406, 202407} and len({c for (c, _, _) in by}) == 32
    for item in ['PAGO', 'DUR']:
        assert by['shift/D+P/k3', 202407, item] == by['shift/D+P/kall', 202407, item] == by['shift/D+P/k6', 202407, item]
        assert by['shift/D+P/k1', 202407, item] != by['shift/D+P/kall', 202407, item]
