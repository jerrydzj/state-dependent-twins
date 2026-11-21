import numpy as np
import pandas as pd
import pytest

from sdt.realtime.candidates import Settings
from sdt.realtime.simple import decayed, smoothing


def test_decay_halves_the_weight_every_half_life():
    df = pd.DataFrame({'YYYYMM': [202407, 202307, 202207], 'WT': [2.0, 2.0, 2.0]})
    assert decayed(df, 202407, 12).WT.tolist() == pytest.approx([2.0, 1.0, 0.5])
    assert decayed(df, 202407, None) is df


def test_smoothing_starts_from_april_2024_and_forecasts_the_level_at_the_data_end():
    months = [202404, 202405, 202406]
    df = pd.DataFrame({'YYYYMM': np.repeat(months, 2), 'web': 1, 'WT': 1.0,
                       **{item: ['fav', 'unf', 'fav', 'fav', 'unf', 'unf'] for item in
                          ['PAGO', 'PEXP', 'BUS12', 'BUS5', 'DUR']}})
    ctx = type('Ctx', (), {'df': df, 'last_month': 202408, 'settings': Settings(alphas=(1.0, 0.5))})
    rows, _ = smoothing(ctx, 202406)
    by = {(r['candidate'], r['month'], r['item']): r for r in rows}
    fav = [0.5, 1.0, 0.0]                                                # each month's share of fav
    assert by['smooth/-/a1.0', 202407, 'PAGO']['fav'] == pytest.approx(fav[2])          # persistence
    half = 0.5 * fav[2] + 0.5 * (0.5 * fav[1] + 0.5 * fav[0])
    assert by['smooth/-/a0.5', 202408, 'DUR']['fav'] == pytest.approx(half)
    assert {m for (_, m, _) in by} == {202407, 202408}
