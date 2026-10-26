import pandas as pd
import pytest

from sdt.mvp.state import PRESIDENT, VARIABLES, WINNER, build, level, load, party, round_half_up, snapshot

T = pd.Timestamp


def versions(rows):
    """ALFRED-style rows: (date, value, realtime_start); None is a blank value."""
    return pd.DataFrame(rows, columns=['date', 'value', 'realtime_start']).astype(
        {'date': 'datetime64[ns]', 'realtime_start': 'datetime64[ns]', 'value': float})


def test_snapshot_keeps_the_version_current_at_the_cutoff():
    v = versions([('2024-01-01', 1.0, '2024-02-01'), ('2024-01-01', 2.0, '2024-03-01'),
                  ('2024-02-01', 5.0, '2024-03-10')])
    assert snapshot(v, T('2024-02-15')).to_dict() == {T('2024-01-01'): 1.0}
    assert snapshot(v, T('2024-03-01')).to_dict() == {T('2024-01-01'): 2.0}              # published on the cutoff day
    assert snapshot(v, T('2024-03-10')).to_dict() == {T('2024-01-01'): 2.0, T('2024-02-01'): 5.0}


def test_snapshot_drops_blank_observations():
    v = versions([('2025-09-01', 4.4, '2025-11-20'), ('2025-10-01', None, '2025-12-16'),
                  ('2025-11-01', 4.6, '2025-12-16')])
    assert snapshot(v, T('2025-12-16')).index.tolist() == [T('2025-09-01'), T('2025-11-01')]


def test_blank_year_earlier_value_raises():
    v = versions([('2024-10-01', None, '2024-11-01'), ('2025-10-01', 4.0, '2025-11-01')])
    with pytest.raises(KeyError):
        level(snapshot(v, T('2025-11-01')), 1)


def test_round_half_up():
    assert round_half_up(0.25, 1) == 0.3 and round_half_up(-0.25, 1) == -0.3 and round_half_up(4.1 - 3.7, 1) == 0.4


@pytest.mark.parametrize('calendar, cutoff, expected', [
    (PRESIDENT, '2025-01-19', 0), (PRESIDENT, '2025-01-20', 1),       # an event counts from its own date
    (WINNER, '2024-11-05', 0), (WINNER, '2024-11-06', 1),
    (WINNER, '2020-11-06', 1), (WINNER, '2020-11-07', 0),
])
def test_party_calendar(calendar, cutoff, expected):
    assert party(calendar, T(cutoff)) == expected


def test_party_calendar_does_not_extrapolate():
    with pytest.raises(ValueError):
        party(PRESIDENT, T('2017-01-19'))


@pytest.fixture(scope='module')
def state():
    months = pd.period_range('2017-02', '2026-02', freq='M')
    return build(months.year * 100 + months.month).set_index('YYYYMM')


def test_one_complete_row_per_month(state):
    assert len(state) == 109 and state.index.is_unique and state.notna().all(axis=None)
    assert state.index.min() == 201702 and state.index.max() == 202602


@pytest.mark.parametrize('series, col', [('CPIAUCNS', 'cpi_obs'), ('UNRATE', 'unrate_obs'), ('GDPC1', 'gdp_obs'),
                                         ('DFF', 'effr_obs')])
def test_current_value_is_latest_published_before_cutoff(state, series, col):
    # Recomputed independently: the latest observation whose first non-blank version is public by the cutoff
    v = load(series).dropna(subset=['value'])
    first = v.groupby('date').realtime_start.min()
    for m, row in state.iterrows():
        assert row[col] == first[first <= row.cutoff].index.max(), m


@pytest.mark.parametrize('month, expected', [
    (202208, {'cpi': 9.1, 'unrate': 3.6}),                             # June 2022 CPI peak
    (202006, {'unrate': 14.7}),                                        # April 2020 as first printed (later 14.8)
    (202502, {'cpi': 2.9, 'unrate': 4.1, 'effr': 4.33, 'pres_R': 1, 'winner_R': 1}),   # cutoff is inauguration day
    (202011, {'winner_R': 1}), (202012, {'winner_R': 0, 'pres_R': 1}), (202102, {'pres_R': 0}),
    (202411, {'winner_R': 0}), (202412, {'winner_R': 1, 'pres_R': 0}),
])
def test_known_values(state, month, expected):
    assert state.loc[month, list(expected)].to_dict() == expected


@pytest.mark.parametrize('month, cpi_obs, unrate_obs', [
    (202511, '2025-08-01', '2025-08-01'),       # September releases delayed by the 2025 shutdown
    (202512, '2025-09-01', '2025-08-01'),
    (202601, '2025-09-01', '2025-09-01'),       # October never published; November out after the cutoff
])
def test_shutdown_months(state, month, cpi_obs, unrate_obs):
    assert state.loc[month, ['cpi_obs', 'unrate_obs']].tolist() == [T(cpi_obs), T(unrate_obs)]


def test_variables_are_rounded_to_published_precision(state):
    for c, digits in {'cpi': 1, 'cpi_d1y': 1, 'unrate': 1, 'unrate_d1y': 1, 'gdp': 1, 'gdp_d1y': 1, 'effr': 2,
                      'effr_d1y': 2}.items():
        assert (state[c].round(digits) == state[c]).all(), c
    assert set(VARIABLES) <= set(state.columns)
