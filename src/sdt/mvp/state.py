"""State variables for the MVP: the economy and the presidency as they stood when each interview window opened.

    python -m sdt.mvp.state     # writes data/processed/mvp_state.csv for inspection

For each survey month and each series in data/external/alfred/:

1. The cutoff is the day before the interview window opens.
2. Keep the rows with `realtime_start` on or before the cutoff.
3. Keep one version per observation: the row with the latest `realtime_start`.
4. Drop observations whose kept value is blank.
5. The latest remaining observation gives the current value.
6. Earlier values come from the same rows.

Values are rounded to the precision the agency publishes, and changes are computed from the rounded values. The
president's party and the election winner's party follow the same cutoff: an event dated on or before it counts.
"""
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from sdt.paths import DATA, PROCESSED

ALFRED = DATA / 'external' / 'alfred'
WINDOWS = DATA / 'external' / 'sca' / 'interview_windows.csv'
OUT_PATH = PROCESSED / 'mvp_state.csv'
PULLED = pd.Timestamp('2026-10-26')     # ALFRED pull date: a later cutoff could miss versions published after it
YEAR = pd.DateOffset(years=1)

# Party from each date on: inaugurations, and the day the AP called the presidential election
PRESIDENT = [('2017-01-20', 'R'), ('2021-01-20', 'D'), ('2025-01-20', 'R')]
WINNER = [('2016-11-09', 'R'), ('2020-11-07', 'D'), ('2024-11-06', 'R')]

MACRO = ['cpi', 'cpi_d1y', 'unrate', 'unrate_d1y', 'gdp', 'gdp_d1y', 'effr', 'effr_d1y']
POLITICS = ['pres_R', 'winner_R']
VARIABLES = MACRO + POLITICS


def round_half_up(x, digits):
    return float(Decimal(repr(float(x))).quantize(Decimal(10) ** -digits, rounding=ROUND_HALF_UP))


def load(series):
    return pd.read_csv(ALFRED / f'{series}.csv', parse_dates=['date', 'realtime_start'])


def snapshot(versions, cutoff):
    """The series as published at the end of the cutoff day: one value per observation, blanks dropped (steps 2-4)."""
    kept = versions[versions.realtime_start <= cutoff].sort_values('realtime_start').groupby('date').tail(1)
    return kept.dropna(subset=['value']).set_index('date').value.sort_index()


def at(snap, date):
    if date not in snap.index:
        raise KeyError(f'{date:%Y-%m-%d} is missing or blank in the snapshot')
    return snap[date]


def yoy(snap, t):
    """Percent change from a year earlier, rounded to 0.1."""
    return round_half_up(100 * (at(snap, t) / at(snap, t - YEAR) - 1), 1)


def growth(snap):
    """CPI and GDP: percent change from a year earlier, and its change from a year earlier."""
    t = snap.index[-1]
    now = yoy(snap, t)
    return t, now, round_half_up(now - yoy(snap, t - YEAR), 1)


def level(snap, digits):
    """Unemployment and fed funds: the latest value, and its change from the latest value on or before the same date a
    year earlier."""
    t = snap.index[-1]
    now = round_half_up(snap.iloc[-1], digits)
    before = snap[snap.index <= t - YEAR]
    if before.empty:
        raise KeyError(f'no value on or before {t - YEAR:%Y-%m-%d}')
    return t, now, round_half_up(now - round_half_up(before.iloc[-1], digits), digits)


def party(calendar, cutoff):
    started = [p for d, p in calendar if pd.Timestamp(d) <= cutoff]
    if not started:
        raise ValueError(f'calendar starts after {cutoff:%Y-%m-%d}')
    return int(started[-1] == 'R')


def build(months):
    """One row per survey month: the cutoff, the observation each series' current value comes from, and VARIABLES."""
    win = pd.read_csv(WINDOWS, parse_dates=['interview_start']).set_index('YYYYMM')
    series = {s: load(s) for s in ['CPIAUCNS', 'UNRATE', 'GDPC1', 'DFF']}
    rows = []
    for m in sorted(months):
        cutoff = win.interview_start[m] - pd.Timedelta(days=1)
        if cutoff > PULLED:
            raise ValueError(f'{m}: cutoff {cutoff:%Y-%m-%d} is after the ALFRED pull')
        snap = {s: snapshot(v, cutoff) for s, v in series.items()}
        cpi_obs, cpi, cpi_d1y = growth(snap['CPIAUCNS'])
        unrate_obs, unrate, unrate_d1y = level(snap['UNRATE'], 1)
        gdp_obs, gdp, gdp_d1y = growth(snap['GDPC1'])
        effr_obs, effr, effr_d1y = level(snap['DFF'], 2)
        rows.append(dict(YYYYMM=m, cutoff=cutoff, cpi_obs=cpi_obs, unrate_obs=unrate_obs, gdp_obs=gdp_obs,
                         effr_obs=effr_obs, cpi=cpi, cpi_d1y=cpi_d1y, unrate=unrate, unrate_d1y=unrate_d1y, gdp=gdp,
                         gdp_d1y=gdp_d1y, effr=effr, effr_d1y=effr_d1y, pres_R=party(PRESIDENT, cutoff),
                         winner_R=party(WINNER, cutoff)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    from sdt.mvp.clean import OUT_PATH as CLEAN_PATH
    months = pd.read_parquet(CLEAN_PATH, columns=['YYYYMM']).YYYYMM.unique()
    state = build(months)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    state.to_csv(OUT_PATH, index=False, date_format='%Y-%m-%d')
    print(f'{len(state)} months, {state.YYYYMM.min()}-{state.YYYYMM.max()} -> {OUT_PATH.relative_to(OUT_PATH.parents[2])}')
