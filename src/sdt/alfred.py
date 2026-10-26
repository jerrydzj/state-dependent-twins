"""Download every published version of the state-variable series from ALFRED, FRED's archive of past releases.

    python -m sdt.alfred        # writes data/external/alfred/<SERIES>.csv

Each row is one value of one observation over the period it was the published value: `date` is the observation
(month, quarter or day), and `realtime_start` to `realtime_end` are the first and last days that value was current.
A value still current today has `realtime_end` 9999-12-31. FRED's missing value "." is written as blank.

The API key is read from the environment variable FRED_API_KEY, or else from `.env` in the repository root.
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from sdt.paths import DATA, ROOT

OUT = DATA / 'external' / 'alfred'
SERIES = ['CPIAUCNS', 'UNRATE', 'GDPC1', 'DFF']
OBS_START = '2014-01-01'            # leaves a year of history before the 12-month changes of February 2017
API = 'https://api.stlouisfed.org/fred/'
MAX_VINTAGES = 1500                 # below the limit of 2,000 versions per request, which counts one more than listed
PAGE = 100_000                      # the API's limit on rows per request


def api_key():
    if os.environ.get('FRED_API_KEY'):
        return os.environ['FRED_API_KEY']
    env = ROOT / '.env'
    pairs = dict(line.strip().split('=', 1) for line in env.read_text().splitlines()
                 if '=' in line and not line.lstrip().startswith('#')) if env.exists() else {}
    if not pairs.get('FRED_API_KEY'):
        raise RuntimeError('set FRED_API_KEY in the environment or in .env')
    return pairs['FRED_API_KEY']


def get(path, **params):
    """One API call. Errors report FRED's message but never the URL, which contains the key."""
    url = API + path + '?' + urllib.parse.urlencode({**params, 'api_key': api_key(), 'file_type': 'json'})
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'{path} {params}: HTTP {e.code}: {json.loads(e.read()).get("error_message")}') from None


def paged(path, field, **params):
    rows, offset = [], 0
    while True:
        page = get(path, **params, limit=PAGE if field == 'observations' else 10_000, offset=offset)
        rows += page[field]
        offset += len(page[field])
        if offset >= page['count'] or not page[field]:
            return rows


def vintage_dates(series):
    return paged('series/vintagedates', 'vintage_dates', series_id=series, realtime_start='1776-07-04')


def observations(series):
    """All versions of every observation from OBS_START, fetched in blocks of at most MAX_VINTAGES versions and joined
    so that each row covers the whole period its value was current."""
    vintages = vintage_dates(series)
    starts = vintages[::MAX_VINTAGES]
    ends = [str((pd.Timestamp(s) - pd.Timedelta(days=1)).date()) for s in starts[1:]] + ['9999-12-31']
    rows = [r for s, e in zip(starts, ends)
            for r in paged('series/observations', 'observations', series_id=series, observation_start=OBS_START,
                           realtime_start=s, realtime_end=e)]
    df = pd.DataFrame(rows)[['date', 'value', 'realtime_start', 'realtime_end']]
    df['value'] = df.value.replace('.', None)
    return join_blocks(df)


def join_blocks(df):
    """Merge rows that a block boundary split: same observation and value, with the second starting the day after the
    first ends."""
    df = df.sort_values(['date', 'realtime_start']).reset_index(drop=True)
    prev_end = pd.to_datetime(df.realtime_end.shift(), errors='coerce')     # 9999-12-31 is out of range, so NaT
    cont = ((df.date == df.date.shift()) & (df.value.fillna('.') == df.value.shift().fillna('.'))
            & (pd.to_datetime(df.realtime_start) == prev_end + pd.Timedelta(days=1)))
    group = (~cont).cumsum()
    return (df.groupby(group).agg(date=('date', 'first'), value=('value', 'first'),
                                  realtime_start=('realtime_start', 'first'), realtime_end=('realtime_end', 'last'))
            .reset_index(drop=True))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SERIES:
        df = observations(s)
        df.to_csv(OUT / f'{s}.csv', index=False)
        print(s, f'{len(df):,} rows, {df.date.nunique():,} observations, {df.realtime_start.nunique():,} versions')


if __name__ == '__main__':
    main()
