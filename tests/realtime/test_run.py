import json

import numpy as np
import pandas as pd
import pytest

from realtime_inputs import TINY, random_raw
from sdt.data.clean import OUTCOMES
from sdt.experiment import sha256
from sdt.paths import RAW_CSV
from sdt.realtime import run as realtime
from sdt.realtime.candidates import Settings, table
from sdt.realtime.months import between, data_ends, targets
from sdt.realtime.run import (Context, analyse, assemble, base_probabilities, changed_interviews, check_reproduction,
                              execute_all, twin_inputs)
from sdt.twin.prompts import RAW


# ------------------------------------------------------------------------------- the analysis over the plan's windows

@pytest.fixture(scope='module')
def synthetic():
    """Synthetic web interviews for every month from April 2024 to September 2026, and random forecasts of every
    candidate at every data end."""
    rng = np.random.default_rng(0)
    months = between(202404, 202609)
    n = 20
    df = pd.DataFrame({'CASEID': np.arange(len(months) * n), 'YYYYMM': np.repeat(months, n), 'web': 1,
                       'WT': rng.uniform(0.5, 1.5, len(months) * n)})
    for item in OUTCOMES:
        df[item] = rng.choice(['fav', 'mid', 'unf'], len(df))
    cands = table()
    keys = pd.DataFrame([(e, m) for e in data_ends() for m in targets(e)], columns=['data_end', 'month'])
    keys = keys.merge(cands[['candidate']], how='cross').merge(pd.DataFrame({'item': list(OUTCOMES)}), how='cross')
    shares = rng.dirichlet([3, 2, 3], len(keys))
    forecasts = keys.assign(fav=shares[:, 0], mid=shares[:, 1], unf=shares[:, 2])
    return df, forecasts


def test_analysis_follows_the_plan_s_windows(synthetic):
    df, forecasts = synthetic
    res = analyse(forecasts, df, 202609, Settings())
    p, d = res['primary'], res['differences']
    assert (p['T'], p['m'], p['df'], p['first_month'], p['last_month']) == (26, 2, 4, 202408, 202609)
    at2 = d[d.horizon == 2]
    assert p['mean'] == pytest.approx(at2.d.mean()) and np.allclose(d.d, d.simple_loss - d.twin_loss)
    assert p['avoidable_loss'] == pytest.approx(at2.simple_loss.mean() - res['actual'].groupby('month').floor.mean()
                                                .loc[at2.month].mean())
    assert p['as_share_of_avoidable_loss']['mean'] == pytest.approx(p['mean'] / p['avoidable_loss'])
    assert (p['subperiods']['known']['T'], p['subperiods']['unseen']['T']) == (19, 7)
    assert p['subperiods']['unseen']['df'] == 2
    s = res['secondary']
    assert s['months'] == between(202506, 202609) and list(s['curve']) == list(range(1, 13))
    assert s['curve'][5] == pytest.approx(d[(d.horizon == 5) & d.month.isin(s['months'])].d.mean())
    assert len(res['curve_draws']) == 1000
    picks = res['picks']
    assert set(picks.data_end) == set(between(202406, 202608)) and picks.candidate.notna().all()
    assert (picks[picks.cls == 'twin'].candidate.str.split('/').str[0].isin(['shift', 'scale'])).all()


# --------------------------------------------------------------------------------- every unit on real data to July 2024

@pytest.fixture(scope='module')
def fitted(to_july_2024, tmp_path_factory):
    df = to_july_2024
    ctx = Context(df, twin_inputs(random_raw(df), df, 202407), 202407, Settings(**TINY))
    chunks = tmp_path_factory.mktemp('one-worker') / 'chunks'
    n = execute_all(ctx, chunks, 1, chunks / 'log.jsonl')
    return ctx, n, *assemble(chunks)


def test_every_candidate_forecasts_only_months_to_july_2024(fitted):
    ctx, n, forecasts, settings = fitted
    cands = table(ctx.settings)
    assert n == len(settings) and forecasts.month.max() == 202407 and (forecasts.month > forecasts.data_end).all()
    complete = forecasts.groupby(['candidate', 'data_end']).item.nunique()
    assert set(complete.index.get_level_values(0)) == set(cands.candidate) and (complete == 5).all()
    assert set(complete.index.get_level_values(1)) == {202404, 202405, 202406}
    assert all(s['tuned_at'] == 202404 for u, s in settings.items() if u.startswith('gb--'))
    res = analyse(forecasts, ctx.df, 202407, ctx.settings)
    assert res['primary'] is None and res['secondary'] is None and set(res['picks'].data_end) == {202406}
    assert len(res['differences']) == 0


def test_units_give_the_same_forecasts_on_several_workers(fitted, tmp_path):
    ctx, _, forecasts, _ = fitted
    chunks = tmp_path / 'chunks'
    execute_all(ctx, chunks, 2, chunks / 'log.jsonl')
    pd.testing.assert_frame_equal(assemble(chunks)[0], forecasts)


# -------------------------------------------------------------------------------------------------------------- checks

def twin_run(folder, raw):
    folder.mkdir()
    raw.to_parquet(folder / 'option_probs.parquet', index=False)
    json.dump({'option_probs_sha256': sha256(folder / 'option_probs.parquet'), 'commit': 'abc',
               'data_csv_sha256': 'x'}, open(folder / 'run.json', 'w'))
    return folder


def test_inputs_must_cover_every_web_interview(to_july_2024, tmp_path):
    raw = random_raw(to_july_2024)
    folder = twin_run(tmp_path / 'twins', raw)
    loaded, meta = base_probabilities(folder)
    assert len(loaded) == len(raw) and meta['commit'] == 'abc'
    with pytest.raises(RuntimeError, match='lacks web interviews'):
        twin_inputs(raw.iloc[1:], to_july_2024, 202407)
    json.dump({**meta, 'option_probs_sha256': '0' * 64}, open(folder / 'run.json', 'w'))
    with pytest.raises(RuntimeError, match='differs from the file its run wrote'):
        base_probabilities(folder)


def test_reproduction_check_allows_only_changed_interviews_to_differ(to_july_2024, tmp_path, monkeypatch):
    monkeypatch.setattr(realtime, 'PREVIOUS_DATA_SHA256', sha256(RAW_CSV) if RAW_CSV.exists() else 'none')
    if not RAW_CSV.exists():
        pytest.skip('data/data.csv not downloaded')
    raw = random_raw(to_july_2024)
    previous = twin_run(tmp_path / 'previous', raw)
    r = check_reproduction(raw.assign(p1=raw.p1 + 1e-6), to_july_2024, previous)
    assert r['max_abs_difference'] == pytest.approx(1e-6) and r['prompts_compared'] == len(raw)
    with pytest.raises(RuntimeError, match='differs from previous'):
        check_reproduction(raw.assign(p1=raw.p1 + 1e-3), to_july_2024, previous)


def test_changed_interviews_count_a_change_to_or_from_missing(to_july_2024):
    old = to_july_2024.head(4).copy()
    new = old.copy()
    ids = old.CASEID.tolist()
    new.loc[new.index[0], 'NUMADT'] = pd.NA                              # nullable Int8: a value becomes missing
    new.loc[new.index[1], 'POLAFF'] = np.nan                             # categorical
    new.loc[new.index[2], 'AGE'] = 30 if old.AGE.iloc[2] != 30 else 31
    assert changed_interviews(old, new) == set(ids[:3])
    assert changed_interviews(new, old) == set(ids[:3])                 # and missing becomes a value
    assert changed_interviews(old, new.iloc[:3]) == set(ids[:3])        # an interview in only one table does not count
