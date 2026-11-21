import json

import numpy as np
import pandas as pd
import pytest

from realtime_inputs import TINY, random_raw
from sdt.data.clean import OUTCOMES
from sdt.experiment import sha256
from sdt.realtime import run as realtime
from sdt.realtime.candidates import Settings, table
from sdt.realtime.months import between, data_ends, targets
from sdt.realtime.run import (Context, analyse, assemble, base_probabilities, changed_interviews, check_answers,
                              check_reproduction, execute_all, twin_inputs, unit_results)


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
    assert p['subperiods']['unseen']['df'] == 2 and 'outcome' in p
    assert all('outcome' not in q for q in p['subperiods'].values())            # no decision rule
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
    n = execute_all(ctx, chunks, 1, 'abc')
    forecasts, settings, by_kind, errors = assemble(unit_results(chunks))
    return ctx, n, forecasts, settings, by_kind, errors


def test_every_candidate_forecasts_only_months_to_july_2024(fitted):
    ctx, n, forecasts, settings, by_kind, errors = fitted
    assert errors == {} and set(by_kind) == {'gb', 'gb_tuning', 'ridge', 'smoothing', 'twins', 'two_stage'}
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
    ctx, _, forecasts, *_ = fitted
    chunks = tmp_path / 'chunks'
    execute_all(ctx, chunks, 2, 'abc')
    results = unit_results(chunks)
    assert {r['commit'] for r in results} == {'abc'}
    pd.testing.assert_frame_equal(assemble(results)[0], forecasts, check_exact=True)


def test_run_writes_every_output(fitted, tmp_path, monkeypatch):
    ctx, *_ = fitted
    monkeypatch.setattr(realtime, 'load_context', lambda *a, **k: (ctx, {'data_csv_sha256': 'x'}))
    realtime.run('check', None, out_dir=tmp_path, last_month=202407, settings=ctx.settings, check_previous=False)
    (out,) = tmp_path.iterdir()
    assert {p.name for p in out.iterdir()} == {'chunks', 'forecasts.parquet', 'actual.csv', 'losses.csv',
                                               'candidates.csv', 'settings.json', 'picks.csv', 'differences.csv',
                                               'primary.json', 'secondary.json', 'curve_draws.csv', 'run.json'}
    meta = json.load(open(out / 'run.json'))
    assert meta['units'] == len(json.load(open(out / 'settings.json'))) and meta['units_with_errors'] == {}
    assert not list((out / 'chunks').iterdir())
    with pytest.raises(RuntimeError, match='complete'):
        realtime.run(out.name, None, out_dir=tmp_path, last_month=202407, settings=ctx.settings, resume=True,
                     check_previous=False)


def test_failed_fits_leave_their_candidates_without_a_fit(to_july_2024, monkeypatch):
    from sdt.realtime import twins, two_stage

    def no_maximum(*args, **kwargs):
        raise RuntimeError('calibration did not converge')
    df = to_july_2024
    ctx = Context(df, twin_inputs(random_raw(df), df, 202407), 202407, Settings(**TINY))
    monkeypatch.setattr(twins, 'fit_calibration', no_maximum)
    rows, fits = twins.fit_forecast(ctx, 202405)
    assert rows == [] and len(fits['errors']) == 2 * 4 * 2 * 5               # models x input sets x k x items

    def not_converged(*args, **kwargs):
        raise two_stage.NotConverged('stage 1 converged at no C of the grid')
    monkeypatch.setattr(two_stage, 'StageOne', not_converged)
    rows, settings = two_stage.fit_forecast(ctx, 202405, 'D', 'PAGO')
    assert rows == [] and settings['errors'] == ['stage 1: stage 1 converged at no C of the grid']


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
    with pytest.raises(RuntimeError, match='missing probabilities'):
        twin_inputs(raw.assign(p2=raw.p2.where(raw.index != 0)), to_july_2024, 202407)
    check_answers(to_july_2024, 202407)
    with pytest.raises(RuntimeError, match=r'DUR has no web answers in \[202406\]'):
        check_answers(to_july_2024.assign(DUR=to_july_2024.DUR.where(to_july_2024.YYYYMM != 202406)), 202407)
    json.dump({**meta, 'option_probs_sha256': '0' * 64}, open(folder / 'run.json', 'w'))
    with pytest.raises(RuntimeError, match='differs from the file its run wrote'):
        base_probabilities(folder)


def test_reproduction_check_allows_only_changed_interviews_to_differ(to_july_2024):
    previous = random_raw(to_july_2024)
    r = check_reproduction(previous.assign(p1=previous.p1 + 1e-6), previous, set(), 'previous')
    assert r['max_abs_difference'] == pytest.approx(1e-6) and r['prompts_compared'] == len(previous)
    with pytest.raises(RuntimeError, match='differs from previous'):
        check_reproduction(previous.assign(p1=previous.p1 + 1e-3), previous, set(), 'previous')
    one = previous.CASEID.iloc[0]
    moved = previous.assign(p1=previous.p1.where(previous.CASEID != one, 0.5))
    r = check_reproduction(moved, previous, {one}, 'previous')
    assert r['max_abs_difference'] == 0 and r['interviews_with_changed_inputs'] == [int(one)]
    with pytest.raises(RuntimeError, match='missing probabilities on different prompts'):
        check_reproduction(previous.assign(p1=previous.p1.where(previous.CASEID != one)), previous, set(), 'previous')


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


def test_units_mix_their_kinds_evenly():
    first, second = realtime.units(202609, Settings())
    assert len(first) == 600 + 290 + 1740 + 58 and len(set(first)) == len(first) and len(second) == 1740
    kinds = pd.Series([u[0] for u in first])
    for part in np.array_split(kinds, 10):                           # every tenth of the queue has every heavy kind
        assert {'gb_tuning', 'two_stage', 'ridge'} <= set(part)
    assert {u[0] for u in second} == {'gb'}
