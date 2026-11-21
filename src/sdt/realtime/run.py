"""Run the realtime-benchmark procedure and save the results to a new folder experiments/realtime/<date>-<description>/.

    python -m sdt.realtime.run benchmark --twins <twin run folder> --workers 64
    python -m sdt.realtime.run <folder name> --twins <twin run folder> --workers 64 --resume

The run follows sections 5 to 8 of docs/prereg/realtime-benchmark.md. Before fitting, it checks its inputs and stops if
any check fails:

1. The clean table is rebuilt from data.csv, whose SHA-256 the run records, and the twin run used the same data.csv.
2. The twin run's option_probs.parquet has the SHA-256 that its run.json recorded, and it holds base-stage
   probabilities for every web interview from April 2024 to the last month, on every feature set and item.
3. The twin run reproduces the probabilities of the run frozen by prereg/mvp-twin (PREVIOUS_TWINS) within
   REPRODUCTION_TOLERANCE on every interview whose inputs are the same in both data.csv versions (PREVIOUS_DATA).

4. Every item has web answers in every month from April 2024 to the last month.

The work is split into units: per data end, all twins and all smoothing candidates; per data end, feature set,
half-life and item, the one-stage ridge and boosting; per data end, stage-1 spec and item, the two-stage ridge; and per
tuning data end, feature set, half-life and item, boosting's tuning, which runs before the boosting fits. Each unit
writes chunks/<unit>.json with the commit that ran it, so an interrupted run can continue at the same commit. Units run
in parallel on --workers processes of one thread each, and a unit's results do not depend on the number of workers.
A fit that fails in a way the plan foresees (a calibration or stage 1 with no maximum, or one that does not converge)
leaves its candidate without a fit at that data end, and run.json lists it.

When every unit is done, the run writes:

- forecasts.parquet: every forecast's predicted shares (candidate, data end, month, item, fav, mid, unf).
- actual.csv: each month and item's actual shares, web interviews and noise floor (loss.py).
- losses.csv: L(c, e, m) for every candidate, data end and month.
- candidates.csv: the 92 candidates in the plan's order, which breaks ties.
- settings.json: what each unit tuned or fitted (penalties, boosting settings, calibrations).
- picks.csv: each class's pick at every data end, with its score (procedure.py).
- differences.csv: d_t(h) for the primary test and the secondary analysis.
- primary.json: section 7: d-bar, the test, the outcome, the interval in loss units and as a share of the simple
  pick's avoidable loss, and the two subperiods.
- secondary.json and curve_draws.csv: section 8: the curve d-bar(h), rho, slope and breakeven with their intervals,
  and the three summaries on every bootstrap draw.
- run.json: commit, data and twin hashes, the reproduction check, settings, package versions, seconds per kind of
  unit and the units with errors.
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from sdt.data.clean import INPUTS, clean, load_raw
from sdt.experiment import git, in_experiments, require_clean_tree, sha256
from sdt.features import FEATURE_SETS
from sdt.paths import DATA, EXPERIMENTS, RAW_CSV
from sdt.realtime import RUNS, simple, stats, twins, two_stage
from sdt.realtime.candidates import Settings, table
from sdt.realtime.loss import ITEMS, actual, floors, losses
from sdt.realtime.months import (CURVE, FIRST_VALIDATION, FIRST_WEB, HORIZONS, KNOWN, LAST_MONTH, OPERATING_HORIZON,
                                 PRIMARY, UNSEEN, between, data_ends, tuning_end)
from sdt.realtime.procedure import differences, picks
from sdt.twin.prompts import OPTIONS, RAW

PREVIOUS_TWINS = EXPERIMENTS / 'mvp' / '2026-11-05-twin-prereg-eval'
PREVIOUS_DATA = DATA / 'data-2026-10-15.csv'          # the data.csv that PREVIOUS_TWINS used, kept under this name
PREVIOUS_DATA_SHA256 = '5e0a7e1cd2a04e8378e43bc1b4316b4bb5387241bb40744f0c50c6f7cd49ca22'
REPRODUCTION_TOLERANCE = 1e-4
PACKAGES = ['numpy', 'pandas', 'scipy', 'scikit-learn']


@dataclass
class Context:
    """What every unit reads: the clean table up to the last month, the twins' raw probabilities by feature set and
    item (indexed by CASEID), the last month and the settings."""
    df: pd.DataFrame
    raw: dict
    last_month: int
    settings: Settings


# ---------------------------------------------------------------------------------------------------------- checks

def base_probabilities(twin_dir):
    """Check 2's hash: the base-stage rows of a twin run's option_probs.parquet."""
    meta = json.load(open(twin_dir / 'run.json'))
    if sha256(twin_dir / 'option_probs.parquet') != meta['option_probs_sha256']:
        raise RuntimeError(f'{twin_dir.name}/option_probs.parquet differs from the file its run wrote')
    raw = pd.read_parquet(twin_dir / 'option_probs.parquet')
    return raw[raw.stage == 'base'], meta


def twin_inputs(raw, df, last_month):
    """Check 2's coverage: the probabilities of the web interviews from April 2024 to the last month by feature set and
    item. Raises unless every such interview has finite probabilities of every option of every item."""
    web = df[(df.web == 1) & df.YYYYMM.between(FIRST_WEB, last_month)].CASEID
    out = {}
    for (fs, item), g in raw.groupby(['feature_set', 'item']):
        out[fs, item] = g.set_index('CASEID')[RAW]
    for fs in FEATURE_SETS:
        for item in ITEMS:
            if (fs, item) not in out or not web.isin(out[fs, item].index).all():
                raise RuntimeError(f'the twin run lacks web interviews of {fs}, {item}')
            out[fs, item] = out[fs, item].loc[web]
            if not np.isfinite(out[fs, item][RAW[:len(OPTIONS[item])]].to_numpy(float)).all():
                raise RuntimeError(f'the twin run has missing probabilities for {fs}, {item}')
    return out


def check_answers(df, last_month):
    """Raise unless every item has web answers in every month from April 2024 to the last month."""
    for item in ITEMS:
        months = set(df[(df.web == 1) & df[item].notna()].YYYYMM)
        missing = [m for m in between(FIRST_WEB, last_month) if m not in months]
        if missing:
            raise RuntimeError(f'{item} has no web answers in {missing}')


def changed_interviews(old, new):
    """CASEIDs in both clean tables whose month, mode or inputs differ between them. Missing counts as a value, so a
    change between missing and a value is a change."""
    columns = ['YYYYMM', 'web', *INPUTS]
    a, b = old.set_index('CASEID')[columns], new.set_index('CASEID')[columns]
    both = a.index.intersection(b.index)
    a, b = a.loc[both], b.loc[both]
    same = (a == b).fillna(False).astype(bool) | (a.isna() & b.isna())
    return set(both[~same.all(axis=1).to_numpy()])


def check_reproduction(raw, previous_raw, changed, previous_name):
    """Check 3: compare the raw probabilities `raw` with `previous_raw` on every prompt of both, except those of the
    interviews in `changed`. Raises if a probability is missing in one run only, or if any difference exceeds
    REPRODUCTION_TOLERANCE."""
    key = ['CASEID', 'feature_set', 'item']
    both = raw[key + RAW].merge(previous_raw[key + RAW], on=key, suffixes=('', '_previous'), validate='one_to_one')
    both = both[~both.CASEID.isin(changed)]
    new, old = both[RAW].to_numpy(float), both[[f'{c}_previous' for c in RAW]].to_numpy(float)
    if (np.isnan(new) != np.isnan(old)).any():
        raise RuntimeError(f'the twin run and {previous_name} have missing probabilities on different prompts')
    diff = np.abs(np.nan_to_num(new - old))
    worst = float(diff.max()) if len(both) else 0.0
    if worst > REPRODUCTION_TOLERANCE:
        raise RuntimeError(f'the twin run differs from {previous_name} by up to {worst:.2g} on unchanged interviews')
    return {'previous_run': previous_name, 'prompts_compared': len(both), 'max_abs_difference': worst,
            'share_identical': float((diff.max(axis=1) == 0).mean()) if len(both) else 1.0,
            'interviews_with_changed_inputs': sorted(int(c) for c in changed)}


def reproduction(raw, df, previous_twins=PREVIOUS_TWINS, previous_data=PREVIOUS_DATA):
    """Check 3 against PREVIOUS_TWINS, with the interviews whose inputs PREVIOUS_DATA and data.csv give differently."""
    previous_raw, _ = base_probabilities(previous_twins)
    if sha256(RAW_CSV) == PREVIOUS_DATA_SHA256:
        old, changed = df, set()
    else:
        if not previous_data.exists() or sha256(previous_data) != PREVIOUS_DATA_SHA256:
            raise RuntimeError(f'{previous_data} must be the data.csv of {previous_twins.name}')
        old = clean(load_raw(previous_data))
        changed = changed_interviews(old, df)
    result = check_reproduction(raw, previous_raw, changed, previous_twins.name)
    ids_old, ids_new = set(old.CASEID), set(df.CASEID)
    return {**result, 'interviews_added': len(ids_new - ids_old), 'interviews_removed': len(ids_old - ids_new)}


# ----------------------------------------------------------------------------------------------------------- units

def units(last_month, settings):
    """(first, second): the units that need nothing else, heaviest kinds first, and the boosting fits, which need the
    tuning units of the first list."""
    ends = data_ends(last_month)
    grid = [(fs, h, item) for fs in FEATURE_SETS for h in settings.half_lives for item in ITEMS]
    first = [('gb_tuning', e, *g) for e in ends if tuning_end(e) == e for g in grid]
    first += [('two_stage', e, spec, item) for e in ends for spec in two_stage.SPECS for item in ITEMS]
    first += [('ridge', e, *g) for e in ends for g in grid]
    first += [(kind, e) for e in ends for kind in ['twins', 'smoothing']]
    second = [('gb', e, *g) for e in ends for g in grid]
    return first, second


def unit_name(unit):
    return '--'.join('none' if x is None else str(x) for x in unit)


def execute(unit, ctx, chunks, commit):
    """Run one unit into chunks/<unit>.json, with the commit that ran it, unless it is there already. Returns its
    seconds, or None if skipped."""
    path = chunks / f'{unit_name(unit)}.json'
    if path.exists():
        return None
    start = time.time()
    kind, *args = unit
    if kind == 'twins':
        rows, settings = twins.fit_forecast(ctx, *args)
    elif kind == 'smoothing':
        rows, settings = simple.smoothing(ctx, *args)
    elif kind == 'ridge':
        rows, settings = simple.ridge(ctx, *args)
    elif kind == 'gb_tuning':
        rows, settings = [], simple.gb_tuning(ctx, *args)
    elif kind == 'gb':
        e, fs, h, item = args
        tuned = json.load(open(chunks / f"{unit_name(('gb_tuning', tuning_end(e), fs, h, item))}.json"))['settings']
        rows = simple.gb_fit(ctx, e, fs, h, item, tuned['chosen'])
        settings = {'tuned_at': tuning_end(e), **tuned['chosen']}
    elif kind == 'two_stage':
        rows, settings = two_stage.fit_forecast(ctx, *args)
    else:
        raise ValueError(f'unknown unit {unit}')
    seconds = round(time.time() - start, 2)
    result = {'unit': list(unit), 'commit': commit, 'seconds': seconds, 'settings': settings, 'forecasts': rows}
    path.with_suffix('.tmp').write_text(json.dumps(result, default=lambda x: x.item()))     # .item(): numpy scalars
    path.with_suffix('.tmp').rename(path)                 # a unit on disk is always complete
    return seconds


_WORKER = {}


def _start_worker(context_path):
    from threadpoolctl import threadpool_limits
    _WORKER['limits'] = threadpool_limits(limits=1)       # one thread per worker, kept for the worker's life
    _WORKER['ctx'] = pickle.load(open(context_path, 'rb'))


def _run_in_worker(unit, chunks, commit):
    return execute(unit, _WORKER['ctx'], Path(chunks), commit)


def run_units(todo, ctx, chunks, workers, commit):
    """Run the units `todo`, in this process if workers == 1, else on `workers` processes of one thread each."""
    done = 0

    def report(unit, seconds):
        nonlocal done
        done += 1
        print(f'{done}/{len(todo)} {unit_name(unit)} {seconds}s', flush=True)
    if workers == 1:
        for u in todo:
            report(u, execute(u, ctx, chunks, commit))
        return
    context_path = chunks / 'context.pkl'                 # written afresh each session, so never a partial file
    pickle.dump(ctx, open(context_path.with_suffix('.tmp'), 'wb'))
    context_path.with_suffix('.tmp').rename(context_path)
    for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[v] = '1'
    with ProcessPoolExecutor(workers, mp_context=get_context('spawn'), initializer=_start_worker,
                             initargs=(str(context_path),)) as pool:
        futures = {pool.submit(_run_in_worker, u, str(chunks), commit): u for u in todo}
        for f in as_completed(futures):
            report(futures[f], f.result())


def execute_all(ctx, chunks, workers, commit):
    """Every unit of the context's months and settings that is not on disk yet: first the units that need nothing
    else, then the boosting fits. Returns the number of units."""
    chunks.mkdir(parents=True, exist_ok=True)
    first, second = units(ctx.last_month, ctx.settings)
    for batch in (first, second):
        run_units([u for u in batch if not (chunks / f'{unit_name(u)}.json').exists()], ctx, chunks, workers, commit)
    return len(first) + len(second)


def unit_results(chunks):
    return [json.load(open(p)) for p in sorted(chunks.glob('*.json'))]


def assemble(results):
    """Every unit's forecasts as one table; every unit's settings by unit name; the seconds per kind of unit; and
    the units that recorded errors (a candidate without a fit), with their errors."""
    rows = [r for res in results for r in res['forecasts']]
    columns = ['candidate', 'data_end', 'month', 'item', 'fav', 'mid', 'unf']
    forecasts = pd.DataFrame(rows, columns=columns).sort_values(['candidate', 'data_end', 'month', 'item'],
                                                               ignore_index=True)
    seconds = pd.Series({unit_name(r['unit']): r['seconds'] for r in results}, dtype=float)
    by_kind = seconds.groupby(seconds.index.str.split('--').str[0]).sum().round().astype(int).to_dict()
    errors = {unit_name(r['unit']): r['settings']['errors'] for r in results if r['settings'].get('errors')}
    return forecasts, {unit_name(r['unit']): r['settings'] for r in results}, by_kind, errors


# -------------------------------------------------------------------------------------------------------- analysis

def window(first, last, last_month):
    return [m for m in between(first, last) if m <= last_month]


def primary(diffs, floor, months, decide=True):
    """Section 7 on the target months `months`: the test, the interval in loss units and as a share of the simple
    pick's avoidable loss, and with `decide` the outcome (the subperiods carry no decision rule). None if there are
    fewer than two months."""
    d = diffs[(diffs.horizon == OPERATING_HORIZON) & diffs.month.isin(months)].sort_values('month')
    if len(d) < 2:
        return None
    test = stats.diebold_mariano(d.d.to_numpy())
    mean_floor = floor.loc[d.month].mean()
    avoidable = d.simple_loss.mean() - mean_floor
    return {**{k: int(v) if k in ('T', 'm', 'df') else float(v) for k, v in test.items()},
            'first_month': int(d.month.iloc[0]), 'last_month': int(d.month.iloc[-1]),
            **({'outcome': stats.outcome(test)} if decide else {}),
            'simple_mean_loss': float(d.simple_loss.mean()), 'twin_mean_loss': float(d.twin_loss.mean()),
            'mean_floor': float(mean_floor), 'avoidable_loss': float(avoidable),
            'as_share_of_avoidable_loss': {k: float(test[k] / avoidable) for k in ['mean', 'lo', 'hi']}}


def analyse(forecasts, df, last_month, settings):
    """Every output of sections 4 to 8 from the forecasts: tables and dicts."""
    act = actual(df, between(FIRST_WEB, last_month))
    L = losses(forecasts, act)
    cands = table(settings)
    ends = data_ends(last_month)
    pick_table = picks(L, cands, [o for o in ends if o >= FIRST_VALIDATION], last_month)
    primary_months = window(*PRIMARY, last_month)
    curve_months = window(*CURVE, last_month) if last_month >= CURVE[1] else []
    pairs = sorted({(t, OPERATING_HORIZON) for t in primary_months} | {(t, h) for t in curve_months for h in HORIZONS})
    diffs = differences(L, pick_table, pairs) if pairs else pd.DataFrame(
        columns=['month', 'horizon', 'data_end', 'simple', 'twin', 'simple_loss', 'twin_loss', 'd'])
    floor = floors(act)
    main = primary(diffs, floor, primary_months)
    if main is not None:
        main['subperiods'] = {name: primary(diffs, floor, window(*span, last_month), decide=False)
                              for name, span in [('known', KNOWN), ('unseen', UNSEEN)]}
    secondary, draws = None, pd.DataFrame()
    if curve_months:
        D = diffs[diffs.month.isin(curve_months)].pivot(index='month', columns='horizon', values='d')[HORIZONS]
        curve, summary, drawn = stats.curve_analysis(D.to_numpy(), HORIZONS)
        label = lambda x: 'below 1' if x == -np.inf else 'above 12' if x == np.inf else float(x)
        secondary = {'months': [int(m) for m in D.index], 'curve': dict(zip(HORIZONS, map(float, curve))),
                     **{k: {s: label(v) if k == 'breakeven' else float(v) for s, v in r.items()}
                        for k, r in summary.items()},
                     'bootstrap': {'draws': stats.N_DRAWS, 'block': stats.BLOCK, 'seed': stats.SEED}}
        draws = pd.DataFrame(drawn).rename_axis('draw').reset_index()
    return {'actual': act, 'losses': L, 'candidates': cands, 'picks': pick_table, 'differences': diffs,
            'primary': main, 'secondary': secondary, 'curve_draws': draws}


# ------------------------------------------------------------------------------------------------------------- run

def load_context(twin_dir, last_month, settings, check_previous=True):
    """Checks 1 to 3 and the context. Returns the context and the run.json entries of the checks."""
    data_hash = sha256(RAW_CSV)
    raw, meta = base_probabilities(twin_dir)
    if meta['data_csv_sha256'] != data_hash:
        raise RuntimeError(f'data.csv differs from the one {twin_dir.name} used')
    df = clean(load_raw())
    df = df[df.YYYYMM <= last_month].reset_index(drop=True)
    check_answers(df, last_month)
    ctx = Context(df, twin_inputs(raw, df, last_month), last_month, settings)
    recorded = ['commit', 'option_probs_sha256', 'data_csv_sha256', 'model', 'revision', 'batch_size']
    checks = {'data_csv_sha256': data_hash, 'twins': {'name': twin_dir.name, **{k: meta.get(k) for k in recorded}}}
    if check_previous:
        checks['reproduction'] = reproduction(raw, df)
    return ctx, checks


def run(description, twins_dir, out_dir=RUNS, last_month=LAST_MONTH, settings=Settings(), workers=1, resume=False,
        check_previous=True):
    """Run into out_dir/<today>-<description>/, which must not exist, or with `resume`, continue the run in
    out_dir/<description>/. A run into experiments/ must cover every month to LAST_MONTH with the plan's settings."""
    start = time.time()
    commit = git('rev-parse', 'HEAD')
    if in_experiments(out_dir):
        require_clean_tree()
        if last_month != LAST_MONTH or settings != Settings() or not check_previous:
            raise ValueError('a run in experiments/ follows the plan: every month, the plan\'s grids and every check')
    name = description if resume else f"{time.strftime('%Y-%m-%d')}-{description}"
    out = out_dir / name
    chunks = out / 'chunks'
    if resume:
        if not out.exists():
            raise FileNotFoundError(f'{out} does not exist: --resume takes the folder name of an earlier run')
        if (out / 'run.json').exists():
            raise RuntimeError(f'{name} is complete; a new run needs a new folder')
        commits = {r['commit'] for r in unit_results(chunks)}
        if commits - {commit}:
            raise RuntimeError(f'{name} was run at commit {commits}; check out that commit to resume it')
    ctx, checks = load_context(twins_dir, last_month, settings, check_previous)
    if not resume:
        out.mkdir(parents=True)                           # a run never writes into an earlier run's folder
    n_units = execute_all(ctx, chunks, workers, commit)

    results = unit_results(chunks)
    if {r['commit'] for r in results} != {commit}:
        raise RuntimeError('every unit must come from this commit')
    forecasts, unit_settings, by_kind, errors = assemble(results)
    res = analyse(forecasts, ctx.df, last_month, settings)
    forecasts.to_parquet(out / 'forecasts.parquet', index=False)
    for k in ['actual', 'losses', 'candidates', 'picks', 'differences', 'curve_draws']:
        res[k].to_csv(out / f'{k}.csv', index=False)
    json.dump(unit_settings, open(out / 'settings.json', 'w'), indent=1)
    json.dump(res['primary'], open(out / 'primary.json', 'w'), indent=1)
    json.dump(res['secondary'], open(out / 'secondary.json', 'w'), indent=1)
    json.dump({'name': name, 'commit': commit, **checks, 'date': time.strftime('%Y-%m-%d'),
               'runtime_seconds_last_session': round(time.time() - start), 'python': sys.version.split()[0],
               'packages': {p: version(p) for p in PACKAGES}, 'last_month': last_month, 'workers': workers,
               'settings': settings.record(), 'units': n_units, 'unit_seconds_by_kind': by_kind,
               'units_with_errors': errors, 'forecast_rows': len(forecasts), 'candidates': len(res['candidates'])},
              open(out / 'run.json', 'w'), indent=1)
    for p in chunks.glob('*.json'):                       # everything is in the outputs now
        p.unlink()
    (chunks / 'context.pkl').unlink(missing_ok=True)
    print(json.dumps(res['primary'], indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('description', help='what the run contains; with --resume, the folder name')
    parser.add_argument('--twins', required=True, help='the twin run folder in experiments/realtime/')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    a = parser.parse_args()
    run(a.description, RUNS / a.twins, workers=a.workers, resume=a.resume)
