"""Pilot checks of the GPU and CPU machines before the realtime-benchmark tag. Each takes under five minutes, and none
forecasts a month after July 2024.

    python -m sdt.realtime.pilot twins /tmp/twin-pilot/<date>-pilot     # GPU: compare with the 2026-11-05 run
    python -m sdt.realtime.pilot timing --workers 32                     # CPU: time one fold of each heavy fit
    python -m sdt.realtime.pilot realtime --workers 32                   # CPU: the realtime run on small grids
    python -m sdt.realtime.pilot scaling --workers 1,8,16,32             # CPU: throughput at each number of workers

`twins` checks that a pilot twin run (sdt.twin.run with --limit) reproduces the 2026-11-05 run's probabilities on the
prompts both ran. `timing` runs one cross-validation fold of boosting, the one-stage ridge and stage 1 at data end
July 2024, on one core, for PAGO on the largest input set (D+M+P), and extrapolates to the whole run on this machine.
`realtime` runs the realtime procedure on real data to July 2024 with small grids and every input check, on
--workers processes, into /tmp/realtime-pilot/. `scaling` runs the same ridge fit and the same boosting fit on each
number of workers at once, each worker on one thread as in the run, and reports how many single cores' worth of work
the machine does at each number: workers who slow each other down, through shared memory bandwidth or shared cores,
add less than one core each.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from sdt import tune
from sdt.data.clean import OUT_PATH
from sdt.features import ENCODERS, item_data
from sdt.models import Ridge, gb
from sdt.realtime import run as realtime
from sdt.realtime.candidates import Settings
from sdt.realtime.months import data_ends, tuning_end
from sdt.realtime.two_stage import StageOne, one_hot, respondent_terms

SMALL = Settings(ks=(1, None), alphas=(1.0, 0.5), half_lives=(None,), ridge_C=(1.0,),
                 gb_grid=({'max_leaf_nodes': 4, 'min_samples_leaf': 20},), gb_max_trees=5, stage1_C=(1.0,),
                 stage2_alphas=(1.0, 0.0), stage1_folds=2)


def twins(folder):
    raw, meta = realtime.base_probabilities(Path(folder))
    previous, _ = realtime.base_probabilities(realtime.PREVIOUS_TWINS)
    result = realtime.check_reproduction(raw, previous, set(), realtime.PREVIOUS_TWINS.name)
    print(json.dumps({'device': meta['device'], 'batch_size': meta['batch_size'], **result}, indent=1))
    print('PASS: the pilot reproduces the earlier run within', realtime.REPRODUCTION_TOLERANCE)


def timing(workers):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        E, ITEM, FS = 202407, 'PAGO', 'demographics_macro_politics'
        C_GRID = list(map(float, tune.C_GRID))
        df = pd.read_parquet(OUT_PATH)
        train = df[df.YYYYMM <= E]
        fit, val = next(tune.folds(train))                      # the first of the 5 month-blocked folds
        out = {}
        X, y, w = item_data(fit, ITEM, ENCODERS[FS]['gb'])
        Xv, yv, wv = item_data(val, ITEM, ENCODERS[FS]['gb'])
        gb_fold = 0.0
        for leaves in (4, 8, 16):                               # every tree size of the grid, 1,000 trees
            t = time.time()
            model = gb(max_iter=tune.MAX_TREES, max_leaf_nodes=leaves, min_samples_leaf=50).fit(X, y, sample_weight=w)
            tune.staged_loss(model, Xv, yv, wv)
            out[f'gb_fold_{leaves}_leaves_seconds'] = round(time.time() - t, 1)
            gb_fold += time.time() - t
            print(out, flush=True)
        X, y, w = item_data(fit, ITEM, ENCODERS[FS]['mnl'])
        t = time.time()
        for C in C_GRID:
            Ridge(C).fit(X, y, sample_weight=w)
        ridge_fold = time.time() - t
        d = train[train[ITEM].notna()]
        months = np.array(sorted(d.YYYYMM.unique()))
        month = np.searchsorted(months, d.YYYYMM.to_numpy())
        R = respondent_terms(d, 'D+P').to_numpy(float)
        Y, wt, web = one_hot(d[ITEM].astype(str)), d.WT.to_numpy(float), d.web.to_numpy(float)
        rows = StageOne.folds(month, Y, 5, 0) != 0
        t = time.time()
        theta = None
        for C in C_GRID:
            theta, _, _ = StageOne.fit(R, month, web, Y, wt, rows, len(months), C, theta)
        stage1_fold = time.time() - t
    out |= {'ridge_fold_seconds': round(ridge_fold, 1), 'stage1_fold_seconds': round(stage1_fold, 1)}
    # Whole run: every data end, feature set, half-life and item, with data growing about 1.25 times on average over
    # the data ends; one fold of 5 plus a final fit on all rows, and boosting's 3 values of min_samples_leaf
    ends, grow, units = data_ends(), 1.25, 4 * 3 * 5
    n_tuning = len([e for e in ends if tuning_end(e) == e])
    hours = {'gb_tuning': n_tuning * units * gb_fold * 3 * tune.N_FOLDS * grow / 3600,
             'gb_fits': len(ends) * units * gb_fold / 3 * 1.25 * grow / 3600,
             'ridge': len(ends) * units * ridge_fold * (tune.N_FOLDS + 1.25 / len(C_GRID)) * grow / 3600,
             'stage1': len(ends) * 2 * 5 * stage1_fold * (5 + 1.25 / len(C_GRID)) * grow / 3600}
    total = sum(hours.values())
    out |= {'core_hours': {k: round(v, 1) for k, v in hours.items()} | {'total': round(total, 1)},
            'wall_hours_on_workers': {'workers': workers, 'hours': round(total / workers, 1)}}
    print(json.dumps(out, indent=1))


def realtime_pilot(workers):
    out = Path('/tmp/realtime-pilot')
    description = f"pilot-{time.strftime('%H%M%S')}"
    start = time.time()
    realtime.run(description, realtime.PREVIOUS_TWINS, out_dir=out, last_month=202407, settings=SMALL, workers=workers)
    folder = next(out.glob(f'*-{description}'))
    meta = json.load(open(folder / 'run.json'))
    forecasts = pd.read_parquet(folder / 'forecasts.parquet')
    summary = {'folder': str(folder), 'seconds': round(time.time() - start), 'units': meta['units'],
               'units_with_errors': meta['units_with_errors'], 'unit_seconds_by_kind': meta['unit_seconds_by_kind'],
               'reproduction': {k: meta['reproduction'][k] for k in ['prompts_compared', 'max_abs_difference']},
               'last_forecast_month': int(forecasts.month.max()), 'picks': len(pd.read_csv(folder / 'picks.csv'))}
    print(json.dumps(summary, indent=1))
    if summary['units_with_errors'] or summary['last_forecast_month'] > 202407:
        raise SystemExit('FAIL: see units_with_errors and last_forecast_month above')
    print('PASS: every unit ran, the checks passed, and no forecast is later than July 2024')


_PROBE = {}


def _start_probe():
    """Worker set-up for `scaling`: one thread, and one fold's data for both fits."""
    from threadpoolctl import threadpool_limits
    _PROBE['limits'] = threadpool_limits(limits=1)
    df = pd.read_parquet(OUT_PATH)
    fit, _ = next(tune.folds(df[df.YYYYMM <= 202407]))
    fs = 'demographics_macro_politics'
    _PROBE['ridge'] = item_data(fit, 'PAGO', ENCODERS[fs]['mnl'])
    _PROBE['boosting'] = item_data(fit, 'PAGO', ENCODERS[fs]['gb'])


def _probe(kind):
    """Seconds for one fit of `kind`, in a worker; 'ready' only confirms the worker is set up."""
    if kind == 'ready':
        return 0.0
    X, y, w = _PROBE[kind]
    start = time.time()
    if kind == 'ridge':
        for _ in range(3):
            Ridge(1.0).fit(X, y, sample_weight=w)
    else:
        gb(max_iter=300, max_leaf_nodes=16, min_samples_leaf=50).fit(X, y, sample_weight=w)
    return time.time() - start


def scaling(levels):
    for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[v] = '1'
    rows = []
    for kind in ['ridge', 'boosting']:
        single = None
        for n in levels:
            with ProcessPoolExecutor(n, mp_context=get_context('spawn'), initializer=_start_probe) as pool:
                list(pool.map(_probe, ['ready'] * (4 * n)))          # every worker started and set up
                seconds = list(pool.map(_probe, [kind] * n))        # n fits at once
            mean = float(np.mean(seconds))
            single = single or mean
            rows.append({'fit': kind, 'workers': n, 'seconds_per_fit': round(mean, 1),
                         'effective_cores': round(n * single / mean, 1)})
            print(rows[-1], flush=True)
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='check', required=True)
    sub.add_parser('twins').add_argument('folder', help='the pilot twin run folder')
    for name in ['timing', 'realtime']:
        sub.add_parser(name).add_argument('--workers', type=int, required=True)
    sub.add_parser('scaling').add_argument('--workers', required=True, help='comma-separated, first 1')
    a = parser.parse_args()
    {'twins': lambda: twins(a.folder), 'timing': lambda: timing(a.workers),
     'realtime': lambda: realtime_pilot(a.workers),
     'scaling': lambda: scaling([int(n) for n in a.workers.split(',')])}[a.check]()
