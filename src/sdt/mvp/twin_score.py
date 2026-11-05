"""Score a twin run's twins against a supervised run's models on the evaluation window, run the primary test of the
mvp-twin preregistration (primary.py), and save the results to a new folder experiments/mvp/<date>-<description>/.

    python -m sdt.mvp.twin_score twin-prereg-scores     # needs the `twin` extra, for the twins' post-processing

Before scoring, the command checks its inputs and stops if any check fails:

1. data.csv has the SHA-256 that both runs recorded.
2. The twin run's option_probs.parquet has the SHA-256 that its run.json recorded.
3. The twins' predictions are recomputed from those probabilities (twin_run.postprocess), and the recomputed
   calibration matches the twin run's calibration.json.
4. The supervised run's predictions reproduce the point scores of its scores.csv in every cell.

The primary test and every cell use the same bootstrap draws of the evaluation window (score.draw_counts, seed 0).
Nothing is written until every check has passed. Writes:

- scores.csv: the format of sdt.mvp.run's, for the supervised run's 13 models and the 12 twins (zero-shot `twin_`,
  shifted `twin_shift_` and scaled `twin_scale_`, on each feature set), with a row per pair in PAIRS.
- primary.json: the plan's twin and comparators, rho on the sample and its interval against each comparator, the
  outcome, the time sibling of rho with its interval (exploratory, primary.py), and the Spearman correlation of d_t
  with elapsed months (a diagnostic in the plan).
- primary_months.csv: per evaluation month, d_t, elapsed months, A_t against each comparator, and rho and its time
  sibling with that month left out (exploratory).
- primary_distances.csv: per item, model and month, the Wasserstein distance behind A_t.
- rho_draws.csv: rho and its time sibling against each comparator on every bootstrap draw.
- calibration_by_period.csv: each twin calibration refit on each evaluation period, next to its fit on the training
  window's web interviews (a diagnostic in the plan).
- run.json: commit, data hash, date, package versions, the two source runs, the bootstrap settings and PAIRS.
"""
import json
import sys
import time
from importlib.metadata import version

import numpy as np
import pandas as pd

from sdt.mvp import run as supervised_run
from sdt.mvp import twin
from sdt.mvp.clean import CLASSES, OUT_PATH, OUTCOMES
from sdt.mvp.primary import COMPARATORS, TWIN, primary_test, spearman, state_dissimilarity
from sdt.mvp.prompts import FEATURE_SETS
from sdt.mvp.run import EXPERIMENTS, git, require_clean_tree, score_run
from sdt.mvp.score import cells, draw_counts
from sdt.mvp.twin_run import CALIBRATIONS, postprocess, sha256
from sdt.mvp.windows import split
from sdt.paths import RAW_CSV

TWIN_RUN = '2026-11-05-twin-prereg-eval'
SUPERVISED_RUN = '2026-10-26-supervised-macro-politics-ridge'
N_BOOT, SEED = 1000, 0
TWINS = ['twin', 'twin_shift', 'twin_scale']
# The supervised run's pairs, then each twin minus the logit on the same feature set, and what each block adds to
# demographics for each twin
PAIRS = (supervised_run.PAIRS + [(f'{t}_{fs}', f'mnl_{fs}') for fs in FEATURE_SETS for t in TWINS]
         + [(f'{t}_{fs}', f'{t}_demographics') for fs in list(FEATURE_SETS)[1:] for t in TWINS])
POINT = ['log_loss', 'wasserstein', 'log_loss_skill', 'wasserstein_skill']


def twin_predictions(df, twins):
    """Checks 2 and 3: the twins' predictions, recomputed from the twin run's raw probabilities."""
    meta = json.load(open(twins / 'run.json'))
    if sha256(twins / 'option_probs.parquet') != meta['option_probs_sha256']:
        raise RuntimeError(f'{twins.name}/option_probs.parquet differs from the file its run wrote')
    pred, calibration = postprocess(pd.read_parquet(twins / 'option_probs.parquet'), df)
    flat = lambda c: pd.json_normalize(c).iloc[0].sort_index().astype(float)
    new, recorded = flat(calibration), flat(json.load(open(twins / 'calibration.json')))
    if not (new.index.equals(recorded.index) and np.allclose(new, recorded, rtol=0, atol=1e-9)):
        raise RuntimeError(f'the recomputed calibration differs from {twins.name}/calibration.json')
    return pred


def check_supervised(scores, supervised):
    """Check 4: the supervised run's point scores, recomputed here, match its scores.csv in every cell."""
    key = ['period', 'group', 'item', 'model']
    old = pd.read_csv(supervised / 'scores.csv').set_index(key)
    missing = old.index.difference(scores.set_index(key).index)
    if len(missing):
        raise RuntimeError(f'rows of {supervised.name}/scores.csv not scored here: {list(missing[:3])}')
    new = scores.set_index(key).loc[old.index, POINT].round(6)
    if not np.allclose(new, old[POINT], rtol=0, atol=1e-6, equal_nan=True):
        raise RuntimeError(f'the supervised predictions do not reproduce {supervised.name}/scores.csv')


def calibration_by_period(pred, df):
    """Each twin calibration (CALIBRATIONS) of each feature set and item, fitted on the training window's web
    interviews and refit on the interviews of each evaluation period, from the zero-shot twin's probabilities."""
    train, _ = split(df)
    fits = {'training web': train[train.web == 1]} | {p: d for p, g, d in cells(df) if p != 'all' and g == 'all'}
    key = df.set_index('CASEID')
    rows = []
    for fs in FEATURE_SETS:
        for item in OUTCOMES:
            P = pred[(pred.model == f'twin_{fs}') & (pred.item == item)].set_index('CASEID')[CLASSES]
            for fitted_on, d in fits.items():
                ids = P.index[P.index.isin(d.CASEID)]
                a = key.loc[ids]
                for name, scale in CALIBRATIONS.items():
                    c = twin.fit_calibration(P.loc[ids], a[item].astype(str), a.WT, scale)
                    rows.append({'feature_set': fs, 'item': item, 'calibration': name, 'fitted_on': fitted_on,
                                 'interviews': len(ids), 'a': c['a'], 'b_fav': c['fav'], 'b_unf': c['unf']})
    return pd.DataFrame(rows)


def score(df, twins, supervised):
    """Every output of the module docstring as tables and dicts, from the clean table `df` and the two run folders."""
    data_hash = sha256(RAW_CSV)
    for source in (twins, supervised):
        if json.load(open(source / 'run.json'))['data_csv_sha256'] != data_hash:
            raise RuntimeError(f'data.csv differs from the one {source.name} used')                  # check 1
    twin_pred = twin_predictions(df, twins)                                                         # checks 2, 3
    _, test = split(df)
    pred = pd.concat([pd.read_parquet(supervised / 'predictions.parquet'),
                      twin_pred[twin_pred.CASEID.isin(test.CASEID)]], ignore_index=True)
    draws = draw_counts(test, N_BOOT, SEED)
    scores = score_run(pred, df, PAIRS, draws)
    check_supervised(scores, supervised)                                                            # check 4

    d = state_dissimilarity(df)
    result = primary_test(pred, test, d, TWIN, COMPARATORS, draws=draws)
    as_dict = lambda t: {c: {k: float(v) for k, v in r.items()} for c, r in t.iterrows()}
    primary = {'twin': TWIN, 'comparators': COMPARATORS, 'outcome': result['outcome'],
               'rho': as_dict(result['rho']), 'rho_time': as_dict(result['rho_time']),
               'spearman_d_elapsed_months': float(spearman(d.to_numpy(float), result['elapsed'].to_numpy(float)))}
    months = pd.concat([d.rename('d'), result['elapsed'], result['A'].add_prefix('A_'),
                        result['leave_one_out'].add_prefix('rho_without_month_'),
                        result['leave_one_out_time'].add_prefix('rho_time_without_month_')],
                       axis=1).rename_axis('month')
    rho_draws = pd.concat([result['rho_draws'].add_prefix('rho_'), result['rho_time_draws'].add_prefix('rho_time_')],
                          axis=1)
    return {'scores': scores, 'primary': primary, 'primary_months': months.reset_index(),
            'primary_distances': result['distances'], 'rho_draws': rho_draws.rename_axis('draw').reset_index(),
            'calibration_by_period': calibration_by_period(twin_pred, df), 'data_hash': data_hash}


def run(description, twins=TWIN_RUN, supervised=SUPERVISED_RUN, source_dir=EXPERIMENTS, out_dir=EXPERIMENTS, df=None):
    """Score into out_dir/<today>-<description>/, which must not exist, from the runs `twins` and `supervised` in
    source_dir. `df` defaults to the clean table."""
    start = time.time()
    if out_dir.resolve().is_relative_to(EXPERIMENTS.resolve()):
        require_clean_tree()
    name = f"{time.strftime('%Y-%m-%d')}-{description}"
    out = out_dir / name
    if out.exists():
        raise FileExistsError(f'{out} exists: a run never writes into an earlier run\'s folder')
    df = pd.read_parquet(OUT_PATH) if df is None else df
    res = score(df, source_dir / twins, source_dir / supervised)

    out.mkdir(parents=True)
    res['scores'].to_csv(out / 'scores.csv', index=False, float_format='%.6f')
    json.dump(res['primary'], open(out / 'primary.json', 'w'), indent=1)
    for k in ['primary_months', 'primary_distances', 'rho_draws', 'calibration_by_period']:
        res[k].to_csv(out / f'{k}.csv', index=False, float_format='%.6f')
    sources = {k: json.load(open(source_dir / r / 'run.json'))
               for k, r in [('twins', twins), ('supervised', supervised)]}
    json.dump({'name': name, 'commit': git('rev-parse', 'HEAD'), 'data_csv_sha256': res['data_hash'],
               'date': time.strftime('%Y-%m-%d'), 'runtime_seconds': round(time.time() - start),
               'python': sys.version.split()[0],
               'packages': {p: version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch']},
               'sources': {'twins': {k: sources['twins'][k] for k in ['name', 'commit', 'option_probs_sha256']},
                           'supervised': {k: sources['supervised'][k] for k in ['name', 'commit']}},
               'bootstrap': {'draws': N_BOOT, 'seed': SEED}, 'pairs': PAIRS},
              open(out / 'run.json', 'w'), indent=1)
    print(json.dumps(res['primary'], indent=1))


if __name__ == '__main__':
    run(sys.argv[1])
