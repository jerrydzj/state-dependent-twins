"""Run the MVP supervised models on every feature set and save the results to a new folder
experiments/mvp/<date>-<description>/, where the description says what the run contains.

    python -m sdt.mvp.run supervised-demographics-state

For each feature set, tunes gradient boosting, then fits the multinomial logit and gradient boosting per item on the
training window; the baseline shares are fitted once. Models are named <family>_<feature set>, e.g.
`mnl_demographics_state`. Predicts the evaluation window and scores both levels. Writes:

- scores.csv: log loss and Wasserstein distance per evaluation cell (period x party group, see score.cells), model and
  item, with skill against the baseline shares and 95% bootstrap intervals (`_lo`, `_hi`); a row per pair in PAIRS
  for the difference a - b; and the uniform guess and the bootstrap noise floor as reference rows.
- gb_settings.json: per feature set and item, the chosen setting and each grid setting's best validation loss and
  tree count.
- run.json: commit, data hash, date, package versions and seeds.
- predictions.parquet: every model's probabilities for every evaluation interview (not in git: a derivative of the
  microdata).
"""
import hashlib
import json
import subprocess
import sys
import time
from importlib.metadata import version

import pandas as pd

from sdt.mvp import tune as tuning
from sdt.mvp.clean import OUT_PATH, OUTCOMES
from sdt.mvp.encode import gb_demographics_state_features, gb_features, mnl_demographics_state_features, mnl_features
from sdt.mvp.models import BaselineShares, gb, mnl
from sdt.mvp.score import benchmarks, bootstrap, cells, fit_predict, score_predictions, skill
from sdt.mvp.windows import split
from sdt.paths import RAW_CSV, ROOT

EXPERIMENTS = ROOT / 'experiments' / 'mvp'
FEATURE_SETS = {'demographics': {'mnl': mnl_features, 'gb': gb_features},   # name: encoder per model family
                'demographics_state': {'mnl': mnl_demographics_state_features, 'gb': gb_demographics_state_features}}
# Differences a - b with paired intervals: MNL vs GB within each feature set, and what the state adds to each family
PAIRS = ([(f'mnl_{fs}', f'gb_{fs}') for fs in FEATURE_SETS]
         + [(f'{family}_demographics_state', f'{family}_demographics') for family in ('mnl', 'gb')])


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def score_run(pred, df, pairs=PAIRS):
    """score_cell for every evaluation cell, with `period` and `group` columns first."""
    out = [score_cell(pred[pred.CASEID.isin(d.CASEID)], d, pairs).assign(period=p, group=g) for p, g, d in cells(df)]
    scores = pd.concat(out, ignore_index=True)
    return scores[['period', 'group'] + [c for c in scores if c not in ('period', 'group')]]


def score_cell(pred, df, pairs=PAIRS):
    """Scores table for one cell's predictions: point scores and skill per model and item, the difference a - b for
    each pair of models present, bootstrap intervals, and the uniform guess and noise floor as reference rows. Uses
    only saved predictions."""
    models = list(pred.model.unique())
    pairs = [(a, b) for a, b in pairs if a in models and b in models]
    scores = pd.concat([score_predictions(g, df).assign(model=k) for k, g in pred.groupby('model', sort=False)])
    point = lambda m: scores[scores.model == m].drop(columns='model')
    diffs = [(point(a) - point(b)).assign(model=f'{a}-{b}') for a, b in pairs]
    ref = benchmarks(df, items=list(pred.item.unique()))
    scores = pd.concat([scores, *diffs,
                        ref[['uniform_log_loss', 'uniform_wasserstein']].set_axis(['log_loss', 'wasserstein'], axis=1)
                        .assign(model='uniform'),
                        ref[['floor_wasserstein']].set_axis(['wasserstein'], axis=1).assign(model='floor')])
    scores = scores.rename_axis('item').reset_index()
    base = scores[scores.model == 'baseline'].set_index('item')
    for s in ['log_loss', 'wasserstein']:
        scores[f'{s}_skill'] = skill(scores[s], scores.item.map(base[s])).where(scores.model.isin(models))
    return scores.merge(bootstrap(pred, df, pairs=pairs), on=['item', 'model'], how='left')


def require_clean_tree():
    if git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('commit your changes first: outputs must trace back to a commit')


def run(description):
    """Fit, predict and score every feature set into experiments/mvp/<today>-<description>/, which must not exist."""
    start = time.time()
    require_clean_tree()
    name = f"{time.strftime('%Y-%m-%d')}-{description}"
    out = EXPERIMENTS / name
    out.mkdir(parents=True)                             # a run never writes into an earlier run's folder
    df = pd.read_parquet(OUT_PATH)
    train, _ = split(df)

    settings, models = {}, {'baseline': (BaselineShares(), mnl_features)}
    for fs, enc in FEATURE_SETS.items():
        settings[fs] = {}
        for item in OUTCOMES:
            best, curves = tuning.tune(train, item, enc['gb'])
            grid = curves.min(axis=1).rename('val_log_loss').to_frame().assign(best_trees=curves.idxmin(axis=1))
            settings[fs][item] = {'chosen': best, 'grid': grid.reset_index().to_dict('records')}
            print(fs, item, best, f'{time.time() - start:.0f}s', flush=True)
        models[f'mnl_{fs}'] = (mnl(), enc['mnl'])
        models[f'gb_{fs}'] = ({item: gb(**s['chosen']) for item, s in settings[fs].items()}, enc['gb'])
    pred = pd.concat([fit_predict(m, enc, df).assign(model=k) for k, (m, enc) in models.items()], ignore_index=True)
    scores = score_run(pred, df)

    scores.to_csv(out / 'scores.csv', index=False, float_format='%.6f')
    json.dump(settings, open(out / 'gb_settings.json', 'w'), indent=1, default=int)
    pred.to_parquet(out / 'predictions.parquet', index=False)
    json.dump({'name': name, 'commit': git('rev-parse', 'HEAD'),
               'data_csv_sha256': hashlib.sha256(RAW_CSV.read_bytes()).hexdigest(),
               'date': time.strftime('%Y-%m-%d'), 'runtime_seconds': round(time.time() - start),
               'python': sys.version.split()[0],
               'packages': {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn']},
               'feature_sets': list(FEATURE_SETS), 'models': list(models), 'pairs': PAIRS,
               'seeds': {'cv_months': tuning.SEED, 'gb': 0, 'noise_floor': 0, 'bootstrap': 0},
               'tuning': {'grid': tuning.GRID, 'max_trees': tuning.MAX_TREES, 'folds': tuning.N_FOLDS}},
              open(out / 'run.json', 'w'), indent=1)
    print(scores[(scores.period == 'all') & (scores.group == 'all')].round(4).to_string())


if __name__ == '__main__':
    run(sys.argv[1])
