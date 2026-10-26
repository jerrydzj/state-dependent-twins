"""Run the MVP supervised models on one feature set and save the results to experiments/mvp/<name>/.

    python -m sdt.mvp.run demographics      # or: state

Tunes gradient boosting, fits the baseline shares, multinomial logit and gradient boosting per item on the training
window, predicts the evaluation window and scores both levels. Writes:

- scores.csv: log loss and Wasserstein distance per model and item, with skill against the baseline shares and 95%
  bootstrap intervals (`_lo`, `_hi`); rows for the MNL - GB difference; and the uniform guess and the bootstrap noise
  floor as reference rows.
- gb_settings.json: the chosen setting per item, and each grid setting's best validation loss and tree count.
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
from sdt.mvp.encode import gb_features, gb_state_features, mnl_features, mnl_state_features
from sdt.mvp.models import BaselineShares, gb, mnl
from sdt.mvp.score import benchmarks, bootstrap, fit_predict, score_predictions, skill
from sdt.mvp.windows import split
from sdt.paths import RAW_CSV, ROOT

EXPERIMENTS = ROOT / 'experiments' / 'mvp'
FEATURE_SETS = {'demographics': {'mnl': mnl_features, 'gb': gb_features},   # name: encoder per model family
                'state': {'mnl': mnl_state_features, 'gb': gb_state_features}}


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def score_run(pred, df):
    """Scores table for a run's predictions: point scores and skill per model and item, the MNL - GB difference,
    bootstrap intervals, and the uniform guess and noise floor as reference rows. Uses only saved predictions."""
    models = list(pred.model.unique())
    scores = pd.concat([score_predictions(g, df).assign(model=k) for k, g in pred.groupby('model', sort=False)])
    diff = scores[scores.model == 'mnl'].drop(columns='model') - scores[scores.model == 'gb'].drop(columns='model')
    ref = benchmarks(df)
    scores = pd.concat([scores, diff.assign(model='mnl-gb'),
                        ref[['uniform_log_loss', 'uniform_wasserstein']].set_axis(['log_loss', 'wasserstein'], axis=1)
                        .assign(model='uniform'),
                        ref[['floor_wasserstein']].set_axis(['wasserstein'], axis=1).assign(model='floor')])
    scores = scores.rename_axis('item').reset_index()
    base = scores[scores.model == 'baseline'].set_index('item')
    for s in ['log_loss', 'wasserstein']:
        scores[f'{s}_skill'] = skill(scores[s], scores.item.map(base[s])).where(scores.model.isin(models))
    return scores.merge(bootstrap(pred, df), on=['item', 'model'], how='left')


def run(name):
    start, encoders = time.time(), FEATURE_SETS[name]
    if git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('commit your changes first: outputs must trace back to a commit')
    out = EXPERIMENTS / name
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(OUT_PATH)
    train, _ = split(df)

    settings = {}
    for item in OUTCOMES:
        best, curves = tuning.tune(train, item, encoders['gb'])
        grid = curves.min(axis=1).rename('val_log_loss').to_frame().assign(best_trees=curves.idxmin(axis=1))
        settings[item] = {'chosen': best, 'grid': grid.reset_index().to_dict('records')}
        print(item, best, f'{time.time() - start:.0f}s', flush=True)

    models = {'baseline': (BaselineShares(), encoders['mnl']), 'mnl': (mnl(), encoders['mnl']),
              'gb': ({item: gb(**s['chosen']) for item, s in settings.items()}, encoders['gb'])}
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
               'seeds': {'cv_months': tuning.SEED, 'gb': 0, 'noise_floor': 0, 'bootstrap': 0},
               'tuning': {'grid': tuning.GRID, 'max_trees': tuning.MAX_TREES, 'folds': tuning.N_FOLDS}},
              open(out / 'run.json', 'w'), indent=1)
    print(scores.round(4).to_string())


if __name__ == '__main__':
    run(sys.argv[1])
