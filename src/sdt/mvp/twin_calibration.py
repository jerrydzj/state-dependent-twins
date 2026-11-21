"""The mvp-twin study's twins, from a twin run's raw option probabilities (sdt.twin.run): the zero-shot twin, and the
shifted and scaled twins calibrated on the training window's web interviews (April to July 2024).

    python -m sdt.mvp.twin_calibration twin-pilot-calibration --from 2026-10-30-twin-pilot-training-web   # no GPU

`postprocess` gives the three twins per feature set in the format of the supervised runs: `twin_<feature set>`
(zero-shot), `twin_shift_<feature set>` (levels calibrated) and `twin_scale_<feature set>` (levels and scale
calibrated). DUR's middle share is the baseline shares' middle share, its weighted share among the training window's
web interviews. twin_score.py calls `postprocess` itself; the command above writes the same twins to a new folder
experiments/mvp/<date>-<description>/ from an earlier run's option_probs.parquet, with:

- predictions.parquet: base-stage class probabilities of the interviews that answered each item (not in git).
- calibration.json: each calibrated twin's a and b per feature set and item, and DUR's middle share.
- calibration_cv.csv: a check of calibration on months it was not fitted on. Each training month with web interviews
  is held out in turn, the calibrations and the baseline shares are fitted on the other months, and the held-out
  month is scored; the log loss pools the held-out predictions of all months.
- run.json: commit, data hash, date, package versions, and the source run with the SHA-256 of the file it read, so
  anyone holding the file can check that it is the same one.
"""
import argparse
import json
import sys
import time
from importlib.metadata import version

import numpy as np
import pandas as pd

from sdt.data.clean import CLASSES, OUT_PATH
from sdt.experiment import git, in_experiments, require_clean_tree, sha256
from sdt.mvp import EXPERIMENTS
from sdt.mvp.score import log_loss
from sdt.mvp.windows import TRAIN_END, split
from sdt.paths import RAW_CSV
from sdt.twin import calibration

CALIBRATIONS = {'shift': False, 'scale': True}      # twin name -> whether the scale is fitted (fit_calibration)


def dur_mid_share(df):
    """The baseline shares' middle share for DUR: its weighted share among the training window's web interviews."""
    train, _ = split(df)
    d = train[(train.web == 1) & train.DUR.notna()]
    return float(np.average(d.DUR == 'mid', weights=d.WT))


def answered(raw, key, item):
    """Rows of `raw` whose interview answered `item`."""
    return raw[key.loc[raw.CASEID, item].notna().to_numpy()]


def in_training_window(raw, key):
    """Whether each row of `raw` is an interview of the training window (up to TRAIN_END), on which the twins are
    calibrated. Every twin is run only on web interviews."""
    return (key.loc[raw.CASEID, 'YYYYMM'] <= TRAIN_END).to_numpy()


def postprocess(raw, df):
    """Base-stage predictions of the three twins in the supervised format, and their calibrations, from the raw option
    probabilities."""
    key = df.set_index('CASEID')
    mid_share = dur_mid_share(df)
    preds, fits = [], {}
    for (fs, item), g in raw[raw.stage == 'base'].groupby(['feature_set', 'item'], sort=False):
        g = answered(g, key, item)
        P = calibration.class_probs(g, item, mid_share)
        fit = in_training_window(g, key)
        a = key.loc[g.CASEID[fit]]
        twins = {f'twin_{fs}': P}
        for name, scale in CALIBRATIONS.items():
            c = calibration.fit_calibration(P[fit], a[item].astype(str), a.WT, scale)
            fits.setdefault(name, {}).setdefault(fs, {})[item] = {'a': c['a'], 'fav': c['fav'], 'unf': c['unf']}
            twins[f'twin_{name}_{fs}'] = calibration.apply_calibration(P, c)
        preds += [Q[CLASSES].assign(item=item, CASEID=g.CASEID.to_numpy(), model=m) for m, Q in twins.items()]
    pred = pd.concat(preds, ignore_index=True)[['item', 'CASEID'] + CLASSES + ['model']]
    return pred, {'dur_mid_share': mid_share, **fits}


def calibration_cv(raw, df):
    """Log loss of each twin and of the baseline shares on training-window months they were not fitted on: each month
    is held out in turn and predicted from fits on the others (DUR's middle share included); one row per feature set,
    item and model, pooling the held-out predictions of every month. Empty if the training window's web interviews
    span fewer than two months."""
    key = df.set_index('CASEID')
    base = raw[raw.stage == 'base']
    rows = []
    for (fs, item), g in base[in_training_window(base, key)].groupby(['feature_set', 'item'], sort=False):
        g = answered(g, key, item)
        a = key.loc[g.CASEID]
        y, w, month = a[item].astype(str).to_numpy(), a.WT.to_numpy(), a.YYYYMM.to_numpy()
        if len(np.unique(month)) < 2:
            return pd.DataFrame()
        held_out = {m: pd.DataFrame(index=g.index, columns=CLASSES, dtype=float) for m in ['baseline', 'twin', *CALIBRATIONS]}
        for m in np.unique(month):
            fit, test = month != m, month == m
            shares = np.array([np.average(y[fit] == c, weights=w[fit]) for c in CLASSES])
            P = calibration.class_probs(g, item, shares[1])
            held_out['baseline'].loc[test] = np.tile(shares, (test.sum(), 1))
            held_out['twin'].loc[test] = P[test][CLASSES].to_numpy()
            for name, scale in CALIBRATIONS.items():
                c = calibration.fit_calibration(P[fit], y[fit], w[fit], scale)
                held_out[name].loc[test] = calibration.apply_calibration(P[test], c)[CLASSES].to_numpy()
        rows += [{'feature_set': fs, 'item': item, 'model': m, 'log_loss': log_loss(y, Q.to_numpy(float), w)}
                 for m, Q in held_out.items()]
    return pd.DataFrame(rows)


def rescore(description, source, out_dir=EXPERIMENTS):
    """Write the twins of the earlier run `source` (a folder name in out_dir) into out_dir/<today>-<description>/,
    without the model."""
    if in_experiments(out_dir):
        require_clean_tree()
    name = f"{time.strftime('%Y-%m-%d')}-{description}"
    out = out_dir / name
    out.mkdir(parents=True)
    src = json.load(open(out_dir / source / 'run.json'))
    src_raw = out_dir / source / 'option_probs.parquet'
    if src.get('option_probs_sha256', sha256(src_raw)) != sha256(src_raw):
        raise RuntimeError(f'{src_raw} differs from the file its run wrote')
    df = pd.read_parquet(OUT_PATH)
    raw = pd.read_parquet(src_raw)
    pred, fits = postprocess(raw, df)
    pred.to_parquet(out / 'predictions.parquet', index=False)
    json.dump(fits, open(out / 'calibration.json', 'w'), indent=1)
    calibration_cv(raw, df).to_csv(out / 'calibration_cv.csv', index=False, float_format='%.6f')
    json.dump({'name': name, 'commit': git('rev-parse', 'HEAD'), 'data_csv_sha256': sha256(RAW_CSV),
               'date': time.strftime('%Y-%m-%d'),
               'python': sys.version.split()[0],
               'packages': {p: version(p) for p in ['numpy', 'pandas', 'scipy']},
               'source': {**{k: src[k] for k in ['name', 'commit', 'model', 'revision', 'device', 'interviews', 'stages']},
                          'option_probs_sha256': sha256(src_raw)}},
              open(out / 'run.json', 'w'), indent=1)
    print(pd.read_csv(out / 'calibration_cv.csv').pivot_table(index=['feature_set', 'item'], columns='model',
                                                             values='log_loss').round(3).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('description', help='what the run contains')
    parser.add_argument('--from', dest='source', required=True, help='twin run folder in experiments/mvp/ to read')
    a = parser.parse_args()
    rescore(a.description, a.source)
