"""The 92 candidates of section 6 of the realtime-benchmark plan, in the plan's order.

A candidate is one model with every setting fixed. Its name is `<model>/<input set>/<setting>`, with the input sets
named as in the plan: D, D+M, D+P and D+M+P. The order breaks exact ties at selection: by model as numbered in the
plan, then by input set as listed, then by the remaining setting as listed.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sdt import tune
from sdt.features import FEATURE_SETS

INPUT_SETS = dict(zip(FEATURE_SETS, ['D', 'D+M', 'D+P', 'D+M+P']))       # feature set -> the plan's name


@dataclass(frozen=True)
class Settings:
    """Every grid of section 6. Tests pass smaller grids; a run records the settings it used."""
    ks: tuple = (1, 3, 6, None)                                          # twins' calibration months; None is all
    alphas: tuple = tuple(round(1 - 0.1 * i, 1) for i in range(10))     # smoothing: 1.0, 0.9, ..., 0.1
    half_lives: tuple = (None, 12, 36)                                  # one-stage ridge and boosting, in months
    ridge_C: tuple = tuple(map(float, tune.C_GRID))                     # one-stage ridge (notebook 04)
    gb_grid: tuple = tuple(tune.GRID)                                   # boosting (notebook 03)
    gb_max_trees: int = tune.MAX_TREES
    stage1_C: tuple = tuple(map(float, tune.C_GRID))                    # two-stage ridge, stage 1
    stage2_alphas: tuple = tuple(0.0 if np.isinf(C) else float(1 / C) for C in tune.C_GRID)   # stage 2, strongest first
    stage1_folds: int = 5
    seed: int = 0

    def record(self):
        clean = lambda v: None if v is None or (isinstance(v, float) and np.isinf(v)) else v
        return {k: [clean(x) if not isinstance(x, dict) else x for x in v] if isinstance(v, tuple) else v
                for k, v in self.__dict__.items()}


def k_name(k):
    return 'kall' if k is None else f'k{k}'


def half_life_name(h):
    return 'hnone' if h is None else f'h{h}'


def alpha_name(a):
    return f'a{a:.1f}'


def twin(model, fs, k):
    return f'{model}/{INPUT_SETS[fs]}/{k_name(k)}'


def smoothing(alpha):
    return f'smooth/-/{alpha_name(alpha)}'


def one_stage(family, fs, half_life):
    return f'{family}/{INPUT_SETS[fs]}/{half_life_name(half_life)}'


def two_stage(fs, option):
    return f'two-stage/{INPUT_SETS[fs]}/{option}'


# Stage-2 options per input set: smoothing does not use M, so it appears only on D and D+P (section 6, candidate 6)
STAGE_TWO = {'demographics': ['smooth'], 'demographics_macro': ['level', 'change'],
             'demographics_politics': ['smooth', 'level', 'change'], 'demographics_macro_politics': ['level', 'change']}


def table(settings=Settings()):
    """One row per candidate in the plan's order: candidate, class (`twin` or `simple`), model, input set, setting."""
    rows = [('twin', m, fs, k_name(k), twin(m, fs, k)) for m in ['shift', 'scale'] for fs in FEATURE_SETS
            for k in settings.ks]
    rows += [('simple', 'smooth', None, alpha_name(a), smoothing(a)) for a in settings.alphas]
    rows += [('simple', f, fs, half_life_name(h), one_stage(f, fs, h)) for f in ['ridge', 'gb'] for fs in FEATURE_SETS
             for h in settings.half_lives]
    for fs, options in STAGE_TWO.items():
        for option in options:
            names = [f'smooth-{alpha_name(a)}' for a in settings.alphas] if option == 'smooth' else [option]
            rows += [('simple', 'two-stage', fs, n, two_stage(fs, n)) for n in names]
    out = pd.DataFrame(rows, columns=['cls', 'model', 'feature_set', 'setting', 'candidate'])
    out['input_set'] = out.feature_set.map(INPUT_SETS)
    return out[['candidate', 'cls', 'model', 'input_set', 'setting']].rename_axis('order').reset_index()
