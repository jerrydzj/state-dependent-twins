"""Inputs for the realtime-benchmark tests: small grids and random twin probabilities."""
import numpy as np
import pandas as pd

from sdt.data.clean import OUTCOMES
from sdt.features import FEATURE_SETS
from sdt.twin.prompts import RAW

TINY = dict(ks=(1, None), alphas=(1.0, 0.5), half_lives=(None,), ridge_C=(1.0,),
            gb_grid=({'max_leaf_nodes': 4, 'min_samples_leaf': 20},), gb_max_trees=5, stage1_C=(1.0,),
            stage2_alphas=(1.0, 0.0), stage1_folds=2)


def random_raw(df, seed=0):
    """Raw option probabilities for every web interview of `df`, as a twin run's base stage would give them."""
    rng = np.random.default_rng(seed)
    web = df[df.web == 1].CASEID
    rows = []
    for fs in FEATURE_SETS:
        for item in OUTCOMES:
            p = rng.dirichlet([2, 2, 2], len(web)) * 0.95
            if item == 'DUR':
                p[:, 2] = np.nan
            rows.append(pd.DataFrame(p, columns=RAW).assign(stage='base', feature_set=fs, item=item,
                                                            CASEID=web.to_numpy()))
    return pd.concat(rows, ignore_index=True)
