"""Training and evaluation windows, rules W1-W3 in notebooks/02_mvp_plan.ipynb."""
import pandas as pd

from sdt.mvp.clean import FIRST_MONTH

HORIZON = pd.Timestamp('2024-07-23')    # Llama 3.1 release, the twin's knowledge horizon
TRAIN_START = FIRST_MONTH               # W1
TRAIN_END = 202407                      # W2: last interview window closing before HORIZON (closes 22 Jul 2024)
EVAL_START = 202408                     # W3: through the last archived month


def split(df):
    return df[df.YYYYMM.between(TRAIN_START, TRAIN_END)], df[df.YYYYMM >= EVAL_START]
