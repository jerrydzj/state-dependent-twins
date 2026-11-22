"""The mvp-twin study's training and evaluation windows, rules W1-W3 in notebooks/02_mvp_plan.ipynb."""
import pandas as pd

from sdt.data.clean import FIRST_MONTH

HORIZON = pd.Timestamp('2024-07-23')    # Llama 3.1 release, the twin's knowledge horizon
TRAIN_START = FIRST_MONTH               # W1
TRAIN_END = 202407                      # W2: last interview window closing before HORIZON (closes 22 Jul 2024)
EVAL_START = 202408                     # W3: through the last archived month, which was EVAL_END when
EVAL_END = 202602                       # prereg/mvp-twin fixed the window at these 19 months; later releases add months


def split(df):
    return df[df.YYYYMM.between(TRAIN_START, TRAIN_END)], df[df.YYYYMM.between(EVAL_START, EVAL_END)]
