"""Rows of the raw file for tests."""
import numpy as np

# One valid raw interview; tests override single cells
BASE = {'CASEID': 1, 'YYYYMM': 202001, 'WT': 1.0, 'ID': 10, 'IDPREV': np.nan, 'DATEPR': np.nan, 'METHOD': 1,
        'PAGO': 1, 'PEXP': 3, 'BUS12': 5, 'BUS5': 1, 'DUR': 1,
        'YTL5': 3, 'HOMEOWN': 1, 'INVEST': 1, 'AGE': 40, 'REGION': 1, 'SEX': 2, 'MARRY': 1, 'NUMKID': 0,
        'NUMADT': 2, 'EDUC': 4, 'POLAFF': 2}
